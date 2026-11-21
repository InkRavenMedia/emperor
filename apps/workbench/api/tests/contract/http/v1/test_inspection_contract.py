from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from emperor_workbench.api import create_app
from emperor_workbench.api.v1.inspection import InspectResponse
from tests.support import lifespan_client

TESTS_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = TESTS_ROOT.parents[3]
FIXTURE_PATH = TESTS_ROOT / "fixtures" / "inspection_contract_v1.json"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode()


def _digest(value: Any) -> tuple[str, int]:
    payload = _canonical_json(value)
    return hashlib.sha256(payload).hexdigest(), len(payload)


class InspectionContractFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        self.runner = asyncio.Runner()
        self.addCleanup(self.runner.close)
        connection = lifespan_client(create_app())
        self.client = self.runner.run(connection.__aenter__())
        self.addCleanup(self.runner.run, connection.__aexit__(None, None, None))

    def _get(self, path, **params):
        response = self.runner.run(self.client.get(path, params=params))
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _inspect(self, model_id, preset, **options):
        model_type, model = model_id.split("/")
        return self.runner.run(
            self.client.post(
                "/inspect",
                json={
                    "modelType": model_type,
                    "model": model,
                    "preset": preset,
                    **options,
                },
            )
        )

    def test_graph_http_payloads_match_ordered_goldens(self) -> None:
        for name, expected in self.fixture["graphs"].items():
            with self.subTest(case=name):
                response = self._inspect(
                    expected["model"],
                    expected["preset"],
                    dataset=expected["dataset"],
                    experimentTask=expected["experimentTask"],
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()

                self.assertEqual(
                    _digest(payload),
                    (expected["sha256"], expected["bytes"]),
                )
                self.assertEqual(len(payload["nodes"]), expected["nodeCount"])
                self.assertEqual(len(payload["edges"]), expected["edgeCount"])
                self.assertEqual(
                    list(payload),
                    [
                        "modelType",
                        "model",
                        "preset",
                        "parameterCount",
                        "parameterSizeBytes",
                        "nodes",
                        "edges",
                    ],
                )

    def test_configuration_and_search_payloads_match_ordered_goldens(self) -> None:
        for name, expected in self.fixture["schemas"].items():
            with self.subTest(case=name):
                base = f"/models/{expected['model']}"
                schema = self._get(f"{base}/config-schema", preset=expected["preset"])
                search = self._get(
                    f"{base}/search-space",
                    preset=expected["preset"],
                    presets=",".join(expected["presets"] or []),
                )

                self.assertEqual(
                    _digest(schema),
                    (expected["configSha256"], expected["configBytes"]),
                )
                self.assertEqual(len(schema["fields"]), expected["fieldCount"])
                self.assertEqual(
                    _digest(search),
                    (expected["searchSha256"], expected["searchBytes"]),
                )
                self.assertEqual(len(search["axes"]), expected["axisCount"])

    def test_catalog_and_model_metadata_payloads_match_goldens(self) -> None:
        catalog_digest, catalog_bytes, model_count = self.fixture["catalog"]
        models = [
            f"{entry['modelType']}/{entry['model']}"
            for entry in self._get("/models")["models"]
        ]
        self.assertEqual(_digest(models), (catalog_digest, catalog_bytes))
        self.assertEqual(len(models), model_count)

        for model_name, expected in self.fixture["discovery"].items():
            with self.subTest(model=model_name):
                base = f"/models/{model_name}"
                datasets = self._get(f"{base}/datasets")
                payload = {
                    "presets": self._get(f"{base}/presets")["presets"],
                    "datasets": {
                        key: datasets[key]
                        for key in ("defaultExperimentTask", "datasetGroups")
                    },
                    "monitors": self._get(f"{base}/monitors")["monitors"],
                }
                digest, byte_count, presets, groups, monitors = expected
                self.assertEqual(_digest(payload), (digest, byte_count))
                self.assertEqual(len(payload["presets"]), presets)
                self.assertEqual(len(payload["datasets"]["datasetGroups"]), groups)
                self.assertEqual(len(payload["monitors"]), monitors)

    def test_inspection_failures_preserve_detail_and_http_status_goldens(self) -> None:
        cases = {
            "unknown-model": ("unknown/model", "baseline", {}),
            "unknown-override": (
                "linears/linear",
                "baseline",
                {"overrides": {"NO_SUCH_FIELD": "1"}},
            ),
            "malformed-override": (
                "linears/linear",
                "baseline",
                {"overrides": {"HIDDEN_DIM": "not-an-int"}},
            ),
            "locked-override": (
                "linears/linear",
                "gating",
                {"overrides": {"STACK_GATE_FLAG": "false"}},
            ),
            "incompatible-dataset": (
                "linears/linear",
                "baseline",
                {"dataset": "WikiText2"},
            ),
            "incompatible-task": (
                "linears/linear",
                "baseline",
                {"experimentTask": "causal-language-modeling"},
            ),
            "path-dataset": ("linears/linear", "baseline", {"dataset": "../Mnist"}),
        }
        for name, (model, preset, options) in cases.items():
            with self.subTest(case=name):
                response = self._inspect(model, preset, **options)
                detail, http_status = self.fixture["errors"][name]
                self.assertEqual(response.status_code, http_status, response.text)
                self.assertEqual(response.json()["detail"], detail)

    def test_cli_json_and_workbench_graph_are_equivalent(self) -> None:
        completed = self._run_cli(
            "--model-type",
            "linears",
            "--model",
            "linear",
            "--preset",
            "baseline",
            "--datasets",
            "Mnist",
            "--format",
            "json",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        cli_graph = InspectResponse.model_validate_json(completed.stdout)
        response = self._inspect("linears/linear", "baseline", dataset="Mnist")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            cli_graph.model_dump(mode="json"),
            response.json(),
        )

    def test_cli_error_and_exit_contracts_are_stable(self) -> None:
        cases = {
            "unknown-model": (
                "--model-type",
                "unknown",
                "--model",
                "model",
                "--preset",
                "baseline",
            ),
            "invalid-preset": (
                "--model-type",
                "linears",
                "--model",
                "linear",
                "--preset",
                "nope",
                "--datasets",
                "Mnist",
            ),
            "locked-override": (
                "--model-type",
                "linears",
                "--model",
                "linear",
                "--preset",
                "gating",
                "--datasets",
                "Mnist",
                "--config",
                "--stack-gate-flag",
                "false",
            ),
            "pre-inspection-malformed-override": (
                "--model-type",
                "linears",
                "--model",
                "linear",
                "--preset",
                "baseline",
                "--datasets",
                "Mnist",
                "--config",
                "--hidden-dim",
                "not-an-int",
            ),
        }

        for name, arguments in cases.items():
            with self.subTest(case=name):
                completed = self._run_cli(*arguments)
                expected_code, expected_last_line, expected_traceback = self.fixture[
                    "cliErrors"
                ][name]
                self.assertEqual(completed.returncode, expected_code)
                self.assertEqual(completed.stdout, "")
                self.assertEqual(
                    completed.stderr.strip().splitlines()[-1],
                    expected_last_line,
                )
                self.assertEqual(
                    "Traceback (most recent call last):" in completed.stderr,
                    expected_traceback,
                )

    @staticmethod
    def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "models.project_cli", "inspect", *arguments],
            cwd=REPOSITORY_ROOT,
            env={
                **os.environ,
                "MPLCONFIGDIR": str(Path(tempfile.gettempdir()) / "matplotlib"),
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )


if __name__ == "__main__":
    unittest.main()
