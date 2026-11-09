from __future__ import annotations

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from emperor_workbench.run_history import LogRun
from tests.unit.run_history._support import log_run_scanner


def _write_run(logs_root: Path, experiment: str, run_name: str) -> None:
    logs_root.joinpath(
        experiment,
        "linears",
        "linear",
        "BASELINE",
        "Mnist",
        run_name,
        "version_0",
    ).mkdir(parents=True)


class RunCatalogIndexTests(unittest.TestCase):
    def test_new_scanner_loads_persistent_catalog_without_recursive_rescan(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            state_root = root / "state"
            _write_run(logs_root, "experiment", "run_20260712_010203")
            first = log_run_scanner(
                logs_root=logs_root,
                state_root=state_root,
                cache_ttl_seconds=3600,
            )
            self.assertEqual(len(first.list_runs(result_projection="none")), 1)
            catalog_payload = json.loads(
                (state_root / "catalogs" / "run-history.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                set(catalog_payload["entries"][0]),
                {
                    "id",
                    "group",
                    "experiment",
                    "model",
                    "preset",
                    "dataset",
                    "runName",
                    "timestamp",
                    "version",
                    "relativePath",
                    "hasResult",
                    "eventFileCount",
                    "checkpointCount",
                    "hasHparams",
                },
            )

            restarted = log_run_scanner(
                logs_root=logs_root,
                state_root=state_root,
                cache_ttl_seconds=3600,
            )
            with patch.object(
                restarted,
                "_version_dirs_and_fingerprint",
                side_effect=AssertionError("persistent catalog must be reused"),
            ):
                runs = restarted.list_runs(result_projection="none")

            self.assertEqual(len(runs), 1)
            self.assertTrue((state_root / "catalogs" / "run-history.json").is_file())

    def test_delayed_scan_preserves_newer_catalog(self) -> None:
        for delay_at, invalidate in (
            ("publication", True),
            ("publication", False),
            ("observation", False),
        ):
            with self.subTest(delay_at=delay_at, invalidate=invalidate):
                self._assert_newer_catalog_preserved(delay_at, invalidate=invalidate)

    def _assert_newer_catalog_preserved(
        self, delay_at: str, *, invalidate: bool
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            state_root = root / "state"
            experiment = "experiment"
            old_name = "run_20260712_010203"
            new_name = "run_20260712_020304"
            _write_run(logs_root, experiment, old_name)
            scanner = log_run_scanner(
                logs_root=logs_root, state_root=state_root, cache_ttl_seconds=0
            )
            old_scan_ready = Event()
            release_old_scan = Event()
            publish = scanner._publish_persistent_catalog
            observe = scanner._version_dirs_and_fingerprint

            def wait_for_newer_scan() -> None:
                if not old_scan_ready.is_set():
                    old_scan_ready.set()
                    self.assertTrue(release_old_scan.wait(timeout=5))

            def delay_publication(runs: list[LogRun], **generations: int) -> None:
                wait_for_newer_scan()
                publish(runs, **generations)

            def delay_observation(root: Path):
                result = observe(root)
                wait_for_newer_scan()
                return result

            if delay_at == "publication":
                delay = patch.object(
                    scanner, "_publish_persistent_catalog", delay_publication
                )
            else:
                delay = patch.object(
                    scanner, "_version_dirs_and_fingerprint", delay_observation
                )

            with delay, ThreadPoolExecutor(max_workers=1) as executor:
                old_scan = executor.submit(scanner.list_runs, result_projection="none")
                try:
                    self.assertTrue(old_scan_ready.wait(timeout=5))
                    _write_run(logs_root, experiment, new_name)
                    if invalidate:
                        scanner.invalidate_experiment(experiment)
                    newer_runs = scanner.list_runs(result_projection="none")
                    self.assertEqual(
                        [run.run_name for run in newer_runs], [new_name, old_name]
                    )
                finally:
                    release_old_scan.set()
                self.assertEqual(
                    [run.run_name for run in old_scan.result(timeout=5)], [old_name]
                )

            restarted = log_run_scanner(logs_root=logs_root, state_root=state_root)
            with patch.object(
                restarted,
                "_version_dirs_and_fingerprint",
                side_effect=AssertionError("persistent catalog must be reused"),
            ):
                runs = restarted.list_runs(result_projection="none")

            self.assertEqual([run.run_name for run in runs], [new_name, old_name])

    def test_scanner_rejects_flat_model_identity_in_persistent_catalog(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            state_root = root / "state"
            catalog_path = state_root / "catalogs" / "run-history.json"
            _write_run(logs_root, "experiment", "run_20260712_010203")
            first = log_run_scanner(
                logs_root=logs_root,
                state_root=state_root,
                cache_ttl_seconds=3600,
            )
            self.assertEqual(len(first.list_runs(result_projection="none")), 1)

            catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            catalog_payload["entries"][0]["model"] = "linear"
            catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")

            restarted = log_run_scanner(
                logs_root=logs_root,
                state_root=state_root,
                cache_ttl_seconds=3600,
            )
            with patch.object(
                restarted,
                "_version_dirs_and_fingerprint",
                wraps=restarted._version_dirs_and_fingerprint,
            ) as rescan:
                runs = restarted.list_runs(result_projection="none")

            self.assertEqual([run.model for run in runs], ["linears/linear"])
            rescan.assert_called_once()
            rewritten = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(rewritten["entries"][0]["model"], "linears/linear")

    def test_newer_fingerprint_match_preserves_cached_catalog(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            state_root = root / "state"
            _write_run(logs_root, "experiment", "run_20260712_010203")
            scanner = log_run_scanner(
                logs_root=logs_root, state_root=state_root, cache_ttl_seconds=0
            )
            expected = scanner.list_runs(result_projection="none")
            self.assertEqual(len(expected), 1)
            run_dir = expected[0].path
            parked_run = root / "parked-run"
            run_dir.rename(parked_run)
            old_scan_ready = Event()
            release_old_scan = Event()
            observe = scanner._version_dirs_and_fingerprint

            def delay_first_observation(root: Path):
                result = observe(root)
                if not old_scan_ready.is_set():
                    old_scan_ready.set()
                    self.assertTrue(release_old_scan.wait(timeout=5))
                return result

            with (
                patch.object(
                    scanner, "_version_dirs_and_fingerprint", delay_first_observation
                ),
                ThreadPoolExecutor(max_workers=1) as executor,
            ):
                old_scan = executor.submit(scanner.list_runs, result_projection="none")
                try:
                    self.assertTrue(old_scan_ready.wait(timeout=5))
                    parked_run.rename(run_dir)
                    self.assertEqual(
                        scanner.list_runs(result_projection="none"), expected
                    )
                finally:
                    release_old_scan.set()
                self.assertEqual(old_scan.result(timeout=5), [])

            restarted = log_run_scanner(logs_root=logs_root, state_root=state_root)
            with patch.object(
                restarted,
                "_version_dirs_and_fingerprint",
                side_effect=AssertionError("persistent catalog must be reused"),
            ):
                runs = restarted.list_runs(result_projection="none")
            self.assertEqual(runs, expected)

    def test_due_reconciliation_discovers_external_run_additions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            scanner = log_run_scanner(
                logs_root=logs_root,
                state_root=root / "state",
                cache_ttl_seconds=0,
            )
            _write_run(logs_root, "experiment", "run_20260712_010203")
            self.assertEqual(len(scanner.list_runs(result_projection="none")), 1)

            _write_run(logs_root, "experiment", "run_20260712_020304")

            self.assertEqual(len(scanner.list_runs(result_projection="none")), 2)


if __name__ == "__main__":
    unittest.main()
