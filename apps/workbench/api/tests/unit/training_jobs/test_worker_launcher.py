from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor_workbench.training_jobs._containment._cgroup_v2 import (
    StrictCancellationUnavailable,
)
from emperor_workbench.training_jobs._containment._launcher import (
    TRAINING_LOGS_ROOT_ENV,
    TrainingWorkerLauncher,
)
from tests.support.training_jobs import FakeProcess, FakeRunner
from tests.unit.training_jobs._support import FakeCgroup, RecordingRunner


class TrainingWorkerLauncherTests(unittest.TestCase):
    def test_strict_launch_rejects_ready_marker_left_by_previous_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_root = root / "job-1"
            job_root.mkdir()
            process = FakeProcess(exit_code=125)
            (job_root / "cgroup.ready").write_text(f"{process.pid}\n", encoding="ascii")
            cgroup = FakeCgroup()
            cgroup.processes = False
            runner = FakeRunner(process)
            launcher = TrainingWorkerLauncher(
                cwd=root,
                runner=runner,
                cancellation_mode="strict-cgroup",
                cgroup_manager=SimpleNamespace(
                    create_job_cgroup=lambda _job_id: cgroup
                ),
            )

            with self.assertRaisesRegex(
                StrictCancellationUnavailable, "before the worker joined its cgroup"
            ):
                launcher.launch(job_root=job_root, payload={"id": "job-1"})

            self.assertEqual(len(runner.commands), 1)
            self.assertTrue(cgroup.cleaned)

    def test_launcher_writes_payload_and_starts_worker_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = RecordingRunner()
            launcher = TrainingWorkerLauncher(cwd=root, runner=runner)
            relative_logs_root = Path("relative-worker-logs")
            launch = launcher.launch(
                job_root=root / "job-1",
                payload={"id": "job-1"},
                logs_root=relative_logs_root,
            )

            payload_path = root / "job-1" / "payload.json"
            progress_path = root / "job-1" / "progress.jsonl"
            log_path = root / "job-1" / "training.log"

            self.assertEqual(json.loads(payload_path.read_text()), {"id": "job-1"})
            self.assertEqual(
                launch.command,
                [
                    sys.executable,
                    "-m",
                    "emperor_workbench.training_jobs.worker",
                    "--payload",
                    str(payload_path),
                    "--progress",
                    str(progress_path),
                ],
            )
            self.assertIs(launch.process, runner.process)
            self.assertEqual(runner.calls[0]["log_path"], log_path)
            self.assertEqual(
                runner.calls[0]["env"][TRAINING_LOGS_ROOT_ENV],
                str(relative_logs_root.resolve()),
            )
            self.assertEqual(
                launcher._wrap_strict_cgroup_command(
                    launcher.build_command(
                        Path("{payload}"),
                        Path("{progress}"),
                    ),
                    cgroup=SimpleNamespace(cgroup_path="{cgroup}"),
                    ready_path=Path("{ready}"),
                ),
                [
                    sys.executable,
                    "-m",
                    "emperor_workbench.training_jobs.cgroup_worker",
                    "--cgroup",
                    "{cgroup}",
                    "--ready",
                    "{ready}",
                    "--",
                    sys.executable,
                    "-m",
                    "emperor_workbench.training_jobs.worker",
                    "--payload",
                    "{payload}",
                    "--progress",
                    "{progress}",
                ],
            )


if __name__ == "__main__":
    unittest.main()
