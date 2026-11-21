"""Run the repository test Interface from the project-owned CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def run_tests(argv: Sequence[str], *, repository_root: Path | None = None) -> int:
    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    test_root = root / "tests"
    try:
        distribution("emperor")
    except PackageNotFoundError:
        print(
            "Error: install Emperor in the active environment before running tests.",
            file=sys.stderr,
        )
        print("Run: python -m pip install --no-deps -e .", file=sys.stderr)
        return 1

    environment = {
        **os.environ,
        "PYTHONSAFEPATH": "1",
        "PYTHONPATH": os.pathsep.join((str(root), str(test_root))),
    }
    arguments = list(argv)
    if not arguments:
        arguments = [str(test_root)]
    elif all(argument.startswith("-") for argument in arguments):
        arguments.append(str(test_root))
    if not any(argument == "-m" or argument.startswith("-m") for argument in arguments):
        arguments[:0] = ["-m", "not training"]
    command = [sys.executable, "-P", "-m", "pytest", *arguments]
    return subprocess.run(  # noqa: S603 - fixed interpreter and pytest module
        command,
        cwd=root,
        env=environment,
        check=False,
    ).returncode


__all__ = ["run_tests"]
