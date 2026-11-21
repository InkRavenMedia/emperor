from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.settings import get_workbench_api_settings
from tests.support.model_packages import install_project_adapter


@pytest.fixture(autouse=True)
def isolated_workbench_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Give every app created by this test its own writable roots."""
    for name in ("logs", "snapshots", "state"):
        monkeypatch.setenv(f"WORKBENCH_API_{name.upper()}_ROOT", str(tmp_path / name))
    get_workbench_api_settings.cache_clear()
    try:
        yield
    finally:
        get_workbench_api_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def project_adapter_fixture() -> Iterator[ProjectAdapterClient]:
    with ProjectAdapterClient() as client:
        install_project_adapter(client)
        try:
            yield client
        finally:
            install_project_adapter(None)
