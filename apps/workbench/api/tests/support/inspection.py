from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.inspection import InspectionRequest, ParsedOverrides
from model_runtime.inspection.model_graph import inspect_model_graph
from model_runtime.packages import is_safe_model_identity

from emperor_workbench.api.v1.inspection import (
    graph_edge_payload,
    graph_node_payload,
    inspection_response,
)
from emperor_workbench.api.v1.model_packages import (
    datasets_response,
    monitors_response,
    presets_response,
    search_space_response,
)
from emperor_workbench.inspection import (
    InProcessInspectionExecutor,
    InspectionFailure,
    InspectionService,
)
from emperor_workbench.model_packages import (
    ModelPackageCatalog,
    ModelPackageFailure,
    SelectedModelPackage,
)
from tests.support.model_packages import project_adapter_client


def discover_models() -> list[str]:
    return [
        identity.catalog_key
        for identity in ModelPackageCatalog(project_adapter_client()).identities()
    ]


def _selected(model_name: str) -> SelectedModelPackage:
    parts = model_name.split("/")
    if len(parts) != 2 or not is_safe_model_identity(parts[0], parts[1]):
        raise InspectionFailure(f"Invalid model name: {model_name!r}")
    try:
        return ModelPackageCatalog(project_adapter_client()).select(model_name)
    except ModelPackageFailure as exc:
        raise InspectionFailure(exc.detail, kind=exc.kind) from exc


def search_space_schema(
    model_name: str,
    preset_name: str | None = None,
    preset_names: list[str] | None = None,
) -> dict[str, Any]:
    search_space = _selected(model_name).search_space(
        preset_name,
        preset_names,
    )
    return search_space_response(search_space).model_dump()


def list_model_presets(model_name: str) -> list[dict[str, str]]:
    selected = _selected(model_name)
    return presets_response(selected.identity, selected.metadata()).model_dump()[
        "presets"
    ]


def list_model_datasets(model_name: str) -> dict[str, Any]:
    selected = _selected(model_name)
    return datasets_response(selected.identity, selected.metadata()).model_dump()


def list_model_monitors(model_name: str) -> list[dict[str, Any]]:
    selected = _selected(model_name)
    return monitors_response(selected.identity, selected.metadata()).model_dump()[
        "monitors"
    ]


def inspect_model(
    model_name: str,
    preset_name: str,
    overrides: Mapping[str, Any] | None = None,
    dataset: str | None = None,
    experiment_task: str | None = None,
    *,
    parsed_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = _selected(model_name)
    executor = InProcessInspectionExecutor()
    if parsed_overrides is not None:
        result = executor.inspect(
            selected,
            InspectionRequest(
                preset=preset_name,
                overrides=ParsedOverrides(parsed_overrides),
                dataset=dataset,
                experiment_task=experiment_task,
            ),
        )
    else:
        result = InspectionService(executor).inspect(
            selected,
            preset=preset_name,
            overrides=overrides or {},
            dataset=dataset,
            experiment_task=experiment_task,
        )
    return inspection_response(result).model_dump(mode="json")


def serialize_graph(
    module: object,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    graph = inspect_model_graph(module)
    return (
        [graph_node_payload(node) for node in graph.nodes],
        [graph_edge_payload(edge) for edge in graph.edges],
    )


__all__ = [
    "discover_models",
    "inspect_model",
    "list_model_datasets",
    "list_model_monitors",
    "list_model_presets",
    "search_space_schema",
    "serialize_graph",
]
