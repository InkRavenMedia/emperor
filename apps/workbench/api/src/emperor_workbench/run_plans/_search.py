from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.inspection import SearchAxis
from model_runtime.packages import (
    config_key_to_model_param,
    normalize_key,
    serialize_config_value,
)
from model_runtime.runs import (
    RunPlan,
    SearchAxisSelection,
    SearchMode,
    SearchOptionError,
    SearchSpec,
    resolve_search_options,
)

from emperor_workbench.failures import FailureKind
from emperor_workbench.model_packages import ModelPackageFailure, SelectedModelPackage
from emperor_workbench.project_adapter import (
    ModelPackageReference,
    ProjectAdapterFailure,
)
from emperor_workbench.run_plans._errors import RunPlanFailure
from emperor_workbench.run_plans._limits import (
    MAX_TRAINING_PLANNED_RUNS,
    MAX_TRAINING_SEARCH_AXES,
    MAX_TRAINING_SEARCH_AXIS_VALUES,
)
from emperor_workbench.run_plans._records import TrainingSearch


def _parse_search_value(
    package: ModelPackageReference,
    axis: SearchAxis,
    raw_value: Any,
) -> Any:
    if raw_value is None:
        return None
    try:
        return package.client.call(
            "parse_search_value",
            {
                "model_id": package.catalog_key,
                "search_key": axis.search_key,
                "value": raw_value,
            },
        )
    except ProjectAdapterFailure as exc:
        if exc.kind != FailureKind.INVALID:
            raise RunPlanFailure(exc.detail, kind=exc.kind) from exc
        raise RunPlanFailure(
            f"Invalid search value for axis '{axis.key}': {raw_value!r}. {exc}"
        ) from exc


def _deduplicate(values: list[Any]) -> tuple[Any, ...]:
    deduplicated: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return tuple(deduplicated)


def training_search_options(
    mode: object, random_samples: object, *, path: str = "search"
) -> tuple[SearchMode, int | None]:
    try:
        parsed_mode, count = resolve_search_options(mode, random_samples)
    except SearchOptionError as exc:
        field = "randomSamples" if exc.field == "random_samples" else "mode"
        reason = str(exc).removeprefix(f"{exc.field} ")
        raise ValueError(f"{path}.{field} {reason}") from exc
    if count is not None and count > MAX_TRAINING_PLANNED_RUNS:
        raise ValueError(
            f"{path}.randomSamples must be an integer between 1 and "
            f"{MAX_TRAINING_PLANNED_RUNS}."
        )
    return parsed_mode, count


def materialized_search_options(
    search: Mapping[str, Any], *, path: str = "search"
) -> tuple[SearchMode, int | None]:
    mode, count = training_search_options(
        search.get("mode"), search.get("randomSamples"), path=path
    )
    if mode == "random" and search.get("randomSamples") is None:
        raise ValueError(
            f"{path}.randomSamples must contain the materialized random sample count."
        )
    if mode == "grid" and "randomSamples" in search:
        raise ValueError(f"{path}.randomSamples cannot be stored for grid search.")
    return mode, count


def adapt_search(
    package: ModelPackageReference,
    preset_name: str,
    search: TrainingSearch | None,
) -> tuple[SearchSpec | None, set[str]]:
    if search is None:
        return None, set()
    try:
        mode, random_samples = training_search_options(
            search.mode, search.random_samples
        )
    except ValueError as exc:
        raise RunPlanFailure(str(exc)) from exc
    values_payload = search.values
    if not values_payload:
        raise RunPlanFailure("Training search requires at least one selected axis.")
    if len(values_payload) > MAX_TRAINING_SEARCH_AXES:
        raise RunPlanFailure(
            f"Training search accepts at most {MAX_TRAINING_SEARCH_AXES} selected axes."
        )

    try:
        semantic_search = SelectedModelPackage(package).search_space(preset_name)
    except ModelPackageFailure as exc:
        raise RunPlanFailure(exc.detail, kind=exc.kind) from exc
    axes_by_key = {normalize_key(axis.key): axis for axis in semantic_search.axes}
    ordered_keys: list[str] = []
    selections: dict[str, SearchAxisSelection] = {}
    model_params: set[str] = set()
    for raw_key, raw_values in values_payload.items():
        axis = axes_by_key.get(normalize_key(str(raw_key)))
        if axis is None:
            raise RunPlanFailure(f"Unknown search axis '{raw_key}'.")
        if axis.locked:
            raise RunPlanFailure(
                f"Search axis '{axis.key}' is locked by preset '{preset_name}'."
            )
        if not isinstance(raw_values, list) or not raw_values:
            raise RunPlanFailure(
                f"Search axis '{axis.key}' requires at least one selected value."
            )
        if len(raw_values) > MAX_TRAINING_SEARCH_AXIS_VALUES:
            raise RunPlanFailure(
                f"Search axis '{axis.key}' accepts at most "
                f"{MAX_TRAINING_SEARCH_AXIS_VALUES} selected values."
            )

        serialized_values = _deduplicate(
            [_parse_search_value(package, axis, raw_value) for raw_value in raw_values]
        )
        allowed_values = {serialize_config_value(value) for value in axis.values}
        invalid_values = [
            value for value in serialized_values if value not in allowed_values
        ]
        if invalid_values:
            raise RunPlanFailure(
                f"Search axis '{axis.key}' received values outside its "
                f"search space: {invalid_values}."
            )

        canonical_key = normalize_key(axis.key)
        if canonical_key not in selections:
            ordered_keys.append(canonical_key)
        selections[canonical_key] = SearchAxisSelection(
            key=axis.key,
            values=serialized_values,
        )
        model_params.add(config_key_to_model_param(axis.key))

    return (
        SearchSpec(
            mode=mode,
            axes=tuple(selections[key] for key in ordered_keys),
            random_samples=random_samples,
        ),
        model_params,
    )


def search_from_spec(search: SearchSpec | None) -> TrainingSearch | None:
    if search is None:
        return None
    return TrainingSearch(
        mode=search.mode,
        values={axis.key: list(axis.values or ()) for axis in (search.axes or ())},
        random_samples=search.random_samples,
        custom_value_axes=tuple(
            axis.key for axis in (search.axes or ()) if axis.allow_custom_values
        ),
    )


def preset_searches_from_plan(
    plan: RunPlan,
) -> dict[str, TrainingSearch | None]:
    return {
        entry.preset: search_from_spec(entry.search) for entry in plan.preset_searches
    }


__all__ = ["adapt_search", "preset_searches_from_plan", "search_from_spec"]
