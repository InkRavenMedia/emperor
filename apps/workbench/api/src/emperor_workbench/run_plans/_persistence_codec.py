from __future__ import annotations

import copy
from collections.abc import Mapping
from math import isfinite
from typing import Any, cast

from model_runtime.packages import normalize_key

from emperor_workbench.model_packages import ModelPackageIdentity
from emperor_workbench.run_plans._limits import (
    MAX_TRAINING_SEARCH_AXES,
    MAX_TRAINING_SEARCH_AXIS_VALUES,
)
from emperor_workbench.run_plans._records import (
    ConfigSnapshotRevision,
    TrainingCommandsView,
    TrainingRunChangeView,
    TrainingRunPlanSummaryView,
    TrainingRunPlanView,
    TrainingRunView,
    TrainingSearch,
)
from emperor_workbench.run_plans._search import (
    materialized_search_options,
    training_search_options,
)

_RUN_STATUSES = {
    "Pending",
    "Running",
    "Completed",
    "Failed",
    "Cancelled",
    "Skipped",
}
_CHANGE_SOURCES = {"override", "search"}


def _require_fields(
    payload: Mapping[str, Any], fields: tuple[str, ...], *, path: str
) -> None:
    for name in fields:
        if name not in payload:
            raise ValueError(f"{path}.{name} is required.")


def _validate_config_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if isfinite(value):
            return
        raise ValueError(f"{path} must be finite.")
    raise ValueError(f"{path} must be a scalar config value.")


def _validate_config_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object.")
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{path} must use string keys.")
        _validate_config_value(item, path=f"{path}.{key}")
    return value


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if isfinite(value):
            return
        raise ValueError(f"{path} must be finite.")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} must use string keys.")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} must be JSON-compatible.")


def _validate_string_list(
    value: object,
    *,
    path: str,
    allow_empty: bool = True,
    nonempty_items: bool = False,
    unique: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        requirement = "a non-empty list" if not allow_empty else "a list"
        raise ValueError(f"{path} must be {requirement}.")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{path} must contain strings.")
    if nonempty_items and any(not item for item in value):
        raise ValueError(f"{path} must contain non-empty strings.")
    if unique and len(value) != len(set(value)):
        raise ValueError(f"{path} must not contain duplicate names.")
    return value


def _validate_integer(value: object, *, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")


def _validate_search_payload(
    payload: object,
    *,
    path: str,
    allow_empty: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must be an object.")
    materialized_search_options(payload, path=path)
    values = payload.get("values")
    if not isinstance(values, Mapping) or (not values and not allow_empty):
        raise ValueError(f"{path}.values must be a non-empty object.")
    if len(values) > MAX_TRAINING_SEARCH_AXES:
        raise ValueError(
            f"{path}.values accepts at most {MAX_TRAINING_SEARCH_AXES} axes."
        )
    normalized_axes: set[str] = set()
    for key, options in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{path}.values must use non-empty string keys.")
        normalized_key = normalize_key(key)
        if normalized_key in normalized_axes:
            raise ValueError(f"{path}.values contains duplicate axis '{key}'.")
        normalized_axes.add(normalized_key)
        if not isinstance(options, list) or not options:
            raise ValueError(f"{path}.values.{key} must be a non-empty list.")
        if len(options) > MAX_TRAINING_SEARCH_AXIS_VALUES:
            raise ValueError(
                f"{path}.values.{key} accepts at most "
                f"{MAX_TRAINING_SEARCH_AXIS_VALUES} values."
            )
        for index, option in enumerate(options):
            _validate_config_value(
                option,
                path=f"{path}.values.{key}[{index}]",
            )
    custom_value_axes = _validate_string_list(
        payload.get("customValueAxes", []),
        path=f"{path}.customValueAxes",
        nonempty_items=True,
    )
    normalized_custom_axes: set[str] = set()
    for key in custom_value_axes:
        normalized_key = normalize_key(key)
        if normalized_key in normalized_custom_axes:
            raise ValueError(f"{path}.customValueAxes contains duplicate axis '{key}'.")
        if normalized_key not in normalized_axes:
            raise ValueError(f"{path}.customValueAxes contains unknown axis '{key}'.")
        normalized_custom_axes.add(normalized_key)
    return payload


def _validate_preset_searches_payload(
    payload: object,
    *,
    presets: list[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must be an object.")
    if set(payload) != set(presets):
        raise ValueError(f"{path} must exactly match the Run Plan presets.")
    for preset in presets:
        search = payload[preset]
        if search is not None:
            _validate_search_payload(
                search,
                path=f"{path}.{preset}",
                allow_empty=True,
            )
    return payload


def _validate_run_change(payload: object, *, path: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must be an object.")
    for field in ("key", "label"):
        value = payload.get(field)
        if not isinstance(value, str):
            raise ValueError(f"{path}.{field} must be a string.")
    _require_fields(payload, ("value",), path=path)
    _validate_config_value(payload["value"], path=f"{path}.value")
    if payload.get("source") not in _CHANGE_SOURCES:
        raise ValueError(f"{path}.source is invalid.")


def _validate_run_payload(payload: object, *, path: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must be an object.")
    if "command" in payload:
        raise ValueError(f"{path}.command is retired; use commandArgv and commands.")
    _require_fields(
        payload,
        (
            "experimentTask",
            "snapshotId",
            "snapshotName",
            "logDir",
            "error",
            "errorTraceback",
            "index",
            "totalEpochs",
            "currentEpoch",
            "status",
            "changes",
            "overrides",
            "metrics",
        ),
        path=path,
    )
    for field in ("id", "preset", "dataset"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}.{field} must be a non-empty string.")
    if not isinstance(payload["experimentTask"], str):
        raise ValueError(f"{path}.experimentTask must be a string.")
    for field in (
        "logDir",
        "error",
        "errorTraceback",
        "snapshotId",
        "snapshotName",
    ):
        if payload[field] is not None and not isinstance(payload[field], str):
            raise ValueError(f"{path}.{field} must be a string or null.")
    for field in ("index", "totalEpochs", "currentEpoch"):
        _validate_integer(payload[field], path=f"{path}.{field}")
    if payload["status"] not in _RUN_STATUSES:
        raise ValueError(f"{path}.status is invalid.")
    changes = payload["changes"]
    if not isinstance(changes, list):
        raise ValueError(f"{path}.changes must be a list.")
    for index, change in enumerate(changes):
        _validate_run_change(change, path=f"{path}.changes[{index}]")
    _validate_config_mapping(payload["overrides"], path=f"{path}.overrides")
    _validate_string_list(
        payload.get("commandArgv"),
        path=f"{path}.commandArgv",
        allow_empty=False,
        nonempty_items=True,
    )
    commands = payload.get("commands")
    if not isinstance(commands, Mapping):
        raise ValueError(f"{path}.commands must be an object.")
    for shell in ("posix", "powershell"):
        if not isinstance(commands.get(shell), str) or not commands[shell]:
            raise ValueError(f"{path}.commands.{shell} must be a non-empty string.")
    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{path}.metrics must be an object.")
    _validate_json_value(metrics, path=f"{path}.metrics")


def _validate_summary_payload(payload: object) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Persisted Run Plan summary must be an object.")
    for field in (
        "totalRuns",
        "completedRuns",
        "runningRuns",
        "pendingRuns",
        "failedRuns",
        "cancelledRuns",
        "skippedRuns",
        "totalEpochs",
        "completedEpochs",
        "remainingEpochs",
    ):
        _validate_integer(payload.get(field), path=f"runPlan.summary.{field}")


def _validate_plan_payload(payload: Mapping[str, Any]) -> None:
    _require_fields(
        payload,
        (
            "search",
            "presetSearches",
            "snapshotRevisions",
            "isRandomSearch",
            "experimentTask",
        ),
        path="runPlan",
    )
    if ModelPackageIdentity.from_mapping(payload) is None:
        raise ValueError("Persisted Run Plan model identity is invalid.")
    preset = payload.get("preset")
    if not isinstance(preset, str) or not preset:
        raise ValueError("Persisted Run Plan preset must be a non-empty string.")
    presets = _validate_string_list(
        payload.get("presets"),
        path="runPlan.presets",
        allow_empty=False,
        nonempty_items=True,
        unique=True,
    )
    _validate_string_list(
        payload.get("datasets"),
        path="runPlan.datasets",
        allow_empty=False,
        nonempty_items=True,
        unique=True,
    )
    if not isinstance(payload["experimentTask"], str):
        raise ValueError("runPlan.experimentTask must be a string.")
    _validate_config_mapping(payload.get("overrides"), path="runPlan.overrides")
    if payload.get("search") is not None:
        _validate_search_payload(payload["search"], path="runPlan.search")
    _validate_preset_searches_payload(
        payload["presetSearches"],
        presets=presets,
        path="runPlan.presetSearches",
    )
    if not isinstance(payload.get("logFolder"), str):
        raise ValueError("runPlan.logFolder must be a string.")
    if not isinstance(payload["isRandomSearch"], bool):
        raise ValueError("runPlan.isRandomSearch must be a boolean.")
    run_payloads = payload.get("runs")
    if not isinstance(run_payloads, list):
        raise ValueError("Persisted Run Plan rows must be a list of objects.")
    seen_ids: set[str] = set()
    for index, run_payload in enumerate(run_payloads):
        _validate_run_payload(run_payload, path=f"runPlan.runs[{index}]")
        assert isinstance(run_payload, Mapping)
        run_id = cast(str, run_payload["id"])
        if run_id in seen_ids:
            raise ValueError(
                f"Persisted Run Plan contains duplicate run id '{run_id}'."
            )
        seen_ids.add(run_id)
    _validate_summary_payload(payload.get("summary"))
    _snapshot_revisions_from_payload(payload.get("snapshotRevisions"))


def _payload_model_id(payload: Mapping[str, Any]) -> str:
    identity = ModelPackageIdentity.from_mapping(payload)
    assert identity is not None
    return identity.catalog_key


def _model_identity(model_id: str) -> dict[str, str]:
    identity = ModelPackageIdentity.from_id(model_id)
    return {"modelType": identity.model_type, "model": identity.model}


def _search_from_payload(
    payload: Mapping[str, Any] | None,
) -> TrainingSearch | None:
    if payload is None:
        return None
    values = {key: list(options) for key, options in payload["values"].items()}
    mode, random_samples = materialized_search_options(payload)
    return TrainingSearch(
        mode=mode,
        values=values,
        random_samples=random_samples,
        custom_value_axes=tuple(cast(list[str], payload.get("customValueAxes", []))),
    )


def _search_to_payload(search: TrainingSearch) -> dict[str, Any]:
    mode, samples = training_search_options(search.mode, search.random_samples)
    payload: dict[str, Any] = {"mode": mode, "values": search.values}
    if samples is not None:
        payload["randomSamples"] = samples
    if search.custom_value_axes:
        payload["customValueAxes"] = list(search.custom_value_axes)
    return payload


def _preset_searches_from_payload(
    payload: Mapping[str, Any],
    *,
    presets: list[str],
) -> dict[str, TrainingSearch | None]:
    raw_preset_searches = payload["presetSearches"]
    assert isinstance(raw_preset_searches, Mapping)
    return {
        preset: _search_from_payload(
            cast(Mapping[str, Any] | None, raw_preset_searches[preset])
        )
        for preset in presets
    }


def _preset_searches_to_payload(
    plan: TrainingRunPlanView,
) -> dict[str, dict[str, Any] | None]:
    payload: dict[str, dict[str, Any] | None] = {}
    for preset in plan.presets:
        search = plan.preset_searches[preset]
        payload[preset] = _search_to_payload(search) if search is not None else None
    return payload


def _run_change_from_payload(
    payload: Mapping[str, Any],
) -> TrainingRunChangeView:
    return TrainingRunChangeView(
        key=payload["key"],
        label=payload["label"],
        value=payload["value"],
        source=payload["source"],
    )


def _run_change_to_payload(change: TrainingRunChangeView) -> dict[str, Any]:
    return {
        "key": change.key,
        "label": change.label,
        "value": change.value,
        "source": change.source,
    }


def _run_from_payload(payload: Mapping[str, Any]) -> TrainingRunView:
    return TrainingRunView(
        id=payload["id"],
        index=payload["index"],
        status=payload["status"],
        preset=payload["preset"],
        dataset=payload["dataset"],
        experiment_task=payload["experimentTask"],
        changes=[_run_change_from_payload(change) for change in payload["changes"]],
        overrides=dict(payload["overrides"]),
        command_argv=list(payload["commandArgv"]),
        commands=TrainingCommandsView(
            posix=payload["commands"]["posix"],
            powershell=payload["commands"]["powershell"],
        ),
        total_epochs=payload["totalEpochs"],
        snapshot_id=payload["snapshotId"],
        snapshot_name=payload["snapshotName"],
        current_epoch=payload["currentEpoch"],
        metrics=copy.deepcopy(payload["metrics"]),
        log_dir=payload["logDir"],
        error=payload["error"],
        error_traceback=payload["errorTraceback"],
    )


def _run_to_payload(run: TrainingRunView) -> dict[str, Any]:
    return {
        "id": run.id,
        "index": run.index,
        "status": run.status,
        "preset": run.preset,
        "snapshotId": run.snapshot_id,
        "snapshotName": run.snapshot_name,
        "dataset": run.dataset,
        "experimentTask": run.experiment_task,
        "changes": [_run_change_to_payload(change) for change in run.changes],
        "overrides": run.overrides,
        "commandArgv": run.command_argv,
        "commands": {
            "posix": run.commands.posix,
            "powershell": run.commands.powershell,
        },
        "totalEpochs": run.total_epochs,
        "currentEpoch": run.current_epoch,
        "metrics": run.metrics,
        "logDir": run.log_dir,
        "error": run.error,
        "errorTraceback": run.error_traceback,
    }


def _summary_from_payload(
    payload: Mapping[str, Any],
) -> TrainingRunPlanSummaryView:
    return TrainingRunPlanSummaryView(
        total_runs=payload["totalRuns"],
        completed_runs=payload["completedRuns"],
        running_runs=payload["runningRuns"],
        pending_runs=payload["pendingRuns"],
        failed_runs=payload["failedRuns"],
        cancelled_runs=payload["cancelledRuns"],
        skipped_runs=payload["skippedRuns"],
        total_epochs=payload["totalEpochs"],
        completed_epochs=payload["completedEpochs"],
        remaining_epochs=payload["remainingEpochs"],
    )


def _summary_to_payload(summary: TrainingRunPlanSummaryView) -> dict[str, int]:
    return {
        "totalRuns": summary.total_runs,
        "completedRuns": summary.completed_runs,
        "runningRuns": summary.running_runs,
        "pendingRuns": summary.pending_runs,
        "failedRuns": summary.failed_runs,
        "cancelledRuns": summary.cancelled_runs,
        "skippedRuns": summary.skipped_runs,
        "totalEpochs": summary.total_epochs,
        "completedEpochs": summary.completed_epochs,
        "remainingEpochs": summary.remaining_epochs,
    }


def _snapshot_revisions_from_payload(
    payload: object,
) -> tuple[ConfigSnapshotRevision, ...]:
    if not isinstance(payload, list):
        raise ValueError("runPlan.snapshotRevisions must be a list.")
    revisions: list[ConfigSnapshotRevision] = []
    seen: set[str] = set()
    for raw_revision in payload:
        if not isinstance(raw_revision, Mapping):
            raise ValueError("Snapshot revisions must contain objects.")
        snapshot_id = raw_revision.get("id")
        semantic_revision = raw_revision.get("semanticRevision")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("Snapshot revision id must be non-empty.")
        if (
            not isinstance(semantic_revision, str)
            or len(semantic_revision) != 64
            or any(
                character not in "0123456789abcdef" for character in semantic_revision
            )
        ):
            raise ValueError(f"Snapshot '{snapshot_id}' semantic revision is invalid.")
        if snapshot_id in seen:
            raise ValueError(f"Duplicate snapshot revision '{snapshot_id}'.")
        seen.add(snapshot_id)
        revisions.append(
            ConfigSnapshotRevision(
                id=snapshot_id,
                semantic_revision=semantic_revision,
            )
        )
    return tuple(revisions)


def _snapshot_revisions_to_payload(
    revisions: tuple[ConfigSnapshotRevision, ...],
) -> list[dict[str, str]]:
    return [
        {"id": revision.id, "semanticRevision": revision.semantic_revision}
        for revision in revisions
    ]


def _plan_from_payload(payload: Mapping[str, Any]) -> TrainingRunPlanView:
    presets = list(payload["presets"])
    return TrainingRunPlanView(
        model=_payload_model_id(payload),
        preset=payload["preset"],
        presets=presets,
        experiment_task=payload["experimentTask"],
        datasets=list(payload["datasets"]),
        overrides=dict(payload["overrides"]),
        search=_search_from_payload(payload["search"]),
        log_folder=payload["logFolder"],
        is_random_search=payload["isRandomSearch"],
        runs=[_run_from_payload(row) for row in payload["runs"]],
        summary=_summary_from_payload(payload["summary"]),
        snapshot_revisions=_snapshot_revisions_from_payload(
            payload["snapshotRevisions"]
        ),
        preset_searches=_preset_searches_from_payload(payload, presets=presets),
    )


def _plan_to_payload(plan: TrainingRunPlanView) -> dict[str, Any]:
    return {
        **_model_identity(plan.model),
        "preset": plan.preset,
        "presets": plan.presets,
        "experimentTask": plan.experiment_task,
        "datasets": plan.datasets,
        "overrides": plan.overrides,
        "search": _search_to_payload(plan.search) if plan.search else None,
        "presetSearches": _preset_searches_to_payload(plan),
        "logFolder": plan.log_folder,
        "isRandomSearch": plan.is_random_search,
        "runs": [_run_to_payload(run) for run in plan.runs],
        "summary": _summary_to_payload(plan.summary),
        "snapshotRevisions": _snapshot_revisions_to_payload(plan.snapshot_revisions),
    }


class RunPlanPersistenceCodec:
    """Own the stable persisted and worker JSON representation of a Run Plan."""

    @staticmethod
    def decode(payload: object) -> TrainingRunPlanView:
        if not isinstance(payload, Mapping):
            raise ValueError("Persisted Run Plan must be an object.")
        _validate_plan_payload(payload)
        return _plan_from_payload(payload)

    @staticmethod
    def encode(plan: TrainingRunPlanView) -> dict[str, Any]:
        return copy.deepcopy(_plan_to_payload(plan))

    @staticmethod
    def decode_search(payload: Mapping[str, Any] | None) -> TrainingSearch | None:
        if payload is not None:
            _validate_search_payload(payload, path="search")
        return _search_from_payload(payload)

    @staticmethod
    def encode_search(search: TrainingSearch) -> dict[str, Any]:
        return copy.deepcopy(_search_to_payload(search))


__all__ = ["RunPlanPersistenceCodec"]
