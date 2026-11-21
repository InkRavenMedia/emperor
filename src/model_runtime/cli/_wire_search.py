from __future__ import annotations

from typing import Any

from model_runtime.cli._wire_shared import (
    WireCodecError,
    json_value_from_wire,
    json_value_to_wire,
    require_sequence_limit,
    wire_bool,
    wire_fields,
    wire_list,
    wire_string,
)
from model_runtime.runs import (
    PlanningBudget,
    SearchAxisSelection,
    SearchMode,
    SearchOptionError,
    SearchSpec,
    resolve_search_options,
)

_DEFAULT_BUDGET = PlanningBudget()
_MAX_WIRE_SEARCH_AXES = _DEFAULT_BUDGET.max_axes or 16
_MAX_WIRE_SEARCH_VALUES = _DEFAULT_BUDGET.max_values_per_axis or 50


def _wire_search_options(
    mode: object, count: object, path: str, *, planned: bool = False
) -> tuple[SearchMode, int | None]:
    try:
        return resolve_search_options(mode, count, planned=planned)
    except SearchOptionError as exc:
        raise WireCodecError(f"{path}.{exc}") from exc


def search_spec_to_wire_at(
    search: SearchSpec | None,
    path: str,
) -> dict[str, Any] | None:
    if search is None:
        return None
    if search.axes is not None:
        require_sequence_limit(
            search.axes,
            f"{path}.axes",
            _MAX_WIRE_SEARCH_AXES,
        )
        for index, axis in enumerate(search.axes):
            if axis.values is not None:
                require_sequence_limit(
                    axis.values,
                    f"{path}.axes[{index}].values",
                    _MAX_WIRE_SEARCH_VALUES,
                )
    mode, count = _wire_search_options(search.mode, search.random_samples, path)
    payload = {
        "mode": mode,
        "axes": (
            None
            if search.axes is None
            else [
                {
                    "key": axis.key,
                    "values": (
                        None
                        if axis.values is None
                        else [
                            json_value_to_wire(
                                value,
                                path=f"{path}.axes[].values[]",
                            )
                            for value in axis.values
                        ]
                    ),
                    "allow_custom_values": axis.allow_custom_values,
                }
                for axis in search.axes
            ]
        ),
        "random_samples": count,
    }
    search_spec_from_wire_at(payload, path)
    return payload


def search_spec_to_wire(search: SearchSpec | None) -> dict[str, Any] | None:
    return search_spec_to_wire_at(search, "$.search")


def _search_axis_from_wire(
    item: object,
    index: int,
    search_path: str,
    *,
    planned: bool,
) -> SearchAxisSelection:
    path = f"{search_path}.axes[{index}]"
    axis = wire_fields(
        item,
        path=path,
        required=("key", "values", "allow_custom_values") if planned else ("key",),
        optional=() if planned else ("values", "allow_custom_values"),
    )
    raw_values = axis.get("values")
    return SearchAxisSelection(
        key=wire_string(axis["key"], f"{path}.key"),
        values=(
            None
            if raw_values is None
            else tuple(
                json_value_from_wire(
                    value,
                    path=f"{path}.values[{value_index}]",
                )
                for value_index, value in enumerate(
                    wire_list(
                        raw_values,
                        f"{path}.values",
                        maximum_items=_MAX_WIRE_SEARCH_VALUES,
                    )
                )
            )
        ),
        allow_custom_values=wire_bool(
            axis.get("allow_custom_values", False),
            f"{path}.allow_custom_values",
        ),
    )


def _search_axes_from_wire(
    payload: object,
    path: str,
    *,
    planned: bool,
) -> tuple[SearchAxisSelection, ...] | None:
    if payload is None:
        return None
    decoded_axes: list[SearchAxisSelection] = []
    for index, item in enumerate(
        wire_list(
            payload,
            f"{path}.axes",
            maximum_items=_MAX_WIRE_SEARCH_AXES,
        )
    ):
        decoded_axes.append(_search_axis_from_wire(item, index, path, planned=planned))
    return tuple(decoded_axes)


def search_spec_from_wire_at(
    payload: object, path: str, *, planned: bool = False
) -> SearchSpec | None:
    if payload is None:
        return None
    raw = wire_fields(
        payload,
        path=path,
        required=("mode", "axes", "random_samples") if planned else ("mode",),
        optional=() if planned else ("axes", "random_samples"),
    )
    axes = _search_axes_from_wire(raw.get("axes"), path, planned=planned)
    mode, count = _wire_search_options(
        raw["mode"], raw.get("random_samples"), path, planned=planned
    )
    return SearchSpec(mode=mode, axes=axes, random_samples=count)


def search_spec_from_wire(payload: object) -> SearchSpec | None:
    return search_spec_from_wire_at(payload, "$.search")


__all__ = [
    "search_spec_from_wire",
    "search_spec_from_wire_at",
    "search_spec_to_wire",
    "search_spec_to_wire_at",
]
