from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import replace
from typing import Any, TypeVar, cast

_StructuredT = TypeVar("_StructuredT")


def pop_updates(
    values: dict[str, Any],
    field_map: Mapping[str, str],
) -> dict[str, Any]:
    """Remove mapped flat values and return their structured field names."""

    return {
        structured_field: values.pop(flat_field)
        for flat_field, structured_field in field_map.items()
        if flat_field in values
    }


def replace_prefixed_fields(
    values: dict[str, Any],
    source: _StructuredT,
    prefix: str,
    fields: tuple[str, ...],
) -> _StructuredT:
    """Apply one prefixed family of flat values to a structured source."""

    updates = pop_updates(
        values,
        {f"{prefix}_{field}": field for field in fields},
    )
    if not updates:
        return source
    return cast(_StructuredT, replace(cast(Any, source), **updates))


def pop_prefixed_updates(
    values: MutableMapping[str, Any],
    prefix: str,
    field_map: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    updates = {}
    for suffix in field_map:
        key = f"{prefix}{suffix}"
        if key in values:
            updates[suffix] = values.pop(key)
    return updates
