"""Bind declared flat and structured options to package-owned defaults."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import Field, fields, replace
from functools import cache
from types import ModuleType
from typing import Any, ClassVar, Protocol, TypeVar, get_type_hints

from model_runtime.packages.configuration import (
    config_key_to_model_param,
    iter_supported_config_keys,
)
from model_runtime.packages.runtime_values import (
    validate_annotated_runtime_value,
    validate_runtime_default_value_types,
)


class _DefaultsRecord(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


_Record = TypeVar("_Record", bound=_DefaultsRecord)


@cache
def _initialization_annotations(
    record_type: type[_DefaultsRecord],
) -> dict[str, object]:
    annotations = get_type_hints(record_type, include_extras=True)
    return {
        field.name: annotations[field.name]
        for field in fields(record_type)
        if field.init
    }


def apply_runtime_default_fields(
    defaults: _Record,
    values: Mapping[str, object],
    *,
    package: str,
    config_module: ModuleType,
    delegated: Set[str],
) -> _Record:
    """Validate all supplied values before applying the fields owned by this record."""
    if not values:
        return defaults
    annotations = _initialization_annotations(type(defaults))
    declared = {
        config_key_to_model_param(key)
        for key in iter_supported_config_keys(config_module)
    }
    unknown = values.keys() - annotations.keys() - (delegated & declared)
    if unknown:
        names = ", ".join(repr(name) for name in sorted(unknown))
        raise TypeError(f"{package}: unknown runtime option(s): {names}")
    flat = {name: value for name, value in values.items() if name in declared}
    validate_runtime_default_value_types(
        flat, package=package, config_module=config_module
    )
    for name, value in values.items():
        if name not in declared:
            validate_annotated_runtime_value(
                annotations[name], value, key=name, package=package
            )
    overrides = {name: value for name, value in values.items() if name not in delegated}
    return replace(defaults, **overrides)
