from __future__ import annotations

import inspect
import json
import math
from collections import ChainMap
from collections.abc import Mapping, Sequence
from dataclasses import fields
from enum import Enum, StrEnum
from types import ModuleType
from typing import Any, TypeAlias, cast

from emperor.config import ConfigBase
from model_runtime.packages.annotations import (
    annotation_classes,
    annotation_is_nullable,
    resolved_annotations,
)

_ConfigJson: TypeAlias = (
    bool | int | float | str | None | list["_ConfigJson"] | dict[str, "_ConfigJson"]
)

SKIP_CONFIG_KEYS = {
    "CONFIG_SCHEMA_SKIP_KEYS",
    "CONFIG_OVERRIDE_SKIP_KEYS",
    "DEFAULT_EXPERIMENT_TASK",
    "DATASET_OPTIONS_BY_TASK",
    "MONITOR_OPTIONS",
}


class ConfigValueError(ValueError):
    """A Runtime Defaults value cannot be interpreted through its public type."""


def require_finite_learning_rate(config_key: str, value: object) -> None:
    if config_key.removeprefix("SEARCH_SPACE_") == "LEARNING_RATE":
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigValueError(f"{config_key} must be finite, got {value!r}")


def normalize_key(key: str) -> str:
    return key.strip().replace("-", "_").lower()


def config_key_to_flag(key: str) -> str:
    return "--" + key.lower().replace("_", "-")


def config_key_to_model_param(key: str) -> str:
    return key.lower()


def search_key_to_config_key(key: str) -> str:
    return "SEARCH_SPACE_" + normalize_key(key).upper()


def canonical_config_key(key: str) -> str:
    stripped_key = key.strip().replace("-", "_")
    if stripped_key.isupper():
        return stripped_key
    return normalize_key(key).upper()


def _is_supported_constant(value: Any) -> bool:
    return value is None or isinstance(value, (int, float, str, bool, Enum, type))


def iter_supported_config_keys(config_module: ModuleType) -> list[str]:
    module_skip_keys = {
        key
        for key in getattr(config_module, "CONFIG_OVERRIDE_SKIP_KEYS", ())
        if isinstance(key, str)
    }
    skip_keys = SKIP_CONFIG_KEYS | module_skip_keys
    return sorted(
        key
        for key, value in vars(config_module).items()
        if not key.startswith("_")
        and key.isupper()
        and not key.startswith("SEARCH_SPACE_")
        and key not in skip_keys
        and _is_supported_constant(value)
    )


def _bool_value(raw_value: str) -> bool:
    value = raw_value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ConfigValueError(f"expected a boolean value, got '{raw_value}'")


def _class_lookup(config_module: ModuleType, raw_value: str) -> type:
    value = raw_value.split(".")[-1]
    candidate = getattr(config_module, value, None)
    if inspect.isclass(candidate):
        return candidate
    raise ConfigValueError(f"unknown config class '{raw_value}'")


def _enum_lookup(enum_type: type[Enum], raw_value: str) -> Enum:
    value = raw_value.split(".")[-1]
    try:
        return enum_type[value]
    except KeyError as exc:
        if issubclass(enum_type, StrEnum):
            for member in enum_type:
                if member.value == value:
                    return member
        choices = ", ".join(enum_type.__members__)
        raise ConfigValueError(
            f"unknown {enum_type.__name__} value '{raw_value}'. Choices: {choices}"
        ) from exc


def _parse_from_annotation(
    config_module: ModuleType,
    annotation: Any,
    raw_value: str,
) -> Any:
    if raw_value.lower() in {"none", "null"}:
        return None
    classes = annotation_classes(annotation)
    enum_classes = [cls for cls in classes if issubclass(cls, Enum)]
    if enum_classes:
        try:
            parsed: Any = _enum_lookup(enum_classes[0], raw_value)
        except ConfigValueError as exc:
            if annotation_is_nullable(annotation):
                raise ConfigValueError(
                    f"{exc}. Use None to leave this optional setting unset."
                ) from exc
            raise
    elif bool in classes:
        parsed = _bool_value(raw_value)
    elif int in classes:
        parsed = int(raw_value)
    elif float in classes:
        parsed = float(raw_value)
    elif str in classes:
        parsed = raw_value
    elif classes:
        parsed = _class_lookup(config_module, raw_value)
    else:
        parsed = raw_value
    return parsed


def _parse_from_list_value(
    config_module: ModuleType,
    values: list[Any],
    raw_value: str,
) -> Any:
    sample = next((value for value in values if value is not None), None)
    if sample is None:
        return raw_value
    return _parse_from_current_value(config_module, sample, None, raw_value)


def _parse_from_current_value(
    config_module: ModuleType,
    current_value: Any,
    annotation: Any,
    raw_value: str,
) -> Any:
    if raw_value.lower() in {"none", "null"}:
        return None
    if isinstance(current_value, list):
        return _parse_from_list_value(
            config_module,
            cast(list[Any], current_value),
            raw_value,
        )
    if isinstance(current_value, bool):
        parsed: Any = _bool_value(raw_value)
    elif isinstance(current_value, int) and not isinstance(current_value, bool):
        parsed = int(raw_value)
    elif isinstance(current_value, float):
        parsed = float(raw_value)
    elif isinstance(current_value, Enum):
        parsed = _enum_lookup(type(current_value), raw_value)
    elif isinstance(current_value, str):
        parsed = raw_value
    elif inspect.isclass(current_value):
        parsed = _class_lookup(config_module, raw_value)
    elif current_value is None:
        parsed = _parse_from_annotation(config_module, annotation, raw_value)
    else:
        parsed = raw_value
    return parsed


def parse_config_value(
    config_module: ModuleType,
    key: str,
    raw_value: str,
) -> Any:
    annotation = resolved_annotations(config_module).get(key)
    if raw_value.lstrip().startswith("{") and any(
        issubclass(candidate, ConfigBase)
        for candidate in annotation_classes(annotation)
    ):
        try:
            payload = cast(_ConfigJson, json.loads(raw_value))
            extra_types = getattr(config_module, "_CONFIG_OBJECT_TYPES", {}).get(
                key, {}
            )
            return _parse_config_object(
                ChainMap(vars(config_module), extra_types), payload
            )
        except (TypeError, ValueError) as exc:
            raise ConfigValueError(
                f"Invalid configuration object for {key}: {exc}"
            ) from exc
    parsed = _parse_from_current_value(
        config_module,
        getattr(config_module, key, None),
        annotation,
        raw_value,
    )
    require_finite_learning_rate(key, parsed)
    return parsed


def _parse_config_object(namespace: Mapping[str, Any], value: _ConfigJson) -> object:
    """Decode only package-exposed config constructors and qualified enum members."""
    if isinstance(value, dict):
        name = value.get("type")
        if not isinstance(name, str):
            raise ConfigValueError(
                "Configuration objects require a string 'type' field."
            )
        config_type = namespace.get(name.rsplit(".", 1)[-1])
        if not inspect.isclass(config_type):
            raise ConfigValueError(f"unknown config class '{name}'")
        if not issubclass(config_type, ConfigBase):
            raise ConfigValueError(f"{name} is not a configuration class.")
        supplied = value.keys() - {"type"}
        unknown = supplied - {field.name for field in fields(config_type)}
        if unknown:
            raise ConfigValueError(f"Unknown {name} fields: {sorted(unknown)}")
        return config_type(
            **{key: _parse_config_object(namespace, value[key]) for key in supplied}
        )
    if isinstance(value, list):
        return [_parse_config_object(namespace, item) for item in value]
    if isinstance(value, str) and "." in value:
        enum_name, member = value.rsplit(".", 1)
        enum_type = namespace.get(enum_name)
        if inspect.isclass(enum_type) and issubclass(enum_type, Enum):
            return _enum_lookup(enum_type, member)
    return value


def _config_json_value(value: object) -> object:
    if isinstance(value, ConfigBase):
        return {
            "type": type(value).__name__,
            **{
                field.name: _config_json_value(getattr(value, field.name))
                for field in fields(value)
                if getattr(value, field.name) is not None
            },
        }
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, (list, tuple)):
        return [_config_json_value(item) for item in cast(Sequence[object], value)]
    return value


def abstract_config_class_error(candidate: type) -> str | None:
    try:
        if not issubclass(candidate, ConfigBase):
            return None
    except TypeError:
        return None
    try:
        instance = candidate()
    except TypeError:
        return None
    try:
        instance.registry_owner()
    except (NotImplementedError, ValueError) as exc:
        return str(exc)
    return None


def serialize_config_value(value: Any) -> bool | int | float | str | None:
    if isinstance(value, ConfigBase):
        try:
            serialized = json.dumps(_config_json_value(value), separators=(",", ":"))
        except TypeError:
            # Preserve the display form for configs containing non-JSON values.
            serialized = str(value)
        return serialized
    if isinstance(value, Enum) and value.name.lower() in {"none", "null"}:
        return f"{type(value).__name__}.{value.name}"
    if hasattr(value, "name"):
        return value.name
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "ConfigValueError",
    "SKIP_CONFIG_KEYS",
    "abstract_config_class_error",
    "canonical_config_key",
    "config_key_to_flag",
    "config_key_to_model_param",
    "iter_supported_config_keys",
    "normalize_key",
    "parse_config_value",
    "search_key_to_config_key",
    "serialize_config_value",
]
