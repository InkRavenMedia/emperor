"""Bind declared flat and structured options to package-owned defaults."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import Field, fields, replace
from functools import cache
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar, get_type_hints

from model_runtime.packages.configuration import (
    config_key_to_model_param,
    iter_supported_config_keys,
)
from model_runtime.packages.runtime_values import (
    validate_annotated_runtime_value,
    validate_runtime_default_value_types,
)

if TYPE_CHECKING:
    from emperor.layers import (
        ActivationOptions,
        LastLayerBiasOptions,
        LayerNormPositionOptions,
        NormalizationOptions,
        ResidualConfig,
    )


_Result = TypeVar("_Result", covariant=True)


class _OptionalStackConstructor(Protocol[_Result]):
    def __call__(
        self,
        *,
        independent_flag: bool,
        hidden_dim: int | None,
        num_layers: int | None,
        activation: ActivationOptions | None,
        dropout_probability: float | None,
        layer_norm_position: LayerNormPositionOptions | None,
        normalization: NormalizationOptions | None,
        residual_connection_option: type[ResidualConfig] | None,
        residual_model_flag: bool,
        residual_block_size: int | None,
        residual_rms_norm_epsilon: float | None,
        last_layer_bias_option: LastLayerBiasOptions | None,
        apply_output_postprocessing_flag: bool | None,
        bias_flag: bool | None,
    ) -> _Result: ...


def read_optional_stack(
    constructor: _OptionalStackConstructor[_Result],
    settings: object,
    prefix: str,
    *,
    config_constants: bool = False,
) -> _Result:
    """Bind an explicitly named controller or generator scope to its package-owned type."""

    def read(suffix: str) -> Any:
        key = f"{prefix}_{suffix}"
        return getattr(settings, key.upper() if config_constants else key)

    return constructor(
        independent_flag=read("independent_flag"),
        hidden_dim=read("hidden_dim"),
        num_layers=read("num_layers"),
        activation=read("activation"),
        dropout_probability=read("dropout_probability"),
        layer_norm_position=read("layer_norm_position"),
        normalization=read("normalization"),
        residual_connection_option=read("residual_connection_option"),
        residual_model_flag=read("residual_model_flag"),
        residual_block_size=read("residual_block_size"),
        residual_rms_norm_epsilon=read("residual_rms_norm_epsilon"),
        last_layer_bias_option=read("last_layer_bias_option"),
        apply_output_postprocessing_flag=read("apply_output_postprocessing_flag"),
        bias_flag=read("bias_flag"),
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
