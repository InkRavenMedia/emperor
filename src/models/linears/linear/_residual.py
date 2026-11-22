from __future__ import annotations

from dataclasses import dataclass, field

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    ResidualConfig,
)
from models._residual_stack import (
    ResidualBuilder,
    ResidualStackSettings,
    StackDefaults,
    resolve_stack_options,
)
from models._residual_stack import inherit_if_unset as inherit_if_unset
from models.linears.linear import SETTINGS_VALIDATION


@dataclass(frozen=True, slots=True)
class ResidualStackSource:
    coefficient_activation: ActivationOptions = field(
        default=ActivationOptions.SIGMOID, kw_only=True
    )
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    normalization: NormalizationOptions | None = field(default=None, kw_only=True)
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    bias_flag: bool | None


@dataclass(frozen=True, slots=True)
class ResidualStackOptions:
    coefficient_activation: ActivationOptions = field(
        default=ActivationOptions.SIGMOID, kw_only=True
    )
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    bias_flag: bool

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.positive_integer(
            "residual_stack_hidden_dim", self.hidden_dim
        )
        SETTINGS_VALIDATION.stack(self, "residual_stack")


def resolve_residual_stack_options(
    source: ResidualStackSource,
    defaults: StackDefaults,
) -> ResidualStackOptions:
    return resolve_stack_options(source, defaults, ResidualStackOptions)


def build_residual_config(
    residual_connection_option: type[ResidualConfig] | None,
    residual_model_flag: bool,
    residual_stack_options: ResidualStackSettings | None = None,
    *,
    residual_block_size: int | None = None,
    residual_rms_norm_epsilon: float | None = None,
    selector_field: str = "residual_connection_option",
    model_flag_field: str = "residual_model_flag",
) -> ResidualConfig | None:
    builder = ResidualBuilder(lazy_coefficient_activation=True)
    return builder.build_residual_config(
        residual_connection_option,
        residual_model_flag,
        residual_stack_options,
        residual_block_size=residual_block_size,
        residual_rms_norm_epsilon=residual_rms_norm_epsilon,
        selector_field=selector_field,
        model_flag_field=model_flag_field,
    )


def build_residual_stack_config(options: ResidualStackSettings) -> LayerStackConfig:
    builder = ResidualBuilder(lazy_coefficient_activation=True)
    return builder.build_residual_stack_config(options)
