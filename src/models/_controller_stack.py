from typing import Protocol, cast

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from models._residual_stack import ResidualStackSettings, build_residual_config


class ControllerLayerSettings(Protocol):
    @property
    def activation(self) -> ActivationOptions: ...

    @property
    def layer_norm_position(self) -> LayerNormPositionOptions: ...

    @property
    def normalization(self) -> NormalizationOptions: ...

    @property
    def dropout_probability(self) -> float: ...


class ControllerStackSettings(ControllerLayerSettings, Protocol):
    @property
    def hidden_dim(self) -> int: ...

    @property
    def num_layers(self) -> int: ...

    @property
    def last_layer_bias_option(self) -> LastLayerBiasOptions: ...

    @property
    def apply_output_postprocessing_flag(self) -> bool: ...


class _ResidualSettings(ControllerStackSettings, Protocol):
    @property
    def residual_connection_option(self) -> type[ResidualConfig] | None: ...

    @property
    def residual_model_flag(self) -> bool: ...

    @property
    def residual_stack_options(self) -> ResidualStackSettings | None: ...

    @property
    def residual_block_size(self) -> int | None: ...

    @property
    def residual_rms_norm_epsilon(self) -> float | None: ...


class _LinearBias(Protocol):
    @property
    def bias_flag(self) -> bool: ...


def build_controller_stack_config(
    options: _ResidualSettings,
    *,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
    layer_model_config: ConfigBase | None = None,
) -> LayerStackConfig:
    residual_config = build_controller_residual(options)
    if layer_model_config is None:
        layer_model_config = LinearLayerConfig(
            bias_flag=cast(_LinearBias, options).bias_flag
        )
    return assemble_controller_stack(
        options,
        layer_model_config,
        residual_config,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )


def build_controller_stack_with_model(
    options: _ResidualSettings,
    *,
    layer_model_config: ConfigBase,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    residual_config = build_controller_residual(options)
    return assemble_controller_stack(
        options,
        layer_model_config,
        residual_config,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )


def build_controller_residual(options: _ResidualSettings) -> ResidualConfig | None:
    return build_residual_config(
        options.residual_connection_option,
        options.residual_model_flag,
        options.residual_stack_options,
        residual_block_size=options.residual_block_size,
        residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
    )


def assemble_controller_stack(
    options: ControllerStackSettings,
    layer_model_config: ConfigBase,
    residual_config: ResidualConfig | None,
    *,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    layer_config = build_controller_layer(options, layer_model_config, residual_config)
    if hidden_dim is None:
        hidden_dim = options.hidden_dim
    return LayerStackConfig(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=options.num_layers,
        last_layer_bias_option=options.last_layer_bias_option,
        apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
        layer_config=layer_config,
    )


def build_controller_layer(
    options: ControllerLayerSettings,
    layer_model_config: ConfigBase,
    residual_config: ResidualConfig | None,
    *,
    gate_config: GateConfig | None = None,
    halting_config: HaltingConfig | None = None,
) -> LayerConfig:
    return LayerConfig(
        activation=options.activation,
        layer_norm_position=options.layer_norm_position,
        normalization=options.normalization,
        residual_config=residual_config,
        dropout_probability=options.dropout_probability,
        gate_config=gate_config,
        halting_config=halting_config,
        memory_config=None,
        layer_model_config=layer_model_config,
    )
