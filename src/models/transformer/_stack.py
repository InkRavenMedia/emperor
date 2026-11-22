from collections.abc import Callable
from typing import Final, Protocol, TypeVar

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import GatedResidualDynamicMemoryConfig, MemoryPositionOptions
from models._recurrent_block import RecurrentSchedule, build_recurrent_block
from models._residual_stack import (
    ResidualStackSettings,
    StackDefaults,
    build_residual_config,
)


class _ModelWidth(Protocol):
    model_dim: Final[int]


class ControlledStackSettings(Protocol):
    halting_option: Final[type[HaltingConfig]]
    halting_threshold: Final[float | None]
    memory_flag: Final[bool]
    num_layers: Final[int]
    recurrent_flag: Final[bool]
    recurrent_halting_option: Final[type[HaltingConfig]]
    recurrent_halting_threshold: Final[float | None]
    recurrent_residual_block_size: Final[int | None]
    recurrent_residual_connection_option: Final[type[ResidualConfig] | None]
    recurrent_residual_model_flag: Final[bool]
    recurrent_residual_rms_norm_epsilon: Final[float | None]
    recurrent_stack_gate_flag: Final[bool]
    recurrent_stack_halting_flag: Final[bool]
    stack_halting_flag: Final[bool]
    recurrent_max_steps: Final[int]
    recurrent_initial_iterations: Final[int]
    recurrent_no_gradient_transition_count: Final[int | None]
    recurrent_gradient_transition_count: Final[int | None]
    recurrent_iteration_increment: Final[int]
    recurrent_forward_calls_before_iteration_increment: Final[int]
    recurrent_smooth_iteration_growth_flag: Final[bool]


_Settings = TypeVar("_Settings", bound=_ModelWidth)
_Wrapped = TypeVar("_Wrapped", bound=ConfigBase)


def linear_stack(
    *,
    hidden_dim: int,
    output_dim: int | None = None,
    num_layers: int = 1,
    bias_flag: bool = True,
    activation: ActivationOptions = ActivationOptions.RELU,
    dropout_probability: float = 0.0,
) -> LayerStackConfig:
    layer_model_config = LinearLayerConfig(bias_flag=bias_flag)
    layer_config = LayerConfig(
        activation=activation,
        residual_config=None,
        dropout_probability=dropout_probability,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        normalization=NormalizationOptions.RMS_NORM,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=layer_model_config,
    )
    last_layer_bias = LastLayerBiasOptions.DEFAULT
    if output_dim is not None:
        last_layer_bias = LastLayerBiasOptions.DISABLED
    return LayerStackConfig(
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        apply_output_postprocessing_flag=False,
        last_layer_bias_option=last_layer_bias,
        layer_config=layer_config,
    )


def build_gate(
    model_dim: int,
    enabled: bool,
    *,
    activation: ActivationOptions = ActivationOptions.RELU,
) -> GateConfig | None:
    if not enabled:
        return None
    model_config = linear_stack(hidden_dim=model_dim, activation=activation)
    return GateConfig(
        option=LayerGateOptions.MULTIPLIER,
        activation=ActivationOptions.SIGMOID,
        model_config=model_config,
    )


def build_halting(
    model_dim: int,
    enabled: bool,
    option: type[HaltingConfig],
    threshold: float | None,
    *,
    activation: ActivationOptions = ActivationOptions.RELU,
) -> HaltingConfig | None:
    if not enabled:
        return None
    halting_gate_config = linear_stack(
        hidden_dim=model_dim, activation=activation, output_dim=2
    )
    return option(
        threshold=threshold,
        min_steps=1,
        ponder_cost_weight=1.0,
        dropout_probability=0.0,
        hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
        halting_gate_config=halting_gate_config,
    )


def build_memory(
    model_dim: int, enabled: bool
) -> GatedResidualDynamicMemoryConfig | None:
    if not enabled:
        return None
    model_config = linear_stack(hidden_dim=model_dim)
    return GatedResidualDynamicMemoryConfig(
        input_dim=model_dim,
        output_dim=model_dim,
        memory_position_option=MemoryPositionOptions.AFTER_AFFINE,
        test_time_training_learning_rate=None,
        test_time_training_num_inner_steps=None,
        model_config=model_config,
    )


def build_controlled_stack(
    runtime: _Settings,
    options: ControlledStackSettings,
    layer_config: LayerConfig,
    *,
    resolve_residual_stack: Callable[[_Settings], ResidualStackSettings],
    wrap_recurrent: Callable[
        [
            LayerStackConfig,
            GateConfig | None,
            ResidualConfig | None,
            HaltingConfig | None,
        ],
        _Wrapped,
    ],
    gate_activation: ActivationOptions = ActivationOptions.RELU,
) -> LayerStackConfig | _Wrapped:
    shared_halting_config = build_halting(
        runtime.model_dim,
        options.stack_halting_flag,
        options.halting_option,
        options.halting_threshold,
        activation=gate_activation,
    )
    shared_memory_config = build_memory(runtime.model_dim, options.memory_flag)
    stack = LayerStackConfig(
        input_dim=runtime.model_dim,
        hidden_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        num_layers=options.num_layers,
        apply_output_postprocessing_flag=True,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        shared_gate_config=None,
        shared_halting_config=shared_halting_config,
        shared_memory_config=shared_memory_config,
        layer_config=layer_config,
    )
    if not options.recurrent_flag:
        return stack
    gate_config = build_gate(
        runtime.model_dim, options.recurrent_stack_gate_flag, activation=gate_activation
    )
    residual_stack_options = resolve_residual_stack(runtime)
    residual_config = build_residual_config(
        options.recurrent_residual_connection_option,
        options.recurrent_residual_model_flag,
        residual_stack_options,
        residual_block_size=options.recurrent_residual_block_size,
        residual_rms_norm_epsilon=options.recurrent_residual_rms_norm_epsilon,
    )
    halting_config = build_halting(
        runtime.model_dim,
        options.recurrent_stack_halting_flag,
        options.recurrent_halting_option,
        options.recurrent_halting_threshold,
        activation=gate_activation,
    )
    return wrap_recurrent(stack, gate_config, residual_config, halting_config)


def wrap_recurrent_stack(
    runtime: _ModelWidth,
    options: ControlledStackSettings,
    stack: LayerStackConfig,
    gate_config: GateConfig | None,
    residual_config: ResidualConfig | None,
    halting_config: HaltingConfig | None,
) -> RecurrentLayerConfig:
    schedule = RecurrentSchedule.from_controller(options)
    return build_recurrent_block(
        schedule,
        stack,
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        smooth_iteration_growth_flag=options.recurrent_smooth_iteration_growth_flag,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        normalization=NormalizationOptions.LAYER_NORM,
        gate_config=gate_config,
        residual_config=residual_config,
        halting_config=halting_config,
    )


def plain_projection_stack(
    stack_options: StackDefaults,
    residual_stack_options: ResidualStackSettings,
) -> LayerStackConfig:
    residual_config = build_residual_config(
        stack_options.residual_connection_option,
        stack_options.residual_model_flag,
        residual_stack_options,
        residual_block_size=stack_options.residual_block_size,
        residual_rms_norm_epsilon=stack_options.residual_rms_norm_epsilon,
    )
    layer_model_config = LinearLayerConfig(bias_flag=stack_options.bias_flag)
    layer_config = LayerConfig(
        activation=stack_options.activation,
        residual_config=residual_config,
        dropout_probability=stack_options.dropout_probability,
        layer_norm_position=stack_options.layer_norm_position,
        normalization=stack_options.normalization,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=layer_model_config,
    )
    return LayerStackConfig(
        hidden_dim=stack_options.hidden_dim,
        num_layers=stack_options.num_layers,
        apply_output_postprocessing_flag=(
            stack_options.apply_output_postprocessing_flag
        ),
        last_layer_bias_option=stack_options.last_layer_bias_option,
        layer_config=layer_config,
    )
