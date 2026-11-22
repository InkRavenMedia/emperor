from collections.abc import Callable
from dataclasses import replace
from typing import Final, Generic, Protocol, TypeVar

from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from models._expert_flat_options import OptionRecord

_Result = TypeVar("_Result")


class CoreSettings(Protocol):
    batch_size: Final[int]
    learning_rate: Final[float]
    input_dim: Final[int]
    hidden_dim: Final[int]
    output_dim: Final[int]
    stack_bias_flag: Final[bool]
    layer_norm_position: Final[LayerNormPositionOptions]
    normalization: Final[NormalizationOptions]
    stack_num_layers: Final[int]
    stack_activation: Final[ActivationOptions]
    stack_residual_connection_option: Final[type[ResidualConfig] | None]
    stack_residual_model_flag: Final[bool]
    stack_residual_block_size: Final[int | None]
    stack_residual_rms_norm_epsilon: Final[float | None]
    stack_dropout_probability: Final[float]
    stack_last_layer_bias_option: Final[LastLayerBiasOptions]
    stack_apply_output_postprocessing_flag: Final[bool]
    submodule_stack_hidden_dim: Final[int]
    submodule_stack_num_layers: Final[int]
    submodule_stack_activation: Final[ActivationOptions]
    submodule_stack_residual_connection_option: Final[type[ResidualConfig] | None]
    submodule_stack_residual_model_flag: Final[bool]
    submodule_stack_residual_block_size: Final[int | None]
    submodule_stack_residual_rms_norm_epsilon: Final[float | None]
    submodule_stack_dropout_probability: Final[float]
    submodule_stack_layer_norm_position: Final[LayerNormPositionOptions]
    submodule_stack_normalization: Final[NormalizationOptions]
    submodule_stack_last_layer_bias_option: Final[LastLayerBiasOptions]
    submodule_stack_apply_output_postprocessing_flag: Final[bool]
    submodule_stack_bias_flag: Final[bool]
    top_k: Final[int]
    num_experts: Final[int]
    capacity_factor: Final[float]
    dropped_token_behavior: Final[DroppedTokenOptions]
    compute_expert_mixture_flag: Final[bool]
    weighted_parameters_flag: Final[bool]
    weighting_position_option: Final[ExpertWeightingPositionOptions]
    routing_initialization_mode: Final[RoutingInitializationMode]
    expert_stack_hidden_dim: Final[int | None]
    expert_stack_num_layers: Final[int | None]
    expert_stack_activation: Final[ActivationOptions | None]
    expert_stack_residual_connection_option: Final[type[ResidualConfig] | None]
    expert_stack_residual_model_flag: Final[bool]
    expert_stack_residual_block_size: Final[int | None]
    expert_stack_residual_rms_norm_epsilon: Final[float | None]
    expert_stack_dropout_probability: Final[float | None]
    expert_stack_layer_norm_position: Final[LayerNormPositionOptions | None]
    expert_stack_normalization: Final[NormalizationOptions | None]
    expert_stack_last_layer_bias_option: Final[LastLayerBiasOptions | None]
    expert_stack_apply_output_postprocessing_flag: Final[bool | None]
    expert_bias_flag: Final[bool | None]
    sampler_threshold: Final[float]
    sampler_filter_above_threshold: Final[bool]
    sampler_num_topk_samples: Final[int]
    sampler_normalize_probabilities_flag: Final[bool]
    sampler_noisy_topk_flag: Final[bool]
    sampler_coefficient_of_variation_loss_weight: Final[float]
    sampler_switch_loss_weight: Final[float]
    sampler_zero_centred_loss_weight: Final[float]
    sampler_mutual_information_loss_weight: Final[float]
    stack_options: Final[object]
    submodule_stack_options: Final[object]
    mixture_options: Final[object]
    expert_stack_options: Final[object]
    sampler_options: Final[object]


class ExpertCoreDefaults(Generic[_Result]):
    resolve_stack: Callable[..., object]
    stack_type: Callable[..., object]
    submodule_type: Callable[..., object]
    mixture_type: Callable[..., object]
    sampler_type: Callable[..., object]
    defaults_type: Callable[..., _Result]

    def resolve_core_defaults(self, values: CoreSettings) -> _Result:
        stack_options = values.stack_options or self.stack_type(
            hidden_dim=values.hidden_dim,
            bias_flag=values.stack_bias_flag,
            layer_norm_position=values.layer_norm_position,
            normalization=values.normalization,
            num_layers=values.stack_num_layers,
            activation=values.stack_activation,
            residual_connection_option=values.stack_residual_connection_option,
            residual_block_size=values.stack_residual_block_size,
            residual_rms_norm_epsilon=values.stack_residual_rms_norm_epsilon,
            residual_model_flag=values.stack_residual_model_flag,
            dropout_probability=values.stack_dropout_probability,
            last_layer_bias_option=values.stack_last_layer_bias_option,
            apply_output_postprocessing_flag=values.stack_apply_output_postprocessing_flag,
        )
        submodule_stack_options = values.submodule_stack_options or self.submodule_type(
            hidden_dim=values.submodule_stack_hidden_dim,
            num_layers=values.submodule_stack_num_layers,
            last_layer_bias_option=values.submodule_stack_last_layer_bias_option,
            apply_output_postprocessing_flag=values.submodule_stack_apply_output_postprocessing_flag,
            activation=values.submodule_stack_activation,
            layer_norm_position=values.submodule_stack_layer_norm_position,
            normalization=values.submodule_stack_normalization,
            residual_connection_option=values.submodule_stack_residual_connection_option,
            residual_block_size=values.submodule_stack_residual_block_size,
            residual_rms_norm_epsilon=values.submodule_stack_residual_rms_norm_epsilon,
            residual_model_flag=values.submodule_stack_residual_model_flag,
            dropout_probability=values.submodule_stack_dropout_probability,
            bias_flag=values.submodule_stack_bias_flag,
        )
        mixture_options = values.mixture_options or self.mixture_type(
            top_k=values.top_k,
            num_experts=values.num_experts,
            capacity_factor=values.capacity_factor,
            dropped_token_behavior=values.dropped_token_behavior,
            compute_expert_mixture_flag=values.compute_expert_mixture_flag,
            weighted_parameters_flag=values.weighted_parameters_flag,
            weighting_position_option=values.weighting_position_option,
            routing_initialization_mode=values.routing_initialization_mode,
        )
        expert_stack_options = values.expert_stack_options or self.resolve_stack(
            submodule_stack_options,
            hidden_dim=values.expert_stack_hidden_dim,
            num_layers=values.expert_stack_num_layers,
            last_layer_bias_option=values.expert_stack_last_layer_bias_option,
            apply_output_postprocessing_flag=values.expert_stack_apply_output_postprocessing_flag,
            activation=values.expert_stack_activation,
            layer_norm_position=values.expert_stack_layer_norm_position,
            normalization=values.expert_stack_normalization,
            residual_connection_option=values.expert_stack_residual_connection_option,
            residual_block_size=values.expert_stack_residual_block_size,
            residual_rms_norm_epsilon=values.expert_stack_residual_rms_norm_epsilon,
            residual_model_flag=values.expert_stack_residual_model_flag,
            dropout_probability=values.expert_stack_dropout_probability,
            bias_flag=values.expert_bias_flag,
        )
        sampler_options = values.sampler_options or self.sampler_type(
            threshold=values.sampler_threshold,
            filter_above_threshold=values.sampler_filter_above_threshold,
            num_topk_samples=values.sampler_num_topk_samples,
            normalize_probabilities_flag=values.sampler_normalize_probabilities_flag,
            noisy_topk_flag=values.sampler_noisy_topk_flag,
            coefficient_of_variation_loss_weight=values.sampler_coefficient_of_variation_loss_weight,
            switch_loss_weight=values.sampler_switch_loss_weight,
            zero_centred_loss_weight=values.sampler_zero_centred_loss_weight,
            mutual_information_loss_weight=values.sampler_mutual_information_loss_weight,
        )
        return self.defaults_type(
            batch_size=values.batch_size,
            learning_rate=values.learning_rate,
            input_dim=values.input_dim,
            output_dim=values.output_dim,
            stack_options=stack_options,
            submodule_stack_options=submodule_stack_options,
            mixture_options=mixture_options,
            expert_stack_options=expert_stack_options,
            sampler_options=sampler_options,
        )


class _CoreStacks(OptionRecord, Protocol):
    stack_options: Final[OptionRecord]
    submodule_stack_options: Final[OptionRecord]
    expert_stack_options: Final[OptionRecord]


class _ControlStacks(OptionRecord, Protocol):
    residual_stack_options: Final[object]
    router_stack_options: Final[OptionRecord]


class _AdaptiveStacks(OptionRecord, Protocol):
    adaptive_generator_stack_options: Final[OptionRecord]


_Core = TypeVar("_Core", bound=_CoreStacks)
_Control = TypeVar("_Control", bound=_ControlStacks)
_Adaptive = TypeVar("_Adaptive", bound=_AdaptiveStacks)


def bind_residual_stacks(
    core: _Core,
    control: _Control,
    adaptive: _Adaptive,
) -> tuple[_Core, _Control, _Adaptive]:
    residual_stack_options = control.residual_stack_options
    core = replace(
        core,
        stack_options=replace(
            core.stack_options, residual_stack_options=residual_stack_options
        ),
        submodule_stack_options=replace(
            core.submodule_stack_options,
            residual_stack_options=residual_stack_options,
        ),
        expert_stack_options=replace(
            core.expert_stack_options,
            residual_stack_options=residual_stack_options,
        ),
    )
    control = replace(
        control,
        router_stack_options=replace(
            control.router_stack_options,
            residual_stack_options=residual_stack_options,
        ),
    )
    adaptive = replace(
        adaptive,
        adaptive_generator_stack_options=replace(
            adaptive.adaptive_generator_stack_options,
            residual_stack_options=residual_stack_options,
        ),
    )
    return core, control, adaptive
