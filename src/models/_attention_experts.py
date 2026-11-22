from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Final, Generic, Protocol, TypeVar, cast

import torch

from emperor.attention import AttentionSourceMemoryConfig, MixtureOfAttentionHeadsConfig
from emperor.experts import (
    MixtureOfExpertsConfig,
    MixtureOfExpertsModelConfig,
    SoftMoEConfig,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentCompositionConfig,
    RecurrentLayerConfig,
)
from emperor.transformer._soft_moe_config import build_soft_moe_attention_experts
from models._linear_stack import LayerStackSettings
from models._residual_stack import ResidualStackSettings


class ExpertBlockSettings(Protocol):
    activation: Final[ActivationOptions]
    dropout_probability: Final[float]
    layer_norm_position: Final[LayerNormPositionOptions]
    normalization: Final[NormalizationOptions]


class _Attention(Protocol):
    attention_soft_moe_config: Final[SoftMoEConfig | None]
    num_heads: Final[int]
    add_key_value_bias_flag: Final[bool]


class _FeedForward(Protocol):
    bias_flag: Final[bool]
    num_layers: Final[int]


class _ExpertStack(Protocol):
    residual_stack_options: Final[ResidualStackSettings | None]


class _ExpertInputs(Protocol):
    hidden_dim: Final[int]
    attention_options: Final[_Attention]
    feed_forward_options: Final[_FeedForward]
    expert_stack_options: Final[_ExpertStack]
    expert_attention_use_kv_expert_models_flag: Final[bool]


class _ExpertControl(Protocol):
    def build(self) -> MixtureOfExpertsModelConfig | RecurrentLayerConfig: ...


_Inputs = TypeVar("_Inputs", bound=_ExpertInputs)
_Stack = TypeVar("_Stack")
_ControlInputs = TypeVar("_ControlInputs")


def feed_forward_expert_stack(
    factory: Callable[..., _Stack], options: LayerStackSettings
) -> _Stack:
    return factory(
        hidden_dim=options.hidden_dim,
        bias_flag=options.bias_flag,
        layer_norm_position=options.layer_norm_position,
        normalization=options.normalization,
        num_layers=options.num_layers,
        activation=options.activation,
        residual_connection_option=(options.residual_connection_option),
        residual_block_size=options.residual_block_size,
        residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
        residual_model_flag=options.residual_model_flag,
        residual_stack_options=options.residual_stack_options,
        dropout_probability=options.dropout_probability,
        last_layer_bias_option=options.last_layer_bias_option,
        apply_output_postprocessing_flag=(options.apply_output_postprocessing_flag),
    )


def attention_expert_stack(
    factory: Callable[..., _Stack],
    hidden_dim: int,
    block: ExpertBlockSettings,
    feed_forward: _FeedForward,
    experts: _ExpertStack,
) -> _Stack:
    return factory(
        hidden_dim=hidden_dim,
        bias_flag=feed_forward.bias_flag,
        layer_norm_position=block.layer_norm_position,
        normalization=block.normalization,
        num_layers=feed_forward.num_layers,
        activation=block.activation,
        residual_connection_option=None,
        residual_block_size=None,
        residual_rms_norm_epsilon=None,
        residual_model_flag=False,
        residual_stack_options=(experts.residual_stack_options),
        dropout_probability=block.dropout_probability,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=True,
    )


def mixture_attention_heads(
    *,
    batch_size: int,
    num_heads: int,
    hidden_dim: int,
    sequence_length: int,
    dropout_probability: float,
    causal_attention_mask_flag: bool,
    add_key_value_bias_flag: bool,
    projection_model_config: LayerStackConfig | RecurrentCompositionConfig,
    experts_config: MixtureOfExpertsConfig,
    use_kv_expert_models_flag: bool,
    source_memory_flag: bool | None = None,
    source_memory_config: AttentionSourceMemoryConfig | None = None,
) -> MixtureOfAttentionHeadsConfig:
    return MixtureOfAttentionHeadsConfig(
        batch_size=batch_size,
        num_heads=num_heads,
        embedding_dim=hidden_dim,
        query_key_projection_dim=hidden_dim,
        value_projection_dim=hidden_dim,
        target_sequence_length=sequence_length,
        source_sequence_length=sequence_length,
        target_dtype=torch.float32,
        dropout_probability=dropout_probability,
        zero_attention_flag=False,
        causal_attention_mask_flag=causal_attention_mask_flag,
        add_key_value_bias_flag=add_key_value_bias_flag,
        source_memory_flag=source_memory_flag,
        source_memory_config=source_memory_config,
        average_attention_weights_flag=False,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=projection_model_config,
        experts_config=experts_config,
        use_kv_expert_models_flag=(use_kv_expert_models_flag),
    )


class AttentionExpertBuilder(ABC, Generic[_Inputs, _Stack]):
    _stack: Callable[..., _Stack]

    def __init__(self, dependencies: _Inputs) -> None:
        self.dependencies = dependencies

    @property
    @abstractmethod
    def _block_options(self) -> ExpertBlockSettings: ...

    def _causal_attention(self, options: ExpertBlockSettings) -> bool:
        return False

    @abstractmethod
    def _build_control_config(self, stack_options: _Stack) -> _ExpertControl: ...

    def build_feed_forward_base_stack_config(
        self,
        feed_forward_stack_options: LayerStackSettings,
    ) -> MixtureOfExpertsModelConfig:
        stack_options = self._feed_forward_experts_stack_options(
            feed_forward_stack_options
        )
        feed_forward_base_stack_config = self._build_expert_model_config(stack_options)
        return feed_forward_base_stack_config

    def build_attention_config(
        self,
        *,
        batch_size: int,
        hidden_dim: int,
        sequence_length: int,
        projection_model_config: LayerStackConfig | RecurrentLayerConfig,
    ) -> MixtureOfAttentionHeadsConfig:
        dependencies = self.dependencies
        block_options = self._block_options
        attention_options = dependencies.attention_options
        attention_experts_config = self._build_attention_experts_config()
        experts_config = build_soft_moe_attention_experts(
            attention_experts_config,
            attention_options.attention_soft_moe_config,
        )
        return mixture_attention_heads(
            batch_size=batch_size,
            num_heads=attention_options.num_heads,
            hidden_dim=hidden_dim,
            sequence_length=sequence_length,
            dropout_probability=block_options.dropout_probability,
            causal_attention_mask_flag=self._causal_attention(block_options),
            add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
            projection_model_config=projection_model_config,
            experts_config=experts_config,
            use_kv_expert_models_flag=dependencies.expert_attention_use_kv_expert_models_flag,
        )

    def _build_attention_experts_config(self) -> MixtureOfExpertsConfig:
        stack_options = self._attention_experts_stack_options()
        model_config = self._build_expert_model_config(stack_options)
        stack_config = cast(LayerStackConfig, model_config.stack_config)
        layer_config = cast(LayerConfig, stack_config.layer_config)
        return cast(MixtureOfExpertsConfig, layer_config.layer_model_config)

    def _build_expert_model_config(
        self,
        stack_options: _Stack,
    ) -> MixtureOfExpertsModelConfig:
        model_config = self._build_control_config(stack_options).build()
        if isinstance(model_config, MixtureOfExpertsModelConfig):
            return model_config
        return cast(MixtureOfExpertsModelConfig, model_config.block_config)

    def _feed_forward_experts_stack_options(
        self,
        feed_forward_stack_options: LayerStackSettings | None,
    ) -> _Stack:
        if feed_forward_stack_options is None:
            raise ValueError("feed_forward_stack_options is required.")
        return feed_forward_expert_stack(self._stack, feed_forward_stack_options)

    def _attention_experts_stack_options(self) -> _Stack:
        dependencies = self.dependencies
        return attention_expert_stack(
            self._stack,
            dependencies.hidden_dim,
            self._block_options,
            dependencies.feed_forward_options,
            dependencies.expert_stack_options,
        )


class _RoutedInputs(_ExpertInputs, Protocol):
    mixture_options: Final[object]
    sampler_options: Final[object]
    router_options: Final[object]
    router_stack_options: Final[object]
    expert_layer_controller_options: Final[object]
    expert_dynamic_memory_options: Final[object]
    expert_recurrent_controller_options: Final[object]


_Routed = TypeVar("_Routed", bound=_RoutedInputs)


class RoutedAttentionExpertBuilder(
    AttentionExpertBuilder[_Routed, _Stack], Generic[_Routed, _Stack, _ControlInputs]
):
    _control_inputs: Callable[..., _ControlInputs]
    _control: Callable[[_ControlInputs], _ExpertControl]

    def _build_control_config(
        self,
        stack_options: _Stack,
    ) -> _ExpertControl:
        dependencies = self.dependencies
        expert_linear_control_config_dependencies = self._control_inputs(
            stack_options=stack_options,
            submodule_stack_options=None,
            mixture_options=dependencies.mixture_options,
            expert_stack_options=dependencies.expert_stack_options,
            sampler_options=dependencies.sampler_options,
            router_options=dependencies.router_options,
            router_stack_options=dependencies.router_stack_options,
            layer_controller_options=None,
            dynamic_memory_options=None,
            recurrent_controller_options=None,
            expert_layer_controller_options=(
                dependencies.expert_layer_controller_options
            ),
            expert_dynamic_memory_options=(dependencies.expert_dynamic_memory_options),
            expert_recurrent_controller_options=(
                dependencies.expert_recurrent_controller_options
            ),
            hidden_dim=dependencies.hidden_dim,
            output_dim=dependencies.hidden_dim,
        )
        return self._control(expert_linear_control_config_dependencies)


class _AdaptiveComponentSettings(Protocol):
    hidden_dim: Final[int]
    mixture_options: Final[object]
    expert_stack_options: Final[object]
    sampler_options: Final[object]
    router_options: Final[object]
    router_stack_options: Final[object]
    expert_layer_controller_options: Final[object]
    expert_dynamic_memory_options: Final[object]
    expert_recurrent_controller_options: Final[object]
    router_layer_controller_options: Final[object]
    router_dynamic_memory_options: Final[object]
    router_recurrent_controller_options: Final[object]
    adaptive_generator_stack_options: Final[object]
    hidden_adaptive_weight_options: Final[object]
    hidden_adaptive_bias_options: Final[object]
    hidden_adaptive_diagonal_options: Final[object]
    hidden_adaptive_mask_options: Final[object]
    router_grouping_config: Final[object]
    router_adaptive_weight_options: Final[object]
    router_adaptive_bias_options: Final[object]
    router_adaptive_diagonal_options: Final[object]
    router_adaptive_mask_options: Final[object]


class _AdaptiveInputs(_RoutedInputs, _AdaptiveComponentSettings, Protocol):
    mixture_submodule_stack_options: Final[object]
    mixture_layer_controller_options: Final[object]
    mixture_dynamic_memory_options: Final[object]
    mixture_recurrent_controller_options: Final[object]
    grouping_config: Final[object]


_Adaptive = TypeVar("_Adaptive", bound=_AdaptiveInputs)


class AdaptiveAttentionExpertBuilder(
    AttentionExpertBuilder[_Adaptive, _Stack],
    Generic[_Adaptive, _Stack, _ControlInputs],
):
    _control_inputs: Callable[..., _ControlInputs]
    _control: Callable[[_ControlInputs], _ExpertControl]

    def _build_control_config(
        self,
        stack_options: _Stack,
    ) -> _ExpertControl:
        dependencies = self.dependencies
        return adaptive_expert_controller(
            dependencies,
            stack_options,
            submodule_stack_options=dependencies.mixture_submodule_stack_options,
            layer_controller_options=dependencies.mixture_layer_controller_options,
            dynamic_memory_options=dependencies.mixture_dynamic_memory_options,
            recurrent_controller_options=dependencies.mixture_recurrent_controller_options,
            grouping_config=dependencies.grouping_config,
            inputs=self._control_inputs,
            factory=self._control,
        )


_Controller = TypeVar("_Controller", bound=_ExpertControl)


def adaptive_expert_controller(
    settings: _AdaptiveComponentSettings,
    stack_options: object,
    *,
    submodule_stack_options: object,
    layer_controller_options: object,
    dynamic_memory_options: object,
    recurrent_controller_options: object,
    grouping_config: object,
    inputs: Callable[..., _ControlInputs],
    factory: Callable[[_ControlInputs], _Controller],
) -> _Controller:
    control_inputs = inputs(
        stack_options=stack_options,
        submodule_stack_options=submodule_stack_options,
        mixture_options=settings.mixture_options,
        expert_stack_options=settings.expert_stack_options,
        sampler_options=settings.sampler_options,
        router_options=settings.router_options,
        router_stack_options=settings.router_stack_options,
        router_layer_controller_options=(settings.router_layer_controller_options),
        router_dynamic_memory_options=(settings.router_dynamic_memory_options),
        router_recurrent_controller_options=(
            settings.router_recurrent_controller_options
        ),
        layer_controller_options=(layer_controller_options),
        dynamic_memory_options=dynamic_memory_options,
        recurrent_controller_options=(recurrent_controller_options),
        expert_layer_controller_options=(settings.expert_layer_controller_options),
        expert_dynamic_memory_options=(settings.expert_dynamic_memory_options),
        expert_recurrent_controller_options=(
            settings.expert_recurrent_controller_options
        ),
        adaptive_generator_stack_options=(settings.adaptive_generator_stack_options),
        grouping_config=grouping_config,
        hidden_adaptive_weight_options=(settings.hidden_adaptive_weight_options),
        hidden_adaptive_bias_options=settings.hidden_adaptive_bias_options,
        hidden_adaptive_diagonal_options=(settings.hidden_adaptive_diagonal_options),
        hidden_adaptive_mask_options=settings.hidden_adaptive_mask_options,
        router_grouping_config=settings.router_grouping_config,
        router_adaptive_weight_options=(settings.router_adaptive_weight_options),
        router_adaptive_bias_options=settings.router_adaptive_bias_options,
        router_adaptive_diagonal_options=(settings.router_adaptive_diagonal_options),
        router_adaptive_mask_options=settings.router_adaptive_mask_options,
        hidden_dim=settings.hidden_dim,
        output_dim=settings.hidden_dim,
    )
    return factory(control_inputs)
