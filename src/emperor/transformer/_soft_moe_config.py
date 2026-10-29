"""Select sequence-aware Soft MoE while retaining configured feed-forward experts."""

from copy import deepcopy

from emperor.config import ConfigBase
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
    SoftMoEConfig,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
)


def build_soft_moe_stack(
    stack_config: ConfigBase,
    soft_moe_config: SoftMoEConfig | None,
    hidden_dim: int,
) -> ConfigBase:
    if soft_moe_config is None:
        return stack_config
    if not isinstance(soft_moe_config, SoftMoEConfig):
        raise TypeError("Feed-forward soft_moe_config must be a SoftMoEConfig.")
    model_config = _build_mixture_model_config(
        stack_config, soft_moe_config, hidden_dim
    )
    mixture_config = model_config.stack_config.layer_config.layer_model_config
    _configure_soft_routing(model_config, mixture_config, soft_moe_config)
    return model_config


def _build_mixture_model_config(
    stack_config: ConfigBase, soft_moe_config: SoftMoEConfig, hidden_dim: int
) -> MixtureOfExpertsModelConfig:
    if isinstance(stack_config, MixtureOfExpertsModelConfig):
        return deepcopy(stack_config)
    if not isinstance(stack_config, LayerStackConfig):
        raise ValueError(
            "Soft MoE requires a direct feed-forward stack without recurrent wrappers."
        )
    expert_stack_config = deepcopy(stack_config)
    mixture_config = MixtureOfExpertsConfig(
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        num_experts=soft_moe_config.num_experts,
        expert_model_config=expert_stack_config,
    )
    outer_stack_config = _build_outer_mixture_stack(mixture_config, hidden_dim)
    return MixtureOfExpertsModelConfig(
        input_dim=hidden_dim, output_dim=hidden_dim, stack_config=outer_stack_config
    )


def _build_outer_mixture_stack(
    mixture_config: MixtureOfExpertsConfig, hidden_dim: int
) -> LayerStackConfig:
    outer_layer_config = MixtureOfExpertsLayerConfig(
        activation=ActivationOptions.DISABLED,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        dropout_probability=0.0,
        layer_model_config=mixture_config,
    )
    return LayerStackConfig(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=False,
        layer_config=outer_layer_config,
    )


def _configure_soft_routing(
    model_config: MixtureOfExpertsModelConfig,
    mixture_config: MixtureOfExpertsConfig,
    soft_moe_config: SoftMoEConfig,
) -> None:
    if soft_moe_config.num_experts is not None:
        mixture_config.num_experts = soft_moe_config.num_experts
    expert_count = mixture_config.num_experts
    if type(expert_count) is not int or expert_count <= 0:
        raise ValueError("Soft MoE requires a positive num_experts.")
    mixture_config.top_k = expert_count
    mixture_config.capacity_factor = 0.0
    mixture_config.dropped_token_behavior = DroppedTokenOptions.ZEROS
    mixture_config.compute_expert_mixture_flag = True
    mixture_config.weighted_parameters_flag = True
    mixture_config.weighting_position_option = (
        ExpertWeightingPositionOptions.AFTER_EXPERTS
    )
    mixture_config.routing_initialization_mode = RoutingInitializationMode.LAYER
    mixture_config.sampler_config = None
    mixture_config.soft_moe_config = deepcopy(soft_moe_config)
    model_config.top_k = expert_count
    model_config.routing_initialization_mode = RoutingInitializationMode.LAYER
    model_config.sampler_config = None
