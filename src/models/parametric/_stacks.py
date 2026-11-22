from typing import Final, Protocol

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, SamplerConfig
from models._expert_mixture import SamplerSettings, build_expert_sampler
from models._residual_stack import build_residual_config


class _MixtureSettings(Protocol):
    top_k: Final[int]
    num_experts: Final[int]


class _RouterSettings(Protocol):
    activation: Final[ActivationOptions]
    noisy_topk_flag: Final[bool]


def router_hidden_dim(input_dim: int) -> int:
    return max(4, min(input_dim, 32))


def build_linear_stack_config(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    activation: ActivationOptions,
    residual_connection_option: type[ResidualConfig] | None,
    residual_model_flag: bool,
    residual_block_size: int | None = None,
    residual_rms_norm_epsilon: float | None = None,
    dropout_probability: float,
    apply_output_postprocessing_flag: bool,
) -> LayerStackConfig:
    layer_model_config = LinearLayerConfig(
        input_dim=input_dim, output_dim=output_dim, bias_flag=True
    )
    residual_config = build_residual_config(
        residual_connection_option,
        residual_model_flag,
        residual_block_size=residual_block_size,
        residual_rms_norm_epsilon=residual_rms_norm_epsilon,
    )
    layer_config = LayerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        activation=activation,
        residual_config=residual_config,
        dropout_probability=dropout_probability,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        normalization=NormalizationOptions.RMS_NORM,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=layer_model_config,
    )
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=apply_output_postprocessing_flag,
        layer_config=layer_config,
    )


def build_router_config(
    *,
    input_dim: int,
    mixture_options: _MixtureSettings,
    router_options: _RouterSettings,
) -> RouterConfig:
    model_config = build_linear_stack_config(
        input_dim=input_dim,
        hidden_dim=router_hidden_dim(input_dim),
        output_dim=mixture_options.num_experts,
        num_layers=1,
        activation=router_options.activation,
        residual_connection_option=None,
        residual_block_size=None,
        residual_rms_norm_epsilon=None,
        residual_model_flag=False,
        dropout_probability=0.0,
        apply_output_postprocessing_flag=False,
    )
    return RouterConfig(
        input_dim=input_dim,
        num_experts=mixture_options.num_experts,
        noisy_topk_flag=router_options.noisy_topk_flag,
        model_config=model_config,
    )


def build_sampler_config(
    *,
    mixture_options: _MixtureSettings,
    sampler_options: SamplerSettings,
) -> SamplerConfig:
    return build_expert_sampler(mixture_options, sampler_options)
