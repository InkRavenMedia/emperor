from __future__ import annotations

from collections.abc import Callable
from typing import Final, Protocol, TypeVar, cast

from emperor.attention import AttentionSourceMemoryConfig, MixerAttentionConfig
from emperor.config import ConfigBase
from emperor.experts import MixtureOfExpertsModelConfig, SoftMoEConfig
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentCompositionConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import (
    DynamicMemoryConfig,
)
from emperor.transformer import (
    FeedForwardConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from emperor.transformer._soft_moe_config import build_soft_moe_stack
from models import _sequence_expansion
from models.mlp_mixer._component_factory import BuildResidual


class _ImageGrid(Protocol):
    image_height: Final[int]
    image_patch_size: Final[int]


class _OutputSettings(Protocol):
    hidden_dim: Final[int]
    output_bias_flag: Final[bool]
    output_dim: Final[int]


class _EncoderSettings(Protocol):
    hidden_dim: Final[int]
    layer_norm_position: Final[LayerNormPositionOptions]
    mixer_residual_block_size: Final[int | None]
    mixer_residual_connection_option: Final[type[ResidualConfig]]
    mixer_residual_model_flag: Final[bool]
    mixer_residual_rms_norm_epsilon: Final[float | None]
    normalization: Final[NormalizationOptions]
    soft_moe_config: Final[SoftMoEConfig | None]
    stack_apply_output_postprocessing_flag: Final[bool]
    stack_dropout_probability: Final[float]
    stack_last_layer_bias_option: Final[LastLayerBiasOptions]
    stack_num_layers: Final[int]
    stack_residual_block_size: Final[int | None]
    stack_residual_connection_option: Final[type[ResidualConfig] | None]
    stack_residual_model_flag: Final[bool]
    stack_residual_rms_norm_epsilon: Final[float | None]
    token_mixer_soft_moe_config: Final[SoftMoEConfig | None]


_Settings = TypeVar("_Settings", bound=_EncoderSettings)


def patch_sequence_length(runtime: _ImageGrid) -> int:
    image_height = runtime.image_height
    patch_size = runtime.image_patch_size
    if image_height <= 0 or patch_size <= 0:
        raise ValueError("image_height and image_patch_size must be positive")
    if image_height % patch_size != 0:
        raise ValueError(
            "image_height must be exactly divisible by image_patch_size, got "
            f"image_height={image_height}, image_patch_size={patch_size}"
        )
    patches_per_side = image_height // patch_size
    return patches_per_side**2


def build_token_attention(
    runtime: _Settings,
    tokens: int,
    *,
    build_token_model: Callable[[_Settings, int], ConfigBase],
    build_source_memory: Callable[
        [_Settings], DynamicMemoryConfig | AttentionSourceMemoryConfig | None
    ],
) -> MixerAttentionConfig:
    token_mixing_model_config = build_token_model(runtime, tokens)
    token_mixing_config = build_soft_moe_stack(
        token_mixing_model_config,
        runtime.token_mixer_soft_moe_config,
        tokens,
    )
    source_memory_config = build_source_memory(runtime)
    return MixerAttentionConfig(
        embedding_dim=runtime.hidden_dim,
        sequence_length=tokens,
        batch_first_flag=True,
        mixing_model_config=token_mixing_config,
        source_memory_config=source_memory_config,
    )


def build_encoder_stack(
    runtime: _Settings,
    tokens: int,
    *,
    build_residual: BuildResidual[_Settings],
    build_attention: Callable[[_Settings, int], MixerAttentionConfig],
    build_channel_model: Callable[[_Settings], ConfigBase],
) -> LayerStackConfig:
    mixer_residual_config = build_residual(
        runtime,
        runtime.mixer_residual_connection_option,
        runtime.mixer_residual_model_flag,
        residual_block_size=runtime.mixer_residual_block_size,
        residual_rms_norm_epsilon=runtime.mixer_residual_rms_norm_epsilon,
    )
    attention_config = build_attention(runtime, tokens)
    channel_mixing_model_config = build_channel_model(runtime)
    channel_stack_config = build_soft_moe_stack(
        channel_mixing_model_config,
        runtime.soft_moe_config,
        runtime.hidden_dim,
    )
    feed_forward_config = FeedForwardConfig(
        input_dim=runtime.hidden_dim,
        output_dim=runtime.hidden_dim,
        stack_config=cast(
            LayerStackConfig | RecurrentCompositionConfig | MixtureOfExpertsModelConfig,
            channel_stack_config,
        ),
    )
    mixer_layer = TransformerEncoderLayerConfig(
        embedding_dim=runtime.hidden_dim,
        layer_norm_position=runtime.layer_norm_position,
        normalization=runtime.normalization,
        dropout_probability=runtime.stack_dropout_probability,
        residual_config=mixer_residual_config,
        attention_config=attention_config,
        feed_forward_config=feed_forward_config,
    )
    sequence_expansion_config = _sequence_expansion.build_sequence_expansion_config(
        cast(_sequence_expansion.SequenceExpansionOptions, runtime), runtime.hidden_dim
    )
    block_residual_config = build_residual(
        runtime,
        runtime.stack_residual_connection_option,
        runtime.stack_residual_model_flag,
        residual_block_size=runtime.stack_residual_block_size,
        residual_rms_norm_epsilon=runtime.stack_residual_rms_norm_epsilon,
    )
    block_layer = TransformerEncoderBlockLayerConfig(
        sequence_expansion_config=sequence_expansion_config,
        activation=ActivationOptions.DISABLED,
        residual_config=block_residual_config,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        normalization=NormalizationOptions.RMS_NORM,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=mixer_layer,
    )
    stack = LayerStackConfig(
        input_dim=runtime.hidden_dim,
        hidden_dim=runtime.hidden_dim,
        output_dim=runtime.hidden_dim,
        num_layers=runtime.stack_num_layers,
        apply_output_postprocessing_flag=runtime.stack_apply_output_postprocessing_flag,
        last_layer_bias_option=runtime.stack_last_layer_bias_option,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=block_layer,
    )
    return stack


def output_config(runtime: _OutputSettings) -> LayerConfig:
    linear_layer_config = LinearLayerConfig(bias_flag=runtime.output_bias_flag)
    return LayerConfig(
        input_dim=runtime.hidden_dim,
        output_dim=runtime.output_dim,
        activation=ActivationOptions.DISABLED,
        residual_config=None,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        normalization=NormalizationOptions.RMS_NORM,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=linear_layer_config,
    )
