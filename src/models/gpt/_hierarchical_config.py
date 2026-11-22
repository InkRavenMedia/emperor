"""HAT v2 byte components around the selected GPT backbone."""

import copy
from typing import Protocol, cast

import torch

from emperor.attention import SelfAttentionConfig, SelfAttentionProjectionStrategy
from emperor.config import ConfigBase, ModelConfig
from emperor.embedding.hierarchical import (
    HierarchicalByteDecoderConfig,
    HierarchicalByteEmbeddingConfig,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.transformer import (
    SwiGLUFeedForwardConfig,
    TransformerConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from emperor.transformer.hierarchical import HierarchicalLanguageModelConfig


class _ByteOptions(Protocol):
    @property
    def byte_embedding_dim(self) -> int: ...

    @property
    def byte_limit(self) -> int: ...

    @property
    def byte_feed_forward_dim(self) -> int: ...

    @property
    def byte_num_heads(self) -> int: ...

    @property
    def byte_encoder_num_layers(self) -> int: ...

    @property
    def byte_decoder_num_layers(self) -> int: ...

    @property
    def layer_norm_flag(self) -> bool: ...

    @property
    def normalization(self) -> NormalizationOptions: ...

    @property
    def dropout_probability(self) -> float: ...


class _DecoderStages(Protocol):
    @property
    def decoder_config(self) -> ConfigBase | None: ...

    @property
    def positional_embedding_config(self) -> ConfigBase | None: ...

    @property
    def decoder_output_normalization(self) -> NormalizationOptions: ...


def build_hierarchical_config(
    model_config: ModelConfig, options: _ByteOptions
) -> HierarchicalLanguageModelConfig:
    dimension = options.byte_embedding_dim
    maximum = options.byte_limit
    maximum_batch = cast(int, model_config.batch_size) * cast(
        int, model_config.sequence_length
    )

    def linear_stack(input_dim: int | None, output_dim: int | None) -> LayerStackConfig:
        layer_model_config = LinearLayerConfig(bias_flag=False)
        layer_config = LayerConfig(
            activation=ActivationOptions.DISABLED,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            layer_model_config=layer_model_config,
        )
        return LayerStackConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=options.byte_feed_forward_dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_postprocessing_flag=False,
            layer_config=layer_config,
        )

    def transformer(causal: bool, depth: int) -> TransformerConfig:
        projection_model_config = linear_stack(dimension, dimension)
        attention = SelfAttentionConfig(
            batch_size=maximum_batch,
            num_heads=options.byte_num_heads,
            embedding_dim=dimension,
            query_key_projection_dim=dimension,
            value_projection_dim=dimension,
            source_sequence_length=maximum + 1,
            target_sequence_length=maximum + 1,
            target_dtype=torch.float32,
            dropout_probability=0.0,
            zero_attention_flag=False,
            causal_attention_mask_flag=causal,
            add_key_value_bias_flag=False,
            average_attention_weights_flag=False,
            return_attention_weights_flag=False,
            batch_first_flag=True,
            projection_model_config=projection_model_config,
            projection_strategy=SelfAttentionProjectionStrategy.FUSED,
            rotary_base=10000.0,
        )
        residual_config = AdditiveResidualConfig()
        feed_forward_config = SwiGLUFeedForwardConfig(
            input_dim=dimension,
            output_dim=dimension,
            hidden_dim=options.byte_feed_forward_dim,
        )
        if causal:
            layer = TransformerDecoderLayerConfig(
                self_attention_config=attention,
                embedding_dim=dimension,
                layer_norm_position=LayerNormPositionOptions.BEFORE,
                dropout_probability=0.0,
                residual_config=residual_config,
                normalization=NormalizationOptions.RMS_NORM,
                feed_forward_config=feed_forward_config,
            )
            block = TransformerDecoderBlockLayerConfig
        else:
            layer = TransformerEncoderLayerConfig(
                attention_config=attention,
                embedding_dim=dimension,
                layer_norm_position=LayerNormPositionOptions.BEFORE,
                dropout_probability=0.0,
                residual_config=residual_config,
                normalization=NormalizationOptions.RMS_NORM,
                feed_forward_config=feed_forward_config,
            )
            block = TransformerEncoderBlockLayerConfig
        layer_config = block(
            activation=ActivationOptions.DISABLED,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            layer_model_config=layer,
        )
        stack = LayerStackConfig(
            input_dim=dimension,
            output_dim=dimension,
            hidden_dim=dimension,
            num_layers=depth,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=True,
            layer_config=layer_config,
        )
        if causal:
            return TransformerConfig(
                decoder_stack_config=stack,
                final_normalization=NormalizationOptions.RMS_NORM,
            )
        return TransformerConfig(
            encoder_stack_config=stack,
            final_normalization=NormalizationOptions.RMS_NORM,
        )

    experiment = cast(_DecoderStages, model_config.experiment_config)
    encoder_config = transformer(False, options.byte_encoder_num_layers)
    projection_config = linear_stack(dimension, model_config.hidden_dim)
    embedding_config = HierarchicalByteEmbeddingConfig(
        byte_embedding_dim=dimension,
        output_dim=model_config.hidden_dim,
        max_token_bytes=maximum,
        byte_position_config=None,
        encoder_config=encoder_config,
        projection_config=projection_config,
    )
    decoder_config = transformer(True, options.byte_decoder_num_layers)
    conditioning_projection_config = linear_stack(model_config.hidden_dim, dimension)
    output_projection_config = linear_stack(dimension, 256)
    decoding_config = HierarchicalByteDecoderConfig(
        conditioning_dim=model_config.hidden_dim,
        byte_embedding_dim=dimension,
        max_token_bytes=maximum,
        byte_position_config=None,
        decoder_config=decoder_config,
        conditioning_projection_config=conditioning_projection_config,
        output_projection_config=output_projection_config,
    )
    backbone_config = copy.deepcopy(experiment.decoder_config)
    position_config = copy.deepcopy(experiment.positional_embedding_config)
    embedding_normalization_config = None
    if options.layer_norm_flag:
        embedding_normalization_config = LayerConfig(
            input_dim=model_config.hidden_dim,
            output_dim=model_config.hidden_dim,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            normalization=options.normalization,
        )
    output_normalization_config = LayerConfig(
        input_dim=model_config.hidden_dim,
        output_dim=model_config.hidden_dim,
        layer_norm_position=LayerNormPositionOptions.BEFORE,
        normalization=experiment.decoder_output_normalization,
    )
    return HierarchicalLanguageModelConfig(
        sequence_length=model_config.sequence_length,
        embedding_config=embedding_config,
        decoding_config=decoding_config,
        # Preserve this package's selected attention, experts and adaptive layers.
        backbone_config=backbone_config,
        position_config=position_config,
        embedding_normalization_config=embedding_normalization_config,
        output_normalization_config=output_normalization_config,
        dropout_probability=options.dropout_probability,
    )
