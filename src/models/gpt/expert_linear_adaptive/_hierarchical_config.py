"""Package-owned HAT v2 byte components around this package's native backbone."""

import copy

import torch

from emperor.attention import SelfAttentionConfig, SelfAttentionProjectionStrategy
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


def build_hierarchical_config(model_config, options):
    dimension = options.byte_embedding_dim
    maximum = options.byte_limit
    maximum_batch = model_config.batch_size * model_config.sequence_length

    def linear_stack(input_dim, output_dim):
        return LayerStackConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=options.byte_feed_forward_dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_postprocessing_flag=False,
            layer_config=LayerConfig(
                activation=ActivationOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                layer_model_config=LinearLayerConfig(bias_flag=False),
            ),
        )

    def transformer(causal, depth):
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
            projection_model_config=linear_stack(dimension, dimension),
            projection_strategy=SelfAttentionProjectionStrategy.FUSED,
            rotary_base=10000.0,
        )
        arguments = dict(
            embedding_dim=dimension,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            dropout_probability=0.0,
            residual_config=AdditiveResidualConfig(),
            normalization=NormalizationOptions.RMS_NORM,
            feed_forward_config=SwiGLUFeedForwardConfig(
                input_dim=dimension,
                output_dim=dimension,
                hidden_dim=options.byte_feed_forward_dim,
            ),
        )
        if causal:
            layer = TransformerDecoderLayerConfig(
                self_attention_config=attention, **arguments
            )
            block = TransformerDecoderBlockLayerConfig
        else:
            layer = TransformerEncoderLayerConfig(
                attention_config=attention, **arguments
            )
            block = TransformerEncoderBlockLayerConfig
        stack = LayerStackConfig(
            input_dim=dimension,
            output_dim=dimension,
            hidden_dim=dimension,
            num_layers=depth,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=True,
            layer_config=block(
                activation=ActivationOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                layer_model_config=layer,
            ),
        )
        return TransformerConfig(
            final_normalization=NormalizationOptions.RMS_NORM,
            **{"decoder_stack_config" if causal else "encoder_stack_config": stack},
        )

    experiment = model_config.experiment_config
    return HierarchicalLanguageModelConfig(
        sequence_length=model_config.sequence_length,
        embedding_config=HierarchicalByteEmbeddingConfig(
            byte_embedding_dim=dimension,
            output_dim=model_config.hidden_dim,
            max_token_bytes=maximum,
            byte_position_config=None,
            encoder_config=transformer(False, options.byte_encoder_num_layers),
            projection_config=linear_stack(dimension, model_config.hidden_dim),
        ),
        decoding_config=HierarchicalByteDecoderConfig(
            conditioning_dim=model_config.hidden_dim,
            byte_embedding_dim=dimension,
            max_token_bytes=maximum,
            byte_position_config=None,
            decoder_config=transformer(True, options.byte_decoder_num_layers),
            conditioning_projection_config=linear_stack(
                model_config.hidden_dim, dimension
            ),
            output_projection_config=linear_stack(dimension, 256),
        ),
        # Preserve this package's selected attention, experts and adaptive layers.
        backbone_config=copy.deepcopy(experiment.decoder_config),
        position_config=copy.deepcopy(experiment.positional_embedding_config),
        embedding_normalization_config=LayerConfig(
            input_dim=model_config.hidden_dim,
            output_dim=model_config.hidden_dim,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            normalization=options.normalization,
        )
        if options.layer_norm_flag
        else None,
        output_normalization_config=LayerConfig(
            input_dim=model_config.hidden_dim,
            output_dim=model_config.hidden_dim,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            normalization=experiment.decoder_output_normalization,
        ),
        dropout_probability=options.dropout_probability,
    )
