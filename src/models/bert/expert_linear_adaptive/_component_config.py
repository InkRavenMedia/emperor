"""Standard optional BERT components composed from scalar Runtime Defaults."""

import torch

from emperor.attention import SelfAttentionConfig, SelfAttentionProjectionStrategy
from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.transformer import (
    FeedForwardConfig,
    TransformerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)


def _linear_stack(
    input_dim: int | None = None,
    output_dim: int | None = None,
    *,
    hidden_dim: int = 32,
    num_layers: int = 1,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.RELU,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def build_hierarchical_embedding_config(
    *,
    batch_size: int,
    sequence_length: int,
    hidden_dim: int,
    byte_embedding_dim: int,
    byte_encoder_num_layers: int,
    byte_feed_forward_dim: int,
    byte_num_heads: int,
    max_token_bytes: int,
) -> HierarchicalByteEmbeddingConfig:
    attention_config = SelfAttentionConfig(
        batch_size=batch_size * sequence_length,
        num_heads=byte_num_heads,
        embedding_dim=byte_embedding_dim,
        query_key_projection_dim=0,
        value_projection_dim=0,
        target_sequence_length=max_token_bytes + 1,
        source_sequence_length=max_token_bytes + 1,
        target_dtype=torch.float32,
        dropout_probability=0.0,
        zero_attention_flag=False,
        causal_attention_mask_flag=False,
        add_key_value_bias_flag=False,
        source_memory_flag=False,
        average_attention_weights_flag=False,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=_linear_stack(byte_embedding_dim, byte_embedding_dim),
        projection_strategy=SelfAttentionProjectionStrategy.FUSED,
    )
    encoder_layer = TransformerEncoderLayerConfig(
        embedding_dim=byte_embedding_dim,
        layer_norm_position=LayerNormPositionOptions.BEFORE,
        dropout_probability=0.0,
        residual_config=AdditiveResidualConfig(),
        attention_config=attention_config,
        feed_forward_config=FeedForwardConfig(
            input_dim=byte_embedding_dim,
            output_dim=byte_embedding_dim,
            stack_config=_linear_stack(
                byte_embedding_dim,
                byte_embedding_dim,
                hidden_dim=byte_feed_forward_dim,
            ),
        ),
    )
    return HierarchicalByteEmbeddingConfig(
        byte_embedding_dim=byte_embedding_dim,
        output_dim=hidden_dim,
        max_token_bytes=max_token_bytes,
        byte_position_config=TextLearnedPositionalEmbeddingConfig(
            num_embeddings=max_token_bytes + 1,
            embedding_dim=byte_embedding_dim,
        ),
        encoder_config=TransformerConfig(
            encoder_stack_config=LayerStackConfig(
                input_dim=byte_embedding_dim,
                hidden_dim=byte_embedding_dim,
                output_dim=byte_embedding_dim,
                num_layers=byte_encoder_num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_postprocessing_flag=True,
                layer_config=TransformerEncoderBlockLayerConfig(
                    activation=ActivationOptions.DISABLED,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    layer_model_config=encoder_layer,
                ),
            ),
        ),
        projection_config=_linear_stack(byte_embedding_dim, hidden_dim),
    )
