"""Explicit HAT v2 reference composition from reusable Emperor primitives."""


def hat_v2_reference_config(
    *,
    byte_dim: int,
    backbone_dim: int,
    byte_heads: int,
    backbone_heads: int,
    byte_layers: int,
    backbone_layers: int,
    max_token_bytes: int,
    sequence_length: int,
    batch_size: int,
    ffn_multiple: int = 256,
):
    """Build the Llama-family reference, without choosing an experiment scale.

    RoPE base 10000, RMS epsilon 1e-5 and FFN rounding follow original Llama;
    HAT v2 does not separately specify these numerical defaults. Context and
    byte limits are execution bounds, not implicit word-boundary rules.
    """
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

    from ._config import HierarchicalLanguageModelConfig

    dimensions = dict(
        byte_dim=byte_dim,
        backbone_dim=backbone_dim,
        byte_heads=byte_heads,
        backbone_heads=backbone_heads,
        byte_layers=byte_layers,
        backbone_layers=backbone_layers,
        max_token_bytes=max_token_bytes,
        sequence_length=sequence_length,
        batch_size=batch_size,
        ffn_multiple=ffn_multiple,
    )
    for name, value in dimensions.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    def projection(input_dim, output_dim):
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim,
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

    def transformer(dimension, heads, depth, maximum, maximum_batch, causal):
        attention = SelfAttentionConfig(
            batch_size=maximum_batch,
            num_heads=heads,
            embedding_dim=dimension,
            query_key_projection_dim=dimension,
            value_projection_dim=dimension,
            source_sequence_length=maximum,
            target_sequence_length=maximum,
            target_dtype=torch.float32,
            dropout_probability=0.0,
            zero_attention_flag=False,
            causal_attention_mask_flag=causal,
            add_key_value_bias_flag=False,
            average_attention_weights_flag=False,
            return_attention_weights_flag=False,
            batch_first_flag=True,
            projection_model_config=projection(dimension, dimension),
            projection_strategy=SelfAttentionProjectionStrategy.FUSED,
            rotary_base=10000.0,
        )
        intermediate_dim = (
            (8 * dimension // 3 + ffn_multiple - 1) // ffn_multiple
        ) * ffn_multiple
        arguments = dict(
            embedding_dim=dimension,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            dropout_probability=0.0,
            residual_config=AdditiveResidualConfig(),
            normalization=NormalizationOptions.RMS_NORM,
            feed_forward_config=SwiGLUFeedForwardConfig(
                input_dim=dimension,
                output_dim=dimension,
                hidden_dim=intermediate_dim,
            ),
        )
        layer = (
            TransformerDecoderLayerConfig(self_attention_config=attention, **arguments)
            if causal
            else TransformerEncoderLayerConfig(attention_config=attention, **arguments)
        )
        block = (
            TransformerDecoderBlockLayerConfig
            if causal
            else TransformerEncoderBlockLayerConfig
        )
        stack = LayerStackConfig(
            input_dim=dimension,
            hidden_dim=dimension,
            output_dim=dimension,
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

    maximum_words = batch_size * sequence_length
    backbone = transformer(
        backbone_dim, backbone_heads, backbone_layers, sequence_length, batch_size, True
    )
    return HierarchicalLanguageModelConfig(
        sequence_length=sequence_length,
        embedding_config=HierarchicalByteEmbeddingConfig(
            byte_embedding_dim=byte_dim,
            output_dim=backbone_dim,
            max_token_bytes=max_token_bytes,
            byte_position_config=None,
            encoder_config=transformer(
                byte_dim,
                byte_heads,
                byte_layers,
                max_token_bytes + 1,
                maximum_words,
                False,
            ),
            projection_config=projection(byte_dim, backbone_dim),
        ),
        decoding_config=HierarchicalByteDecoderConfig(
            conditioning_dim=backbone_dim,
            byte_embedding_dim=byte_dim,
            max_token_bytes=max_token_bytes,
            byte_position_config=None,
            decoder_config=transformer(
                byte_dim,
                byte_heads,
                byte_layers,
                max_token_bytes + 1,
                maximum_words,
                True,
            ),
            conditioning_projection_config=projection(backbone_dim, byte_dim),
            output_projection_config=projection(byte_dim, 256),
        ),
        backbone_config=backbone.decoder_stack_config,
        position_config=None,
        embedding_normalization_config=None,
        output_normalization_config=LayerConfig(
            input_dim=backbone_dim,
            output_dim=backbone_dim,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            normalization=NormalizationOptions.RMS_NORM,
        ),
        dropout_probability=0.0,
    )
