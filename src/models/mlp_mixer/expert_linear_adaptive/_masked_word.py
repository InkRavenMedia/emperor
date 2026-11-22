"""Mixer-owned byte boundary for hierarchical masked-word pretraining.

Byte-local encoding/decoding matches the reference task. Sentence context is
computed by this package's Mixer backbone, independently of any BERT package.
"""

from dataclasses import replace

import torch
from torch import Tensor

from emperor.attention import SelfAttentionConfig, SelfAttentionProjectionStrategy
from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
from emperor.embedding.hierarchical import (
    HierarchicalByteDecoderConfig,
    HierarchicalByteEmbeddingConfig,
)
from emperor.embedding.hierarchical.decoder import (
    END_OF_DOCUMENT,
    END_OF_TOKEN,
    OUTPUT_SYMBOLS,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.nn import freeze_forward_schedules
from emperor.transformer import (
    FeedForwardConfig,
    TransformerConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)

from ._building import encoder_config
from .experiment_config import MaskedWordExperimentConfig
from .runtime_options import RuntimeOptions


def _linear_stack(input_dim: int, output_dim: int, *, hidden_dim: int = 32):
    linear_layer_config = LinearLayerConfig(bias_flag=True)
    layer_config = LayerConfig(
        activation=ActivationOptions.RELU,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        layer_model_config=linear_layer_config,
    )
    return LayerStackConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=False,
        layer_config=layer_config,
    )


def build_experiment_config(runtime: RuntimeOptions) -> MaskedWordExperimentConfig:
    byte_dim = runtime.hierarchical_byte_embedding_dim
    max_bytes = runtime.hierarchical_max_token_bytes
    for name in (
        "hierarchical_byte_embedding_dim",
        "hierarchical_byte_encoder_num_layers",
        "hierarchical_byte_feed_forward_dim",
        "hierarchical_byte_num_heads",
        "hierarchical_max_token_bytes",
    ):
        value = getattr(runtime, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if byte_dim % runtime.hierarchical_byte_num_heads:
        raise ValueError(
            "hierarchical_byte_embedding_dim must be divisible by hierarchical_byte_num_heads"
        )
    if not 0 <= runtime.embedding_dropout_probability <= 1:
        raise ValueError("embedding_dropout_probability must be between 0 and 1")
    byte_attention_projection = _linear_stack(byte_dim, byte_dim)
    byte_attention = SelfAttentionConfig(
        batch_size=runtime.batch_size * runtime.sequence_length,
        num_heads=runtime.hierarchical_byte_num_heads,
        embedding_dim=byte_dim,
        query_key_projection_dim=0,
        value_projection_dim=0,
        target_sequence_length=max_bytes + 1,
        source_sequence_length=max_bytes + 1,
        target_dtype=torch.float32,
        dropout_probability=0.0,
        zero_attention_flag=False,
        causal_attention_mask_flag=False,
        add_key_value_bias_flag=False,
        source_memory_flag=False,
        average_attention_weights_flag=False,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=byte_attention_projection,
        projection_strategy=SelfAttentionProjectionStrategy.FUSED,
    )
    residual_config = AdditiveResidualConfig()
    byte_feed_forward_stack = _linear_stack(
        byte_dim,
        byte_dim,
        hidden_dim=runtime.hierarchical_byte_feed_forward_dim,
    )
    feed_forward_config = FeedForwardConfig(
        input_dim=byte_dim,
        output_dim=byte_dim,
        stack_config=byte_feed_forward_stack,
    )
    byte_layer = TransformerEncoderLayerConfig(
        embedding_dim=byte_dim,
        layer_norm_position=LayerNormPositionOptions.BEFORE,
        dropout_probability=0.0,
        residual_config=residual_config,
        attention_config=byte_attention,
        feed_forward_config=feed_forward_config,
    )
    encoder_block_config = TransformerEncoderBlockLayerConfig(
        activation=ActivationOptions.DISABLED,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        layer_model_config=byte_layer,
    )
    byte_stack = LayerStackConfig(
        input_dim=byte_dim,
        hidden_dim=byte_dim,
        output_dim=byte_dim,
        num_layers=runtime.hierarchical_byte_encoder_num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=True,
        layer_config=encoder_block_config,
    )
    byte_positions = TextLearnedPositionalEmbeddingConfig(
        num_embeddings=max_bytes + 1, embedding_dim=byte_dim
    )
    byte_encoder_config = TransformerConfig(encoder_stack_config=byte_stack)
    projection_config = _linear_stack(byte_dim, runtime.hidden_dim)
    embedding = HierarchicalByteEmbeddingConfig(
        byte_embedding_dim=byte_dim,
        output_dim=runtime.hidden_dim,
        max_token_bytes=max_bytes,
        byte_position_config=byte_positions,
        encoder_config=byte_encoder_config,
        projection_config=projection_config,
    )
    self_attention_config = replace(byte_attention, causal_attention_mask_flag=True)
    decoder_layer_config = TransformerDecoderLayerConfig(
        embedding_dim=byte_dim,
        layer_norm_position=byte_layer.layer_norm_position,
        dropout_probability=byte_layer.dropout_probability,
        residual_config=byte_layer.residual_config,
        normalization=byte_layer.normalization,
        feed_forward_config=byte_layer.feed_forward_config,
        self_attention_config=self_attention_config,
    )
    decoder_block_config = TransformerDecoderBlockLayerConfig(
        activation=ActivationOptions.DISABLED,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        layer_model_config=decoder_layer_config,
    )
    decoder_stack = replace(
        byte_stack,
        num_layers=1,
        layer_config=decoder_block_config,
    )
    decoder_config = TransformerConfig(decoder_stack_config=decoder_stack)
    conditioning_projection_config = _linear_stack(runtime.hidden_dim, byte_dim)
    output_projection_config = _linear_stack(byte_dim, OUTPUT_SYMBOLS)
    masked_word_decoder_config = HierarchicalByteDecoderConfig(
        conditioning_dim=runtime.hidden_dim,
        byte_embedding_dim=byte_dim,
        max_token_bytes=max_bytes,
        byte_position_config=byte_positions,
        decoder_config=decoder_config,
        conditioning_projection_config=conditioning_projection_config,
        output_projection_config=output_projection_config,
    )
    positional_embedding_config = TextLearnedPositionalEmbeddingConfig(
        num_embeddings=runtime.sequence_length,
        embedding_dim=runtime.hidden_dim,
        padding_idx=0,
    )
    mixer_encoder_config = encoder_config(runtime, runtime.sequence_length)
    return MaskedWordExperimentConfig(
        embedding_config=embedding,
        masked_word_decoder_config=masked_word_decoder_config,
        positional_embedding_config=positional_embedding_config,
        encoder_config=mixer_encoder_config,
        encoder_output_normalization=runtime.encoder_output_normalization,
        embedding_dropout_probability=runtime.embedding_dropout_probability,
    )


@torch.no_grad()
def greedy_word_matches(
    decoder, conditioning: Tensor, targets: Tensor, lengths: Tensor
):
    """Generate each word from an empty prefix and count exact byte matches."""
    if not conditioning.shape[0]:
        return lengths.flatten().bool()
    generated = torch.zeros_like(targets)
    generated_lengths = torch.zeros_like(lengths)
    active = torch.ones_like(lengths, dtype=torch.bool)
    complete = torch.zeros_like(active)
    with freeze_forward_schedules():
        for position in range(targets.shape[-1] + 1):
            output = decoder(conditioning, generated, generated_lengths, active)
            rows = active.flatten().nonzero().flatten()
            last_offsets = output.token_offsets[rows + 1] - 1
            symbols = output.logits[last_offsets].argmax(dim=-1)
            complete[rows, 0] = symbols == END_OF_TOKEN
            if position == targets.shape[-1]:
                break
            continuing = (symbols != END_OF_TOKEN) & (symbols != END_OF_DOCUMENT)
            next_rows = rows[continuing]
            if not next_rows.numel():
                break
            active.zero_()
            active[next_rows, 0] = True
            generated[next_rows, 0, position] = symbols[continuing]
            generated_lengths[next_rows, 0] += 1
    return (
        complete & (generated_lengths == lengths) & (generated == targets).all(dim=-1)
    ).flatten()
