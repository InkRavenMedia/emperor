from dataclasses import replace
from typing import cast

import torch
from torch import Tensor

from emperor.attention import MixtureOfAttentionHeadsConfig, SelfAttentionConfig
from emperor.embedding.hierarchical import (
    HierarchicalByteDecoderConfig,
    HierarchicalByteEmbeddingConfig,
)
from emperor.embedding.hierarchical.decoder import (
    END_OF_DOCUMENT,
    END_OF_TOKEN,
    OUTPUT_SYMBOLS,
    HierarchicalByteDecoderOutput,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.nn import freeze_forward_schedules
from emperor.transformer import (
    TransformerConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)


def validate_masked_word_attention(*, masked_words: bool, causal: bool) -> None:
    if masked_words and causal:
        raise ValueError("Masked-word BERT requires bidirectional sequence attention.")


def _decoder_projection(input_dim: int, output_dim: int) -> LayerStackConfig:
    layer_model_config = LinearLayerConfig(bias_flag=True)
    layer_config = LayerConfig(
        activation=ActivationOptions.RELU,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        layer_model_config=layer_model_config,
    )
    return LayerStackConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=32,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=False,
        layer_config=layer_config,
    )


def build_masked_word_decoder(
    encoder: HierarchicalByteEmbeddingConfig,
) -> HierarchicalByteDecoderConfig:
    """Use the byte encoder's dimensions with causal, token-local attention."""
    transformer = cast(TransformerConfig, encoder.encoder_config)
    stack = cast(LayerStackConfig, transformer.encoder_stack_config)
    block = cast(TransformerEncoderBlockLayerConfig, stack.layer_config)
    layer = cast(TransformerEncoderLayerConfig, block.layer_model_config)
    embedding_dim = layer.embedding_dim
    encoder_attention = cast(
        SelfAttentionConfig | MixtureOfAttentionHeadsConfig, layer.attention_config
    )
    causal_attention_config = replace(
        encoder_attention, causal_attention_mask_flag=True
    )
    decoder_layer = TransformerDecoderLayerConfig(
        embedding_dim=embedding_dim,
        layer_norm_position=layer.layer_norm_position,
        dropout_probability=layer.dropout_probability,
        residual_config=layer.residual_config,
        normalization=layer.normalization,
        feed_forward_config=layer.feed_forward_config,
        self_attention_config=causal_attention_config,
    )
    layer_config = TransformerDecoderBlockLayerConfig(
        activation=block.activation,
        dropout_probability=block.dropout_probability,
        layer_norm_position=block.layer_norm_position,
        layer_model_config=decoder_layer,
    )
    decoder_stack = replace(
        stack,
        num_layers=1,
        layer_config=layer_config,
    )
    byte_dim = encoder.byte_embedding_dim
    decoder_config = TransformerConfig(decoder_stack_config=decoder_stack)
    conditioning_projection_config = _decoder_projection(encoder.output_dim, byte_dim)
    output_projection_config = _decoder_projection(byte_dim, OUTPUT_SYMBOLS)
    return HierarchicalByteDecoderConfig(
        conditioning_dim=encoder.output_dim,
        byte_embedding_dim=byte_dim,
        max_token_bytes=encoder.max_token_bytes,
        byte_position_config=encoder.byte_position_config,
        decoder_config=decoder_config,
        conditioning_projection_config=conditioning_projection_config,
        output_projection_config=output_projection_config,
    )


@torch.no_grad()
def greedy_word_matches(
    decoder, conditioning: Tensor, targets: Tensor, lengths: Tensor
):
    """Batch empty-prefix generation, with exactly one outcome per masked word."""
    if not conditioning.shape[0]:
        return lengths.flatten().bool()
    generated = torch.zeros_like(targets)
    generated_lengths = torch.zeros_like(lengths)
    active = torch.ones_like(lengths, dtype=torch.bool)
    complete = torch.zeros_like(active)
    with freeze_forward_schedules():
        for position in range(targets.shape[-1] + 1):
            output: HierarchicalByteDecoderOutput = decoder(
                conditioning, generated, generated_lengths, active
            )
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
