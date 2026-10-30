"""Append a single, forward-local memory result to attention keys and values."""

from dataclasses import replace

import torch
from torch import Tensor

from emperor.attention._config import MultiHeadAttentionConfig
from emperor.attention._ops.masking import Mask
from emperor.attention._runtime import MultiHeadAttentionInputs
from emperor.attention._validation import AttentionSourceMemoryValidator
from emperor.nn import Module


class AttentionSourceMemory(Module):
    VALIDATOR = AttentionSourceMemoryValidator

    def __init__(self, cfg: MultiHeadAttentionConfig):
        super().__init__()
        self.VALIDATOR.validate_config(cfg)
        self.num_heads = cfg.num_heads
        self.model = self._build_from_config(
            cfg.source_memory_config,
            input_dim=cfg.embedding_dim,
            output_dim=cfg.embedding_dim,
        )

    def forward(
        self,
        inputs: MultiHeadAttentionInputs,
        source: Tensor,
        memory_mask: Tensor | None = None,
    ) -> MultiHeadAttentionInputs:
        sequence_length, batch_size, dimension = source.shape
        # TTT adapts each leading row independently: never adapt a whole sequence.
        token_rows = source.reshape(-1, dimension)
        memory = self.model(token_rows)
        self.VALIDATOR.validate_output(memory, token_rows)
        memory = memory.reshape(sequence_length, batch_size, self.num_heads, -1)
        expert_count = inputs.key.size(0) // (batch_size * self.num_heads)
        memory = memory.permute(1, 2, 0, 3).unsqueeze(1)
        memory = memory.expand(-1, expert_count, -1, -1, -1).reshape(
            inputs.key.size(0), sequence_length, dimension // self.num_heads
        )
        return replace(
            inputs,
            key=torch.cat((inputs.key, memory), dim=1),
            value=torch.cat((inputs.value, memory), dim=1),
            attention_mask=self._extend_mask(inputs.attention_mask, memory_mask),
            key_padding_mask=self._extend_mask(inputs.key_padding_mask),
            runtime_layout=inputs.runtime_layout.with_source_extension(sequence_length),
        )

    @staticmethod
    def _extend_mask(
        mask: Tensor | None, memory_mask: Tensor | None = None
    ) -> Tensor | None:
        if mask is None:
            return None
        if memory_mask is None:
            memory_mask = mask
        else:
            # An explicit memory block may restrict/bias memory independently, but
            # cannot reopen a source excluded by the original token mask.
            memory_mask = memory_mask.masked_fill(mask.isneginf(), -torch.inf)
        return torch.cat((mask, memory_mask), dim=-1)

    @staticmethod
    def prepare_masks(
        inputs: MultiHeadAttentionInputs, masks: Mask
    ) -> tuple[MultiHeadAttentionInputs, Tensor | None]:
        AttentionSourceMemoryValidator.validate_inputs(inputs)
        mask = inputs.attention_mask
        source_length = inputs.key.size(0)
        if mask is None or mask.size(-1) != 2 * source_length:
            return masks.prepare_attention_masks(inputs), None
        token_mask, memory_mask = mask.split(source_length, dim=-1)
        prepared_memory = masks.prepare_attention_masks(
            replace(inputs, attention_mask=memory_mask)
        )
        prepared_tokens = masks.prepare_attention_masks(
            replace(inputs, attention_mask=token_mask)
        )
        return prepared_tokens, prepared_memory.attention_mask
