from dataclasses import replace
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from emperor.attention._memory.config import AttentionSourceMemoryConfig
from emperor.attention._runtime import AttentionRuntimeLayout, MultiHeadAttentionInputs


class GroupedMemoryMask:
    def __init__(self, cfg: AttentionSourceMemoryConfig, *, num_heads: int) -> None:
        self.cfg: AttentionSourceMemoryConfig = cfg
        self.num_heads: int = num_heads
        self.chunk_size: int = self.cfg.grouping_config.chunk_size

    def split_attention_mask(
        self, attention_inputs: MultiHeadAttentionInputs
    ) -> tuple[Tensor | None, Tensor | None]:
        attention_mask = attention_inputs.attention_mask
        source_length = attention_inputs.key.size(0)
        memory_length = (source_length + self.chunk_size - 1) // self.chunk_size
        if (
            attention_mask is None
            or attention_mask.size(-1) != source_length + memory_length
        ):
            return attention_mask, None
        token_mask, memory_mask = attention_mask.split(
            (source_length, memory_length), dim=-1
        )
        return token_mask, memory_mask

    def extend_masks(
        self,
        attention_inputs: MultiHeadAttentionInputs,
        explicit_memory_mask: Tensor | None,
    ) -> MultiHeadAttentionInputs:
        runtime_layout = cast(AttentionRuntimeLayout, attention_inputs.runtime_layout)
        key_padding_mask = attention_inputs.key_padding_mask
        if key_padding_mask is None:
            key_padding_mask = attention_inputs.key.new_zeros(
                (runtime_layout.batch_size, runtime_layout.real_source_sequence_length)
            )
        valid_context_members = self.__group_valid_context_members(key_padding_mask)
        extended_padding_mask = self.__extend_padding_mask(
            key_padding_mask, valid_context_members
        )
        token_attention_mask = attention_inputs.attention_mask
        if token_attention_mask is None and explicit_memory_mask is None:
            return replace(attention_inputs, key_padding_mask=extended_padding_mask)
        extended_attention_mask = self.__extend_attention_mask(
            token_attention_mask,
            explicit_memory_mask,
            key_padding_mask,
            valid_context_members,
            runtime_layout,
        )
        return replace(
            attention_inputs,
            key_padding_mask=extended_padding_mask,
            attention_mask=extended_attention_mask,
        )

    def __group_valid_context_members(self, key_padding_mask: Tensor) -> Tensor:
        batch_size, source_length = key_padding_mask.shape
        num_context_chunks = (source_length + self.chunk_size - 1) // self.chunk_size
        tail_padding_length = num_context_chunks * self.chunk_size - source_length
        valid_token_mask = ~key_padding_mask.isneginf()
        padded_valid_token_mask = F.pad(
            valid_token_mask, (0, tail_padding_length), value=False
        )
        return padded_valid_token_mask.reshape(
            batch_size, num_context_chunks, self.chunk_size
        )

    def __extend_padding_mask(
        self, key_padding_mask: Tensor, valid_context_members: Tensor
    ) -> Tensor:
        batch_size, num_context_chunks, _ = valid_context_members.shape
        valid_memory_mask = valid_context_members.any(-1)
        memory_padding_mask = key_padding_mask.new_zeros(
            (batch_size, num_context_chunks)
        )
        memory_padding_mask = memory_padding_mask.masked_fill(
            ~valid_memory_mask, -torch.inf
        )
        return torch.cat((key_padding_mask, memory_padding_mask), dim=-1)

    def __extend_attention_mask(
        self,
        token_attention_mask: Tensor | None,
        explicit_memory_mask: Tensor | None,
        key_padding_mask: Tensor,
        valid_context_members: Tensor,
        runtime_layout: AttentionRuntimeLayout,
    ) -> Tensor:
        branch_count = self.__resolve_branch_count(
            token_attention_mask, explicit_memory_mask, runtime_layout.batch_size
        )
        source_length = runtime_layout.real_source_sequence_length
        memory_length = valid_context_members.size(1)
        expanded_token_mask = self.__expand_attention_mask(
            token_attention_mask,
            key_padding_mask,
            runtime_layout,
            branch_count=branch_count,
            source_length=source_length,
        )
        hidden_memory_mask = self.__hidden_memory_mask(
            expanded_token_mask, valid_context_members
        )
        expanded_memory_mask = self.__expand_attention_mask(
            explicit_memory_mask,
            key_padding_mask,
            runtime_layout,
            branch_count=branch_count,
            source_length=memory_length,
        )
        memory_attention_mask = expanded_memory_mask.masked_fill(
            hidden_memory_mask, -torch.inf
        )
        extended_attention_mask = torch.cat(
            (expanded_token_mask, memory_attention_mask), dim=-1
        )
        return extended_attention_mask.reshape(
            branch_count,
            runtime_layout.target_sequence_length,
            source_length + memory_length,
        )

    def __resolve_branch_count(
        self,
        token_attention_mask: Tensor | None,
        memory_attention_mask: Tensor | None,
        batch_size: int,
    ) -> int:
        return max(
            batch_size * self.num_heads,
            *(
                mask.size(0) if mask is not None and mask.dim() == 3 else 1
                for mask in (token_attention_mask, memory_attention_mask)
            ),
        )

    def __expand_attention_mask(
        self,
        attention_mask: Tensor | None,
        key_padding_mask: Tensor,
        runtime_layout: AttentionRuntimeLayout,
        *,
        branch_count: int,
        source_length: int,
    ) -> Tensor:
        batch_size = runtime_layout.batch_size
        query_length = runtime_layout.target_sequence_length
        branches_per_sample = branch_count // batch_size
        expanded_mask_shape = (
            batch_size,
            branches_per_sample,
            query_length,
            source_length,
        )
        if attention_mask is None:
            return key_padding_mask.new_zeros(expanded_mask_shape)
        if attention_mask.dim() == 2 or attention_mask.size(0) == 1:
            shared_attention_mask = attention_mask.reshape(
                1, 1, query_length, source_length
            )
            return shared_attention_mask.expand(expanded_mask_shape)
        masks_per_sample = attention_mask.size(0) // batch_size
        per_sample_mask = attention_mask.reshape(
            batch_size, masks_per_sample, query_length, source_length
        )
        if masks_per_sample != branches_per_sample:
            expert_branches_per_mask = branches_per_sample // masks_per_sample
            per_sample_mask = per_sample_mask.unsqueeze(1).expand(
                -1, expert_branches_per_mask, -1, -1, -1
            )
        return per_sample_mask.reshape(expanded_mask_shape)

    def __hidden_memory_mask(
        self, expanded_token_mask: Tensor, valid_context_members: Tensor
    ) -> Tensor:
        batch_size, branches_per_sample, query_length, source_length = (
            expanded_token_mask.shape
        )
        num_context_chunks = valid_context_members.size(1)
        tail_padding_length = num_context_chunks * self.chunk_size - source_length
        hidden_token_mask = F.pad(
            expanded_token_mask.isneginf(), (0, tail_padding_length), value=False
        )
        hidden_context_members = hidden_token_mask.reshape(
            batch_size,
            branches_per_sample,
            query_length,
            num_context_chunks,
            self.chunk_size,
        )
        return (hidden_context_members & valid_context_members[:, None, None]).any(-1)
