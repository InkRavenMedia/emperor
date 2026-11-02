"""Private attention masking operations."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from emperor.attention._ops.memory_mask import GroupedMemoryMask
from emperor.attention._runtime import AttentionRuntimeLayout
from emperor.attention._validation import AttentionValidatorBase

if TYPE_CHECKING:
    from emperor.attention._config import MultiHeadAttentionConfig
    from emperor.attention._runtime import MultiHeadAttentionInputs


class Mask:
    VALIDATOR = AttentionValidatorBase

    def __init__(self, cfg: MultiHeadAttentionConfig) -> None:
        self.cfg = cfg
        self.num_heads = self.cfg.num_heads
        self.target_dtype = self.cfg.target_dtype
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag
        self.source_memory_flag: bool = self.cfg.source_memory_flag
        self.grouped_memory_mask = (
            GroupedMemoryMask(self.cfg.source_memory_config, num_heads=self.num_heads)
            if self.source_memory_flag
            else None
        )
        self.query_dtype: torch.dtype | None = None
        self.query_device: torch.device | None = None

    def prepare_attention_masks(
        self,
        attention_inputs: MultiHeadAttentionInputs,
    ) -> MultiHeadAttentionInputs:
        query = attention_inputs.query
        runtime_layout = attention_inputs.runtime_layout
        self.VALIDATOR.validate_attention_mask_preparation_runtime_layout(
            runtime_layout
        )
        runtime_layout = cast(AttentionRuntimeLayout, runtime_layout)
        self.__set_runtime_query_properties(query)
        key_padding_mask, attention_mask = self.__process_attention_masks(
            attention_inputs.key_padding_mask,
            attention_inputs.attention_mask,
            runtime_layout,
        )
        attention_mask = self.__resolve_causal_attention_mask(
            attention_mask, runtime_layout
        )
        attention_mask = self.__validate_attention_mask(attention_mask)
        self.__clear_runtime_query_properties()
        if (
            key_padding_mask is attention_inputs.key_padding_mask
            and attention_mask is attention_inputs.attention_mask
        ):
            return attention_inputs
        return replace(
            attention_inputs,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )

    def prepare_masks(
        self, attention_inputs: MultiHeadAttentionInputs
    ) -> tuple[MultiHeadAttentionInputs, Tensor | None]:
        if not self.source_memory_flag:
            return self.prepare_attention_masks(attention_inputs), None
        mask = attention_inputs.attention_mask
        source_length = attention_inputs.key.size(0)
        chunk_size = self.grouped_memory_mask.chunk_size
        memory_length = (source_length + chunk_size - 1) // chunk_size
        if mask is None or mask.size(-1) != source_length + memory_length:
            return self.prepare_attention_masks(attention_inputs), None
        token_mask, memory_mask = mask.split((source_length, memory_length), dim=-1)
        prepared_tokens = self.prepare_attention_masks(
            replace(attention_inputs, attention_mask=token_mask)
        )
        memory_layout = replace(
            attention_inputs.runtime_layout, source_sequence_length=memory_length
        )
        self._validate_mask_shapes(None, memory_mask, memory_layout)
        self.__set_runtime_query_properties(attention_inputs.query)
        prepared_memory = self.__canonical_mask(memory_mask, "attention_mask")
        self.__clear_runtime_query_properties()
        return prepared_tokens, prepared_memory

    def extend_source_memory_masks(
        self,
        attention_inputs: MultiHeadAttentionInputs,
        memory_mask: Tensor | None = None,
    ) -> MultiHeadAttentionInputs:
        if not self.source_memory_flag:
            return attention_inputs
        return self.grouped_memory_mask.extend_masks(attention_inputs, memory_mask)

    def __set_runtime_query_properties(self, query: Tensor) -> None:
        self.query_dtype = query.dtype
        self.query_device = query.device

    def __clear_runtime_query_properties(self) -> None:
        self.query_dtype = None
        self.query_device = None

    def __resolve_causal_attention_mask(
        self,
        attention_mask: Tensor | None,
        runtime_layout: AttentionRuntimeLayout,
    ) -> Tensor | None:
        if not self.causal_attention_mask_flag:
            return attention_mask
        target_length = runtime_layout.target_sequence_length
        source_length = runtime_layout.source_sequence_length
        causal_mask = self.__generate_causal_mask(target_length, source_length)
        if attention_mask is None:
            return causal_mask
        return attention_mask.masked_fill(causal_mask.isneginf(), -torch.inf)

    def __generate_causal_mask(
        self,
        target_length: int,
        source_length: int,
    ) -> Tensor:
        causal_mask_shape = (target_length, source_length)
        negative_infinity_tensor = torch.full(
            causal_mask_shape,
            -torch.inf,
            dtype=self.query_dtype,
            device=self.query_device,
        )
        return torch.triu(negative_infinity_tensor, diagonal=1)

    def __process_attention_masks(
        self,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
        runtime_layout: AttentionRuntimeLayout,
    ) -> tuple[Tensor | None, Tensor | None]:
        self._validate_mask_shapes(
            key_padding_mask,
            attention_mask,
            runtime_layout,
        )
        prepared_key_padding_mask = self.__canonical_mask(
            key_padding_mask,
            "key_padding_mask",
        )
        prepared_attention_mask = self.__canonical_mask(
            attention_mask,
            "attention_mask",
        )
        return prepared_key_padding_mask, prepared_attention_mask

    def _validate_mask_shapes(
        self,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
        runtime_layout: AttentionRuntimeLayout,
    ) -> None:
        batch_size = runtime_layout.batch_size
        target_length = runtime_layout.target_sequence_length
        source_length = runtime_layout.source_sequence_length
        standard_branch_count = runtime_layout.branch_count(self.num_heads)
        expected_key_padding_shape = (batch_size, source_length)
        expected_attention_sequence_shape = (target_length, source_length)
        self.VALIDATOR.validate_mask_shapes(
            key_padding_mask,
            attention_mask,
            expected_key_padding_shape=expected_key_padding_shape,
            expected_attention_sequence_shape=expected_attention_sequence_shape,
            standard_branch_count=standard_branch_count,
        )

    def __canonical_mask(
        self,
        mask: Tensor | None,
        mask_name: str,
    ) -> Tensor | None:
        if mask is None:
            return None
        self.VALIDATOR.validate_mask_is_float_or_bool(mask, mask_name)
        if torch.is_floating_point(mask):
            return mask.to(dtype=self.query_dtype, device=self.query_device)
        boolean_mask = mask.to(device=self.query_device)
        placeholder = torch.zeros_like(boolean_mask, dtype=self.query_dtype)
        return placeholder.masked_fill_(boolean_mask, -torch.inf)

    def __validate_attention_mask(
        self,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        self.VALIDATOR.validate_attention_mask_for_required_causal_mask(
            attention_mask, self.causal_attention_mask_flag
        )
        return self.__canonical_mask(attention_mask, "attention_mask")

    def merge_padding_and_attention_mask(
        self,
        attention_inputs: MultiHeadAttentionInputs,
    ) -> MultiHeadAttentionInputs:
        runtime_layout = attention_inputs.runtime_layout
        self.VALIDATOR.validate_attention_mask_merging_runtime_layout(runtime_layout)
        runtime_layout = cast(AttentionRuntimeLayout, runtime_layout)
        key_padding_mask = self.__expand_key_padding_mask_across_heads(
            attention_inputs.key,
            attention_inputs.key_padding_mask,
            runtime_layout,
        )
        merged_attention_mask = self.__combine_padding_and_attention_masks(
            key_padding_mask,
            attention_inputs.attention_mask,
        )
        return replace(
            attention_inputs,
            merged_attention_mask=merged_attention_mask,
        )

    def __expand_key_padding_mask_across_heads(
        self,
        key: Tensor,
        key_padding_mask: Tensor | None,
        runtime_layout: AttentionRuntimeLayout,
    ) -> Tensor | None:
        if key_padding_mask is None:
            return None
        batch_size = runtime_layout.batch_size
        source_sequence_length = key.size(1)
        repeated_key_padding_mask = key_padding_mask.repeat_interleave(
            self.num_heads, dim=0
        )
        branch_count = batch_size * self.num_heads
        expanded_mask_shape = (branch_count, 1, source_sequence_length)
        return repeated_key_padding_mask.reshape(expanded_mask_shape)

    @staticmethod
    def __combine_padding_and_attention_masks(
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        if key_padding_mask is None:
            return attention_mask
        if attention_mask is None:
            return key_padding_mask
        return attention_mask + key_padding_mask
