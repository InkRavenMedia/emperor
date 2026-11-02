"""Reduce each sequence chunk to one context for the memory generator."""

from dataclasses import replace

import torch
from torch import Tensor
from torch.nn import functional as F

from emperor.attention._config import AttentionSourceMemoryConfig
from emperor.attention._validation import AttentionSourceMemoryValidator
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    GroupingConfig,
)
from emperor.nn import Module


class MemoryContext(Module):
    VALIDATOR = AttentionSourceMemoryValidator

    def __init__(self, cfg: AttentionSourceMemoryConfig, embedding_dim: int) -> None:
        super().__init__()
        self.cfg: AttentionSourceMemoryConfig = cfg
        self.VALIDATOR.validate_context_config(self.cfg)
        self.embedding_dim: int = embedding_dim
        self.grouping_config: GroupingConfig = self.cfg.grouping_config
        self.chunk_size: int = self.grouping_config.chunk_size
        self.grouper = self.__build_grouper()

    def __build_grouper(self) -> Module:
        grouping_config = replace(
            self.grouping_config,
            scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
            sequence_length=1,
            input_order=AdaptiveParameterInputOrderOptions.BATCH_FIRST,
            feature_dim=self.embedding_dim,
        )
        return grouping_config.build()

    def forward(
        self, source: Tensor, skip_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        sequence_length, batch_size, _ = source.shape
        num_context_chunks = (sequence_length + self.chunk_size - 1) // self.chunk_size
        grouped_tokens = self.__group_source_tokens(source)
        valid_members = self.__group_valid_members(
            source, skip_mask, num_context_chunks
        )
        context_embeddings, valid_context_mask = self.__summarize_valid_chunks(
            grouped_tokens, valid_members
        )
        # One context per chunk, in batch-first layout: [B, G, D] and [B, G].
        return (
            context_embeddings.reshape(
                batch_size, num_context_chunks, self.embedding_dim
            ),
            valid_context_mask.reshape(batch_size, num_context_chunks),
        )

    def __group_source_tokens(self, source: Tensor) -> Tensor:
        sequence_length, _, feature_dim = source.shape
        sequence_token_rows = source.transpose(0, 1).reshape(-1, feature_dim)
        context_grouping_plan = self.grouper.split(
            sequence_token_rows, sequence_length=sequence_length
        )
        return context_grouping_plan.grouped_members

    def __group_valid_members(
        self, source: Tensor, skip_mask: Tensor | None, num_context_chunks: int
    ) -> Tensor:
        sequence_length, batch_size, _ = source.shape
        valid_token_mask = torch.ones(
            (batch_size, sequence_length), dtype=torch.bool, device=source.device
        )
        if skip_mask is not None:
            valid_token_mask = (
                skip_mask.reshape(sequence_length, batch_size).transpose(0, 1).bool()
            )
        tail_padding_length = num_context_chunks * self.chunk_size - sequence_length
        padded_valid_token_mask = F.pad(
            valid_token_mask, (0, tail_padding_length), value=False
        )
        return padded_valid_token_mask.reshape(
            batch_size * num_context_chunks, self.chunk_size
        )

    def __summarize_valid_chunks(
        self, grouped_tokens: Tensor, valid_members: Tensor
    ) -> tuple[Tensor, Tensor]:
        valid_context_mask = valid_members.any(dim=1)
        context_embeddings = grouped_tokens.new_zeros(
            (grouped_tokens.size(0), self.embedding_dim)
        )
        if valid_context_mask.any():
            context_embeddings[valid_context_mask] = self.grouper.summarize(
                grouped_tokens[valid_context_mask], valid_members[valid_context_mask]
            )
        return context_embeddings, valid_context_mask
