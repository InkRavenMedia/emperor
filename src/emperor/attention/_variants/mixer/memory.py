"""Append memory-generated tokens to the Mixer's token-axis vectors."""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from emperor.attention._variants.mixer.config import MixerAttentionConfig
    from emperor.attention._variants.mixer.reshaping import MixerReshaper
    from emperor.memory import SourceMemoryInterface


class MixerSourceMemory:
    def __init__(
        self,
        cfg: "MixerAttentionConfig",
        reshaper: "MixerReshaper",
        memory_sequence_length: int,
    ) -> None:
        self.source_memory_config = cfg.source_memory_config
        self.embedding_dim = cfg.embedding_dim
        self.batch_first_flag = cfg.batch_first_flag
        self.reshaper = reshaper
        self.memory_sequence_length = memory_sequence_length

    def build_source_memory(self) -> "SourceMemoryInterface | None":
        if self.source_memory_config is None:
            return None
        return self.source_memory_config.build_source_memory(self.embedding_dim)

    def append_sources(
        self,
        source_memory: "SourceMemoryInterface | None",
        token_axis_vectors: Tensor,
        source_tokens: Tensor,
        excluded_token_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        if source_memory is None:
            return token_axis_vectors, None
        memory_tokens, memory_loss = source_memory.generate_source_memory(
            source_tokens,
            excluded_token_mask=excluded_token_mask,
            batch_first_flag=self.batch_first_flag,
        )
        memory_vectors = self.reshaper.prepare_memory_vectors(
            memory_tokens, self.memory_sequence_length
        )
        extended_token_vectors = torch.cat((token_axis_vectors, memory_vectors), dim=-1)
        return extended_token_vectors, memory_loss
