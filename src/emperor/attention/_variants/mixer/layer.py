"""Private mixer-attention layer implementation."""

from copy import deepcopy
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from emperor.attention._memory.config import AttentionSourceMemoryConfig
from emperor.attention._variants.mixer.validation import (
    MixerAttentionValidator,
)
from emperor.layers import Layer
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.attention._variants.mixer.config import MixerAttentionConfig
    from emperor.config import ModelConfig
    from emperor.experts import MixtureOfExpertsModelConfig
    from emperor.layers import LayerStackConfig, RecurrentCompositionConfig


class MixerAttention(Module):
    """Apply one shared configured model across every token-axis vector."""

    VALIDATOR = MixerAttentionValidator

    def __init__(
        self,
        cfg: "MixerAttentionConfig | ModelConfig",
        overrides: "MixerAttentionConfig | None" = None,
    ) -> None:
        super().__init__()
        config = getattr(cfg, "mixer_attention_config", cfg)
        self.cfg: MixerAttentionConfig = self._override_config(config, overrides)

        self.embedding_dim: int = self.cfg.embedding_dim
        self.sequence_length: int = self.cfg.sequence_length
        self.batch_first_flag: bool = self.cfg.batch_first_flag
        self.variable_sequence_length_flag: bool = bool(
            self.cfg.variable_sequence_length_flag
        )
        self.mixing_model_config: (
            LayerStackConfig | MixtureOfExpertsModelConfig | RecurrentCompositionConfig
        ) = self.cfg.mixing_model_config

        self.VALIDATOR.validate(self)
        memory_config = self.cfg.source_memory_config
        self.chunked_source_memory = isinstance(
            memory_config, AttentionSourceMemoryConfig
        )
        self.memory_sequence_length = 0
        if memory_config is not None:
            chunk_size = (
                memory_config.grouping_config.chunk_size
                if self.chunked_source_memory
                else 1
            )
            self.memory_sequence_length = (
                self.sequence_length + chunk_size - 1
            ) // chunk_size
        self.mixing_sequence_length = self.sequence_length + self.memory_sequence_length
        exact_mixing_config = self.__exact_mixing_model_config(self.mixing_model_config)
        self.mixing_model = exact_mixing_config.build()
        self.uses_soft_moe = getattr(self.mixing_model, "uses_soft_moe", False)
        self.source_memory = (
            self._build_from_config(
                self.cfg.source_memory_config,
                **(
                    {"embedding_dim": self.embedding_dim}
                    if self.chunked_source_memory
                    else {
                        "input_dim": self.embedding_dim,
                        "output_dim": self.embedding_dim,
                    }
                ),
            )
            if self.cfg.source_memory_config is not None
            else None
        )

    def __exact_mixing_model_config(self, config):
        from emperor.experts import MixtureOfExpertsModelConfig
        from emperor.layers import RecurrentCompositionConfig

        exact_config = deepcopy(config)
        exact_config.input_dim = self.mixing_sequence_length
        exact_config.output_dim = self.mixing_sequence_length
        if isinstance(exact_config, MixtureOfExpertsModelConfig):
            exact_config.stack_config = self.__exact_mixing_model_config(
                exact_config.stack_config
            )
        elif isinstance(exact_config, RecurrentCompositionConfig):
            exact_config._map_transition_configs(self.__exact_mixing_model_config)
        return exact_config

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        k_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        static_k: Tensor | None = None,
        static_v: Tensor | None = None,
    ) -> tuple[Tensor, None, Tensor | None]:
        self.VALIDATOR.validate_forward_inputs(
            self, q, k, v, k_padding_mask, attention_mask, static_k, static_v
        )

        sequence_axis = 1 if self.batch_first_flag else 0
        input_sequence_length = q.size(sequence_axis)
        token_vectors, excluded, preparation_loss = self.__prepare_mixing_input(
            q, k_padding_mask, sequence_axis
        )
        leading_shape = token_vectors.shape[:-1]
        if self.uses_soft_moe:
            from emperor.experts import MixtureOfExpertsLayerState

            # Route channel vectors within each sample; each expert mixes tokens.
            mixing_input = token_vectors
            mixing_state = self.mixing_model(
                MixtureOfExpertsLayerState(hidden=mixing_input)
            )
        else:
            mixing_input = token_vectors.reshape(-1, self.mixing_sequence_length)
            mixing_state = Layer.run_model_from_hidden(self.mixing_model, mixing_input)
        self.VALIDATOR.validate_mixing_state(
            mixing_state,
            mixing_input,
        )

        mixed_vectors = mixing_state.hidden.reshape(
            *leading_shape,
            self.mixing_sequence_length,
        )
        mixed = mixed_vectors.movedim(-1, sequence_axis)
        mixed = mixed.narrow(sequence_axis, 0, input_sequence_length)
        if excluded is not None:
            mixed = mixed.masked_fill(excluded.unsqueeze(-1), 0)
        loss = mixing_state.loss
        if preparation_loss is not None:
            loss = preparation_loss if loss is None else loss + preparation_loss
        return mixed, None, loss

    def __prepare_mixing_input(
        self, q: Tensor, k_padding_mask: Tensor | None, sequence_axis: int
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        excluded = None
        if k_padding_mask is not None:
            excluded = (
                k_padding_mask
                if k_padding_mask.dtype == torch.bool
                else k_padding_mask.isneginf()
            )
            if not self.batch_first_flag:
                excluded = excluded.transpose(0, 1)
            q = q.masked_fill(excluded.unsqueeze(-1), 0)
        token_vectors = self.__token_vectors(q, sequence_axis)
        if self.source_memory is not None:
            if self.chunked_source_memory:
                source = q.transpose(0, 1) if self.batch_first_flag else q
                active_rows = (
                    None
                    if excluded is None
                    else ~(
                        excluded.transpose(0, 1) if self.batch_first_flag else excluded
                    ).reshape(-1, 1)
                )
                memory, _, memory_loss = self.source_memory(
                    source, skip_mask=active_rows
                )
                memory = memory.reshape(-1, source.size(1), self.embedding_dim)
                if self.batch_first_flag:
                    memory = memory.transpose(0, 1)
                memory_vectors = memory.movedim(sequence_axis, -1)
                memory_vectors = F.pad(
                    memory_vectors,
                    (0, self.memory_sequence_length - memory_vectors.size(-1)),
                )
                token_vectors = torch.cat((token_vectors, memory_vectors), dim=-1)
                return token_vectors, excluded, memory_loss
            # Per-token rows keep optional adaptation independent across tokens.
            token_rows = q.reshape(-1, self.embedding_dim)
            memory = self.source_memory(token_rows)
            self.VALIDATOR.validate_output(memory, token_rows)
            memory = memory.reshape_as(q)
            if excluded is not None:
                memory = memory.masked_fill(excluded.unsqueeze(-1), 0)
            token_vectors = torch.cat(
                (token_vectors, self.__token_vectors(memory, sequence_axis)), dim=-1
            )
        return token_vectors, excluded, None

    def __token_vectors(self, tokens: Tensor, sequence_axis: int) -> Tensor:
        vectors = tokens.movedim(sequence_axis, -1)
        unused_positions = self.sequence_length - vectors.size(-1)
        return F.pad(vectors, (0, unused_positions)) if unused_positions else vectors
