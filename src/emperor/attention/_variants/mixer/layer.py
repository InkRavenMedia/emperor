"""Private mixer-attention layer implementation."""

from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.attention._validation import AttentionSourceMemoryValidator
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
        self.mixing_model_config: (
            LayerStackConfig | MixtureOfExpertsModelConfig | RecurrentCompositionConfig
        ) = self.cfg.mixing_model_config

        self.VALIDATOR.validate(self)
        self.mixing_sequence_length = self.sequence_length * (
            2 if self.cfg.source_memory_config is not None else 1
        )
        exact_mixing_config = self.__exact_mixing_model_config(self.mixing_model_config)
        self.mixing_model = exact_mixing_config.build()
        self.uses_soft_moe = getattr(self.mixing_model, "uses_soft_moe", False)
        self.source_memory = (
            self._build_from_config(
                self.cfg.source_memory_config,
                input_dim=self.embedding_dim,
                output_dim=self.embedding_dim,
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
        if self.source_memory is not None:
            # Per-token rows keep optional adaptation independent across tokens.
            token_rows = q.reshape(-1, self.embedding_dim)
            memory = self.source_memory(token_rows)
            AttentionSourceMemoryValidator.validate_output(memory, token_rows)
            q = torch.cat((q, memory.reshape_as(q)), dim=sequence_axis)
        token_vectors = q.movedim(sequence_axis, -1)
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
        mixing_state = self.VALIDATOR.validate_mixing_state(
            mixing_state,
            mixing_input,
        )

        mixed_vectors = mixing_state.hidden.reshape(
            *leading_shape,
            self.mixing_sequence_length,
        )
        mixed = mixed_vectors.movedim(-1, sequence_axis)
        if self.source_memory is not None:
            mixed = mixed.narrow(sequence_axis, 0, self.sequence_length)
        return mixed, None, mixing_state.loss
