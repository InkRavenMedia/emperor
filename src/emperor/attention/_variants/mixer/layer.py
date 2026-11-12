"""Private mixer-attention layer implementation."""

from copy import deepcopy
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from emperor.attention._variants.mixer.masking import MixerMask
from emperor.attention._variants.mixer.memory import MixerSourceMemory
from emperor.attention._variants.mixer.reshaping import MixerReshaper
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
        self.reshaper = MixerReshaper(self.cfg)
        self.masks = MixerMask(self.reshaper)
        memory_config = self.cfg.source_memory_config
        self.memory_sequence_length = (
            memory_config.source_memory_length(self.sequence_length)
            if memory_config is not None
            else 0
        )
        self.mixing_sequence_length = self.sequence_length + self.memory_sequence_length
        exact_mixing_config = self.__exact_mixing_model_config(self.mixing_model_config)
        self.mixing_model = exact_mixing_config.build()
        self.uses_soft_moe = getattr(self.mixing_model, "uses_soft_moe", False)
        self.memory_handler = MixerSourceMemory(
            self.cfg, self.reshaper, self.memory_sequence_length
        )
        self.source_memory = self.memory_handler.build_source_memory()
        self.token_model = (
            self._build_from_config(
                self.cfg.token_model_config,
                input_dim=self.embedding_dim,
                output_dim=self.embedding_dim,
            )
            if self.cfg.token_model_config is not None
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

        input_sequence_length = q.size(self.reshaper.sequence_axis)
        token_vectors, excluded, preparation_loss = self.__prepare_mixing_input(
            q, k_padding_mask
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
        mixed = self.reshaper.restore_output_layout(
            mixed_vectors, input_sequence_length
        )
        mixed = self.masks.apply_to_tokens(mixed, excluded)
        loss = mixing_state.loss
        if preparation_loss is not None:
            loss = preparation_loss if loss is None else loss + preparation_loss
        return mixed, None, loss

    def __prepare_mixing_input(
        self, q: Tensor, k_padding_mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        q, excluded = self.masks.prepare_inputs(q, k_padding_mask)
        projected, preparation_loss = self.__project_tokens(q, excluded)
        token_vectors = self.reshaper.prepare_token_axis_vectors(projected)
        token_vectors, memory_loss = self.memory_handler.append_sources(
            self.source_memory, token_vectors, q, excluded
        )
        if memory_loss is not None:
            preparation_loss = (
                memory_loss if preparation_loss is None else preparation_loss + memory_loss
            )
        return token_vectors, excluded, preparation_loss

    def __project_tokens(
        self, tokens: Tensor, excluded: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        if self.token_model is None:
            return tokens, None
        from emperor.experts import MixtureOfExpertsLayerState

        tokens = tokens if self.batch_first_flag else tokens.transpose(0, 1)
        active = (
            None
            if excluded is None
            else ~(
                excluded if self.batch_first_flag else excluded.transpose(0, 1)
            ).unsqueeze(-1)
        )
        input_length = tokens.size(1)
        unused_positions = self.sequence_length - input_length
        if unused_positions:
            if active is None:
                active = torch.ones(
                    (*tokens.shape[:2], 1), dtype=torch.bool, device=tokens.device
                )
            tokens = F.pad(tokens, (0, 0, 0, unused_positions))
            active = F.pad(active, (0, 0, 0, unused_positions), value=False)
        state = self.token_model(
            MixtureOfExpertsLayerState(hidden=tokens, skip_mask=active)
        )
        self.VALIDATOR.validate_mixing_state(state, tokens)
        projected = state.hidden[:, :input_length]
        return (
            projected if self.batch_first_flag else projected.transpose(0, 1)
        ), state.loss
