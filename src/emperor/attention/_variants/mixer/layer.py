"""Coordinate Mixer masking, source memory, and token-axis mixing."""

from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention._variants.mixer.masking import MixerMask
from emperor.attention._variants.mixer.memory import MixerSourceMemory
from emperor.attention._variants.mixer.processing import MixerProcessor
from emperor.attention._variants.mixer.reshaping import MixerReshaper
from emperor.attention._variants.mixer.validation import MixerAttentionValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.attention._variants.mixer.config import MixerAttentionConfig
    from emperor.config import ModelConfig


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
        self.embedding_dim = self.cfg.embedding_dim
        self.sequence_length = self.cfg.sequence_length
        self.batch_first_flag = self.cfg.batch_first_flag
        self.VALIDATOR.validate(self)

        self.reshaper = MixerReshaper(self.cfg)
        self.masks = MixerMask(self.reshaper)
        self.processor = MixerProcessor(self.cfg, self.VALIDATOR)
        self.memory_handler = MixerSourceMemory(
            self.cfg,
            self.reshaper,
            self.processor.memory_sequence_length,
        )
        self.source_memory = self.memory_handler.build_source_memory()

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
        source_tokens, excluded_token_mask = self.masks.prepare_inputs(
            q, k_padding_mask
        )
        token_axis_vectors = self.reshaper.prepare_token_axis_vectors(source_tokens)
        token_axis_vectors, memory_loss = self.memory_handler.append_sources(
            self.source_memory, token_axis_vectors, source_tokens, excluded_token_mask
        )
        mixed_token_axis_vectors, mixing_loss = self.processor.compute_mixing(
            token_axis_vectors
        )
        mixed_tokens = self.reshaper.restore_output_layout(
            mixed_token_axis_vectors, input_sequence_length
        )
        mixed_tokens = self.masks.apply_to_tokens(mixed_tokens, excluded_token_mask)
        auxiliary_loss = self.__combine_auxiliary_loss(mixing_loss, memory_loss)
        return mixed_tokens, None, auxiliary_loss

    @staticmethod
    def __combine_auxiliary_loss(
        primary_loss: Tensor | None, additional_loss: Tensor | None
    ) -> Tensor | None:
        if additional_loss is None:
            return primary_loss
        if primary_loss is None:
            return additional_loss
        return primary_loss + additional_loss
