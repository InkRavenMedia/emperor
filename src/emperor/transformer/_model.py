from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.layers import LayerConfig, LayerNormPositionOptions, NormalizationOptions
from emperor.nn import Module
from emperor.transformer._state import (
    TransformerDecoderLayerState,
    TransformerEncoderLayerState,
)
from emperor.transformer._validation import TransformerValidator

if TYPE_CHECKING:
    from emperor.config import ConfigBase
    from emperor.layers import LayerState
    from emperor.transformer._config import TransformerConfig


class Transformer(Module):
    VALIDATOR = TransformerValidator

    def __init__(
        self,
        cfg: "TransformerConfig",
        overrides: "TransformerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "transformer_config", cfg)
        self.cfg: TransformerConfig = self._override_config(config, overrides)
        self.VALIDATOR.validate_transformer(self)

        self.encoder_model = self.__build_stack_if_configured(
            self.cfg.encoder_stack_config
        )
        self.decoder_model = self.__build_stack_if_configured(
            self.cfg.decoder_stack_config
        )
        self.encoder_layer_norm = self.__build_final_norm(self.cfg.encoder_stack_config)
        self.decoder_layer_norm = self.__build_final_norm(self.cfg.decoder_stack_config)

    @staticmethod
    def __build_stack_if_configured(stack_config: "ConfigBase | None"):
        if stack_config is None:
            return None
        return stack_config.build()

    def __build_final_norm(self, stack_config: "ConfigBase | None"):
        if stack_config is None:
            return None
        return LayerConfig(
            input_dim=stack_config.output_dim,
            output_dim=stack_config.output_dim,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            normalization=self.cfg.final_normalization
            or NormalizationOptions.LAYER_NORM,
        ).build_normalization()

    def forward(
        self,
        source_token_embeddings: Tensor | None = None,
        target_token_embeddings: Tensor | None = None,
        source_attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        source_key_padding_mask: Tensor | None = None,
        target_key_padding_mask: Tensor | None = None,
        encoder_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self.VALIDATOR.validate_transformer_forward_inputs(
            self, source_token_embeddings, target_token_embeddings
        )
        encoder_state = self.__run_encoder_if_present(
            source_token_embeddings=source_token_embeddings,
            source_key_padding_mask=source_key_padding_mask,
            attention_mask=source_attention_mask,
        )
        encoder_output = None if encoder_state is None else encoder_state.hidden
        if (
            self.decoder_model is not None
            and isinstance(encoder_state, TransformerEncoderLayerState)
            and encoder_state.sequence_expansions
        ):
            encoder_key_padding_mask = self.__expanded_encoder_padding_mask(
                encoder_state, encoder_key_padding_mask
            )
        decoder_state = self.__run_decoder_if_present(
            target_token_embeddings=target_token_embeddings,
            encoder_output=encoder_output,
            target_key_padding_mask=target_key_padding_mask,
            encoder_key_padding_mask=encoder_key_padding_mask,
            attention_mask=target_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        final_state = decoder_state if decoder_state is not None else encoder_state
        total_loss = self.__sum_present_losses(encoder_state, decoder_state)
        return final_state.hidden, total_loss

    def __run_encoder_if_present(
        self,
        source_token_embeddings: Tensor | None,
        source_key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
    ) -> "LayerState | None":
        if self.encoder_model is None:
            return None
        state = TransformerEncoderLayerState(
            hidden=source_token_embeddings,
            key_padding_mask=source_key_padding_mask,
            attention_mask=attention_mask,
        )
        state = self.encoder_model(state)
        state.hidden = self.encoder_layer_norm(state.hidden)
        return state

    def __expanded_encoder_padding_mask(
        self, state: TransformerEncoderLayerState, padding: Tensor | None
    ) -> Tensor:
        if padding is None:
            return state.key_padding_mask
        original_padding = state.sequence_expansions[0].input_padding_mask
        original_length = original_padding.size(1)
        if (
            padding.ndim != 2
            or padding.size(0) != state.hidden.size(0)
            or padding.size(1) not in (original_length, state.hidden.size(1))
        ):
            raise ValueError(
                "encoder_key_padding_mask must match the original or expanded encoder sequence."
            )
        if padding.dtype != torch.bool and not padding.is_floating_point():
            raise ValueError(
                "encoder_key_padding_mask must be Boolean or floating point."
            )
        if padding.size(1) == original_length:
            padding = self.__compact_encoder_padding_mask(
                padding, original_padding, state.hidden.size(1)
            )
        return padding.masked_fill(
            state.key_padding_mask.to(padding.device),
            True if padding.dtype == torch.bool else -torch.inf,
        )

    def __compact_encoder_padding_mask(
        self, padding: Tensor, original_padding: Tensor, output_length: int
    ) -> Tensor:
        original_valid = ~original_padding.to(padding.device)
        compacted_positions = original_valid.cumsum(1) - 1
        batch_indices, token_indices = original_valid.nonzero(as_tuple=True)
        compacted_padding = padding.new_zeros(padding.size(0), output_length)
        return compacted_padding.index_put(
            (batch_indices, compacted_positions[batch_indices, token_indices]),
            padding[batch_indices, token_indices],
        )

    def __run_decoder_if_present(
        self,
        target_token_embeddings: Tensor | None,
        encoder_output: Tensor | None,
        target_key_padding_mask: Tensor | None,
        encoder_key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
        encoder_attention_mask: Tensor | None,
    ) -> "LayerState | None":
        if self.decoder_model is None:
            return None
        state = TransformerDecoderLayerState(
            hidden=target_token_embeddings,
            target_key_padding_mask=target_key_padding_mask,
            target_attention_mask=attention_mask,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_key_padding_mask,
            cross_attention_mask=encoder_attention_mask,
        )
        state = self.decoder_model(state)
        state.hidden = self.decoder_layer_norm(state.hidden)
        return state

    @staticmethod
    def __sum_present_losses(
        encoder_state: "LayerState | None",
        decoder_state: "LayerState | None",
    ) -> Tensor:
        states = [
            state for state in (encoder_state, decoder_state) if state is not None
        ]
        losses = [state.loss for state in states if state.loss is not None]
        if losses:
            return sum(losses[1:], losses[0])
        return states[-1].hidden.new_zeros(())
