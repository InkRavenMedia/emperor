from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from emperor.experiments.classifier import ClassifierExperiment
from emperor.layers import (
    Layer,
    LayerConfig,
    LayerNormPositionOptions,
)
from emperor.transformer import TransformerEncoderLayerState

from .experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.experiment_config: ExperimentConfig = config.experiment_config
        self.patch = self.experiment_config.patch_config.build()
        self.transformer = self.experiment_config.encoder_config.build()
        self.encoder_layer_norm = LayerConfig(
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.hidden_dim,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            normalization=self.experiment_config.encoder_output_normalization,
        ).build_normalization()
        self.output = self.experiment_config.output_config.build()

    @property
    def mixer(self):
        return self.transformer

    def forward(self, X: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        patch_tokens = self.patch(X.to(self.device))
        encoder_state = self.transformer(
            TransformerEncoderLayerState(hidden=patch_tokens)
        )
        normalized_tokens = self.encoder_layer_norm(encoder_state.hidden)
        if encoder_state.key_padding_mask is None:
            pooled_hidden = normalized_tokens.mean(dim=1)
        else:
            valid_tokens = ~encoder_state.key_padding_mask
            pooled_hidden = normalized_tokens.masked_fill(
                ~valid_tokens.unsqueeze(-1), 0
            ).sum(dim=1)
            pooled_hidden = pooled_hidden / valid_tokens.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
        output_state = Layer.run_model_from_hidden(self.output, pooled_hidden)
        logits = output_state.hidden
        if encoder_state.loss is not None and encoder_state.loss.item() != 0.0:
            return logits, encoder_state.loss
        return logits


__all__ = ["Model"]
