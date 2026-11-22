from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, cast

from torch import Tensor

from emperor.experiments.classifier import ClassifierExperiment
from emperor.layers import (
    Layer,
    LayerConfig,
    LayerNormalizationDelegate,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
    NormalizationOptions,
    RecurrentLayerConfig,
)
from emperor.patch import PatchConfig
from emperor.transformer import TransformerEncoderLayerState

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class _MixerStages(Protocol):
    patch_config: Final[PatchConfig | None]
    encoder_config: Final[LayerStackConfig | RecurrentLayerConfig | None]
    output_config: Final[LayerConfig | None]
    encoder_output_normalization: Final[NormalizationOptions]


class MixerImageClassifier(ClassifierExperiment):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        stages = cast("_MixerStages | None", config.experiment_config)
        if stages is None:
            raise ValueError("experiment_config is required to build the model")
        if stages.patch_config is None:
            raise ValueError("experiment_config.patch_config is required")
        if stages.encoder_config is None:
            raise ValueError("experiment_config.encoder_config is required")
        if stages.output_config is None:
            raise ValueError("experiment_config.output_config is required")
        self.experiment_config: _MixerStages = stages
        self.patch = stages.patch_config.build()
        self.transformer = stages.encoder_config.build()
        layer_config = LayerConfig(
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.hidden_dim,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            normalization=self.experiment_config.encoder_output_normalization,
        )
        self.encoder_layer_norm = LayerNormalizationDelegate(layer_config)
        self.output = stages.output_config.build()

    @property
    def mixer(self):
        return self.transformer

    def forward(self, X: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        patch_state = LayerState.run_tensor_module(self.patch, X.to(self.device))
        encoder_state = self.transformer(
            TransformerEncoderLayerState(hidden=patch_state.hidden)
        )
        normalized_tokens = self.encoder_layer_norm.before_model(
            LayerState(hidden=encoder_state.hidden)
        ).hidden
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
        encoder_loss = encoder_state.loss
        if (
            encoder_loss is not None
            and not encoder_loss.requires_grad
            and encoder_loss.item() == 0.0
        ):
            # The encoder returns a detached scalar zero for a penalty-free block.
            encoder_loss = None
        loss = LayerState.sum_losses(patch_state.loss, encoder_loss, output_state.loss)
        if loss is not None:
            return logits, loss
        return logits
