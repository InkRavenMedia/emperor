from typing import Final, Protocol, cast

import torch.nn as nn
from torch import Tensor

from emperor.config import ConfigBase, ModelConfig
from emperor.experiments.classifier import ClassifierExperiment
from emperor.layers import (
    Layer,
    LayerConfig,
    LayerNormalizationDelegate,
    LayerNormPositionOptions,
    LayerState,
    NormalizationOptions,
)
from emperor.nn import Module
from emperor.transformer import TransformerEncoderLayerState
from models import _attention_settings


class ImageStages(Protocol):
    patch_config: Final[ConfigBase]
    positional_embedding_config: Final[ConfigBase]
    output_config: Final[ConfigBase]
    encoder_output_normalization: Final[NormalizationOptions]


class _EncoderStages(ImageStages, Protocol):
    encoder_config: Final[ConfigBase]


class VisionClassifier(ClassifierExperiment):
    patch: nn.Module
    positional_embedding: nn.Module
    transformer: nn.Module
    output: nn.Module
    encoder_layer_norm: LayerNormalizationDelegate

    def _build_image_stages(self, stages: ImageStages) -> None:
        self.patch = stages.patch_config.build()
        self.positional_embedding = stages.positional_embedding_config.build()
        self.transformer = self._build_encoder(stages)
        self.encoder_layer_norm = LayerNormalizationDelegate(
            LayerConfig(
                input_dim=self.cfg.hidden_dim,
                output_dim=self.cfg.hidden_dim,
                layer_norm_position=LayerNormPositionOptions.BEFORE,
                normalization=stages.encoder_output_normalization,
            )
        )
        self.output = stages.output_config.build()

    def _build_encoder(self, stages: ImageStages) -> nn.Module:
        return cast(_EncoderStages, stages).encoder_config.build()

    def _classify_tokens(
        self, encoded: TransformerEncoderLayerState
    ) -> tuple[Tensor, Tensor | None]:
        hidden = self.encoder_layer_norm.before_model(
            LayerState(hidden=encoded.hidden)
        ).hidden
        logits = self.output(hidden[:, 0, :])
        return logits, LayerState.module_loss(self.output)

    def _classify_image(self, images: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        patches = LayerState.run_tensor_module(self.patch, images.to(self.device))
        tokens = self.positional_embedding(patches.hidden)
        position_loss = LayerState.module_loss(self.positional_embedding)
        encoded = self.transformer(TransformerEncoderLayerState(hidden=tokens))
        logits, head_loss = self._classify_tokens(encoded)
        encoder_loss = encoded.loss
        if (
            encoder_loss is not None
            and not encoder_loss.requires_grad
            and encoder_loss.detach().eq(0).item()
        ):
            encoder_loss = None
        loss = LayerState.sum_losses(
            patches.loss, position_loss, encoder_loss, head_loss
        )
        if loss is None:
            return logits
        return logits, loss


class EncoderClassifier(VisionClassifier):
    def __init__(self, config: ModelConfig):
        stages = self._resolve_experiment(config)
        super().__init__(config)
        self.experiment_config = stages
        self._build_image_stages(stages)

    @classmethod
    def _resolve_experiment(cls, config: ModelConfig) -> ImageStages:
        _attention_settings.require_stages(
            config.experiment_config,
            "patch_config",
            "positional_embedding_config",
            "encoder_config",
            "output_config",
        )
        return cast(ImageStages, config.experiment_config)


class LayerHeadClassifier(EncoderClassifier):
    def _classify_tokens(
        self, encoded: TransformerEncoderLayerState
    ) -> tuple[Tensor, Tensor | None]:
        self.encoder_layer_norm.before_model(encoded)
        output = Layer.run_model_from_hidden(
            cast(Module, self.output), encoded.hidden[:, 0, :]
        )
        return output.hidden, output.loss

    def forward(self, X: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        return self._classify_image(X)


class ExpertHeadClassifier(EncoderClassifier):
    experiment_type: type[ConfigBase]
    invalid_experiment_message: str

    @classmethod
    def _resolve_experiment(cls, config: ModelConfig) -> ImageStages:
        if config.experiment_config is None:
            raise ValueError("experiment_config is required to build the model")
        if not isinstance(config.experiment_config, cls.experiment_type):
            raise TypeError(cls.invalid_experiment_message)
        return super()._resolve_experiment(config)

    def forward(self, images: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        return self._classify_image(images)
