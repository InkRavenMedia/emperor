from typing import TYPE_CHECKING, cast

import torch.nn as nn
from torch import Tensor

from emperor.neuron import NeuronClusterConfig
from models import _attention_settings
from models.vit._classifier import ImageStages, VisionClassifier

from .experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(VisionClassifier):
    def __init__(self, config: "ModelConfig"):
        experiment = config.experiment_config
        if experiment is None:
            raise ValueError("experiment_config is required to build the model")
        super().__init__(config)
        self.experiment_config = config.experiment_config
        if not isinstance(self.experiment_config, ExperimentConfig):
            raise TypeError(
                "Expected a ViT Expert Linear Adaptive Cluster ExperimentConfig."
            )
        _attention_settings.require_stages(
            experiment,
            "patch_config",
            "positional_embedding_config",
            "neuron_cluster_config",
            "output_config",
        )
        self._build_image_stages(cast(ImageStages, self.experiment_config))

    def _build_encoder(self, stages: ImageStages) -> nn.Module:
        cluster = cast(ExperimentConfig, stages).neuron_cluster_config
        return cast(NeuronClusterConfig, cluster).build()

    def forward(self, images: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        return self._classify_image(images)
