from typing import ClassVar, Final, Protocol, cast

import torch
from torch import Tensor

from emperor.config import ConfigBase, ModelConfig
from emperor.experiments.classifier import ClassifierExperiment
from emperor.layers import Layer
from emperor.neuron import NeuronClusterConfig
from models._classifier_stages import build_stage


class _NeuronStages(Protocol):
    input_model_config: Final[ConfigBase | None]
    neuron_cluster_config: Final[NeuronClusterConfig | None]
    output_model_config: Final[ConfigBase | None]


class NeuronClassifier(ClassifierExperiment):
    experiment_type: ClassVar[type[ConfigBase]]
    invalid_experiment_message: ClassVar[str]

    def __init__(self, config: "ModelConfig") -> None:
        stages = config.experiment_config
        if stages is None:
            raise ValueError("experiment_config is required to build the model")
        if not isinstance(stages, self.experiment_type):
            raise TypeError(self.invalid_experiment_message)
        super().__init__(config)
        stages = cast(_NeuronStages, stages)
        if stages.input_model_config is None:
            raise ValueError("experiment_config.input_model_config is required")
        if stages.neuron_cluster_config is None:
            raise ValueError("experiment_config.neuron_cluster_config is required")
        if stages.output_model_config is None:
            raise ValueError("experiment_config.output_model_config is required")
        self.experiment_config = stages
        self.input_model = self._build_input_model()
        self.neuron_cluster = stages.neuron_cluster_config.build()
        self.output_model = self._build_output_model()

    def _build_input_model(self):
        return build_stage(
            cast(ConfigBase, self.experiment_config.input_model_config),
            input_dim=self.cfg.input_dim,
            output_dim=self.cfg.hidden_dim,
        )

    def _build_output_model(self):
        return build_stage(
            cast(ConfigBase, self.experiment_config.output_model_config),
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.output_dim,
        )

    def forward(
        self,
        X: Tensor,
    ) -> tuple[Tensor, Tensor]:
        inputs = torch.flatten(X.to(self.device), start_dim=1)
        input_state = Layer.run_model_from_hidden(self.input_model, inputs)
        hidden = input_state.hidden
        hidden, auxiliary_loss = self.neuron_cluster(hidden)
        output_state = Layer.run_model_from_hidden(self.output_model, hidden)
        logits = output_state.hidden
        for boundary in (input_state, output_state):
            if boundary.loss is not None:
                auxiliary_loss = auxiliary_loss + boundary.loss
        return logits, auxiliary_loss
