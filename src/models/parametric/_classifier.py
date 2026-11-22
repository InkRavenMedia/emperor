from typing import ClassVar, cast

import torch
from torch import Tensor

from emperor.config import ConfigBase, ModelConfig
from emperor.experiments.classifier import ClassifierExperiment
from emperor.layers import Layer, LayerState
from emperor.parametric import ParametricLayerState
from models._classifier_stages import ClassifierStages, build_stage


class ParametricClassifier(ClassifierExperiment):
    experiment_type: ClassVar[type[ConfigBase]]
    invalid_experiment_message: ClassVar[str]

    def __init__(
        self,
        config: "ModelConfig",
    ):
        super().__init__(config)
        stages = config.experiment_config
        if stages is None:
            raise ValueError("experiment_config is required to build the model")
        if not isinstance(stages, self.experiment_type):
            raise TypeError(self.invalid_experiment_message)

        stages = cast(ClassifierStages, stages)
        if stages.input_model_config is None:
            raise ValueError("experiment_config.input_model_config is required")
        if stages.model_config is None:
            raise ValueError("experiment_config.model_config is required")
        if stages.output_model_config is None:
            raise ValueError("experiment_config.output_model_config is required")
        self.experiment_config = stages
        self.input_config = stages.input_model_config
        self.model_config = stages.model_config
        self.output_config = stages.output_model_config

        self.input_model = build_stage(
            self.input_config,
            input_dim=self.cfg.input_dim,
            output_dim=self.cfg.hidden_dim,
        )
        self.model = build_stage(
            self.model_config,
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.hidden_dim,
        )
        self.output_model = build_stage(
            self.output_config,
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.output_dim,
        )

    def forward(
        self,
        X: Tensor,
    ) -> tuple[Tensor, Tensor]:
        inputs = torch.flatten(X.to(self.device), start_dim=1)
        input_state = Layer.run_model_from_hidden(self.input_model, inputs)

        parametric_state = ParametricLayerState(hidden=input_state.hidden)
        parametric_state = self.model(parametric_state)

        output_state = Layer.run_model_from_hidden(
            self.output_model,
            parametric_state.hidden,
        )
        logits = output_state.hidden
        loss = LayerState.sum_losses(
            input_state.loss, parametric_state.loss, output_state.loss
        )
        return logits, logits.new_zeros(()) if loss is None else loss
