from typing import Final, Protocol, cast

import torch
from torch import Tensor

from emperor.config import ConfigBase, ModelConfig
from emperor.experiments.classifier import ClassifierExperiment
from emperor.layers import Layer, LayerState
from emperor.nn import Module


class ClassifierStages(Protocol):
    input_model_config: Final[ConfigBase | None]
    model_config: Final[ConfigBase | None]
    output_model_config: Final[ConfigBase | None]


class _DimensionOverrides(Protocol):
    def __call__(
        self, *, input_dim: int | None, output_dim: int | None
    ) -> ConfigBase: ...


def build_stage(
    stage: ConfigBase, *, input_dim: int | None, output_dim: int | None
) -> Module:
    overrides_type = cast(_DimensionOverrides, type(stage))
    overrides = overrides_type(input_dim=input_dim, output_dim=output_dim)
    return stage.build(overrides=overrides)


class ThreeStageClassifier(ClassifierExperiment):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        stages = cast(ClassifierStages | None, config.experiment_config)
        if stages is None:
            raise ValueError("experiment_config is required to build the model")
        if stages.input_model_config is None:
            raise ValueError("experiment_config.input_model_config is required")
        if stages.model_config is None:
            raise ValueError("experiment_config.model_config is required")
        if stages.output_model_config is None:
            raise ValueError("experiment_config.output_model_config is required")

        self.experiment_config = stages
        self.input_model = build_stage(
            stages.input_model_config,
            input_dim=config.input_dim,
            output_dim=config.hidden_dim,
        )
        self.main_model = build_stage(
            stages.model_config,
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
        )
        self.output_model = build_stage(
            stages.output_model_config,
            input_dim=config.hidden_dim,
            output_dim=config.output_dim,
        )

    def forward(self, X: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        inputs = torch.flatten(X.to(self.device), start_dim=1)
        input_state = Layer.run_model_from_hidden(self.input_model, inputs)
        main_state = Layer.run_model_from_hidden(self.main_model, input_state.hidden)
        output_state = Layer.run_model_from_hidden(self.output_model, main_state.hidden)
        loss = LayerState.sum_losses(
            input_state.loss, main_state.loss, output_state.loss
        )
        if loss is None:
            return output_state.hidden
        return output_state.hidden, loss
