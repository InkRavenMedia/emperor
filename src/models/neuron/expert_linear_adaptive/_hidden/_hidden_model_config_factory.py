from __future__ import annotations

from emperor.experts import MixtureOfExpertsModelConfig
from emperor.layers import LayerConfig, RecurrentLayerConfig
from models.neuron.expert_linear_adaptive._hidden._control_config_factory import (
    ControlConfigFactory,
)
from models.neuron.expert_linear_adaptive._hidden._projection_config_factory import (
    BoundaryModelConfigFactory,
)
from models.neuron.expert_linear_adaptive._hidden.runtime_options import RuntimeOptions


class HiddenModelConfigFactory:
    """Build package-local boundaries and hidden expert-adaptive configuration."""

    def __init__(self, runtime: RuntimeOptions) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")
        self.runtime = runtime
        self.hidden_dim = runtime.stack_options.hidden_dim

    def build_input_model_config(self) -> LayerConfig:
        boundary_model_factory = BoundaryModelConfigFactory(self.runtime)
        input_model_config = boundary_model_factory.build_input_model_config()
        return input_model_config

    def build_hidden_model_config(
        self,
    ) -> MixtureOfExpertsModelConfig | RecurrentLayerConfig:
        control_factory = ControlConfigFactory(self.runtime)
        hidden_model_config = control_factory.build()
        return hidden_model_config

    def build_output_model_config(self) -> LayerConfig:
        boundary_model_factory = BoundaryModelConfigFactory(self.runtime)
        output_model_config = boundary_model_factory.build_output_model_config()
        return output_model_config


__all__ = ["HiddenModelConfigFactory"]
