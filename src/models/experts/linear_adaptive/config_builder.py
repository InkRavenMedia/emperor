from __future__ import annotations

from emperor.config import ModelConfig
from models.experts.linear_adaptive._control_config_factory import (
    ControlConfigFactory,
)
from models.experts.linear_adaptive._projection_config_factory import (
    BoundaryModelConfigFactory,
)
from models.experts.linear_adaptive.experiment_config import ExperimentConfig
from models.experts.linear_adaptive.runtime_defaults import DEFAULT_RUNTIME
from models.experts.linear_adaptive.runtime_options import RuntimeOptions


class LinearAdaptiveConfigBuilder:
    """Build adaptive-expert configs from one immutable runtime value."""

    def __init__(
        self,
        *,
        runtime: RuntimeOptions = DEFAULT_RUNTIME,
    ) -> None:
        self.runtime = runtime

    @property
    def runtime(self) -> RuntimeOptions:
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: RuntimeOptions) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")
        self._runtime = runtime

    def build(self) -> ModelConfig:
        runtime = self.runtime
        input_factory = BoundaryModelConfigFactory(runtime)
        input_model_config = input_factory.build_input_model_config()
        hidden_factory = ControlConfigFactory(runtime)
        hidden_model_config = hidden_factory.build()
        output_factory = BoundaryModelConfigFactory(runtime)
        output_model_config = output_factory.build_output_model_config()
        experiment_config = ExperimentConfig(
            input_model_config=input_model_config,
            model_config=self.runtime.configure_stack(hidden_model_config),
            output_model_config=output_model_config,
        )
        return ModelConfig(
            learning_rate=runtime.learning_rate,
            batch_size=runtime.batch_size,
            input_dim=runtime.input_dim,
            hidden_dim=runtime.stack_options.hidden_dim,
            output_dim=runtime.output_dim,
            experiment_config=experiment_config,
        )


__all__ = ["LinearAdaptiveConfigBuilder"]
