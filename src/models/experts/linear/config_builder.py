from dataclasses import replace
from typing import TYPE_CHECKING

from models.experts.linear._hidden_model_config_factory import (
    HiddenModelConfigFactory,
)
from models.experts.linear._projection_config_factory import (
    ProjectionConfigFactory,
)
from models.experts.linear.experiment_config import ExperimentConfig
from models.experts.linear.runtime_defaults import DEFAULT_RUNTIME
from models.experts.linear.runtime_options import RuntimeOptions

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LinearConfigBuilder:
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

        self.batch_size = runtime.batch_size
        self.learning_rate = runtime.learning_rate
        self.input_dim = runtime.input_dim
        self.hidden_dim = runtime.hidden_dim
        self.output_dim = runtime.output_dim
        self.stack_options = runtime.stack_options
        self.submodule_stack_options = runtime.submodule_stack_options
        self.mixture_options = runtime.mixture_options
        self.expert_stack_options = runtime.expert_stack_options
        self.sampler_options = runtime.sampler_options
        self.router_options = runtime.router_options
        self.router_stack_options = runtime.router_stack_options
        self.layer_controller_options = runtime.layer_controller_options
        self.dynamic_memory_options = runtime.dynamic_memory_options
        self.recurrent_controller_options = runtime.recurrent_controller_options
        self.expert_layer_controller_options = runtime.expert_layer_controller_options
        self.expert_dynamic_memory_options = runtime.expert_dynamic_memory_options
        self.expert_recurrent_controller_options = (
            runtime.expert_recurrent_controller_options
        )
        self._runtime = runtime

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        projection_factory = ProjectionConfigFactory(self.stack_options)
        hidden_settings = replace(
            self.runtime,
            output_dim=self.output_dim,
            stack_options=self.stack_options,
            submodule_stack_options=self.submodule_stack_options,
            mixture_options=self.mixture_options,
            expert_stack_options=self.expert_stack_options,
            sampler_options=self.sampler_options,
            router_options=self.router_options,
            router_stack_options=self.router_stack_options,
            layer_controller_options=self.layer_controller_options,
            dynamic_memory_options=self.dynamic_memory_options,
            recurrent_controller_options=self.recurrent_controller_options,
            expert_layer_controller_options=self.expert_layer_controller_options,
            expert_dynamic_memory_options=self.expert_dynamic_memory_options,
            expert_recurrent_controller_options=self.expert_recurrent_controller_options,
        )
        hidden_factory = HiddenModelConfigFactory(hidden_settings)
        input_model_config = projection_factory.build_input_model_config()
        hidden_model_config = hidden_factory.build()
        output_model_config = projection_factory.build_output_model_config()
        experiment_config = ExperimentConfig(
            input_model_config=input_model_config,
            model_config=self.runtime.configure_stack(hidden_model_config),
            output_model_config=output_model_config,
        )
        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=hidden_settings.stack_options.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=experiment_config,
        )
