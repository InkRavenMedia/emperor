from __future__ import annotations

from emperor.config import ModelConfig
from models.neuron._cluster_settings import NeuronClusterSettings
from models.neuron.linear_adaptive._hidden._hidden_model_config_factory import (
    HiddenModelConfigFactory,
)
from models.neuron.linear_adaptive._hidden._projection_config_factory import (
    ProjectionConfigFactory,
)
from models.neuron.linear_adaptive._hidden.runtime_options import (
    RuntimeOptions as HiddenRuntimeOptions,
)
from models.neuron.linear_adaptive._neuron_control_config_factory import (
    NeuronControlConfigFactory,
)
from models.neuron.linear_adaptive.experiment_config import ExperimentConfig
from models.neuron.linear_adaptive.runtime_options import (
    ClusterRouteHaltingOptions,
    NeuronClusterCapacityOptions,
    NeuronSubmoduleStackOptions,
    NeuronTerminalOptions,
    NeuronTerminalSamplerOptions,
)


class NeuronConfigBuilder(NeuronClusterSettings):
    def __init__(
        self,
        *,
        hidden_runtime: HiddenRuntimeOptions,
        cluster_capacity_options: NeuronClusterCapacityOptions,
        terminal_options: NeuronTerminalOptions,
        terminal_router_options: NeuronSubmoduleStackOptions,
        terminal_sampler_options: NeuronTerminalSamplerOptions,
        cluster_halting_options: ClusterRouteHaltingOptions,
    ) -> None:
        self.hidden_runtime = hidden_runtime
        self.cluster_capacity_options = cluster_capacity_options
        self.terminal_options = terminal_options
        self.terminal_router_options = terminal_router_options
        self.terminal_sampler_options = terminal_sampler_options
        self.cluster_halting_options = cluster_halting_options
        self._expose_flat_settings()

    def build(self) -> ModelConfig:
        projection_factory = ProjectionConfigFactory(self.hidden_runtime)
        hidden_factory = HiddenModelConfigFactory(self.hidden_runtime)
        neuron_control_factory = NeuronControlConfigFactory(
            cluster_capacity_options=self.cluster_capacity_options,
            terminal_options=self.terminal_options,
            terminal_router_options=self.terminal_router_options,
            terminal_sampler_options=self.terminal_sampler_options,
            cluster_halting_options=self.cluster_halting_options,
        )
        hidden_model_config = hidden_factory.build_hidden_model_config()
        neuron_cluster_config = neuron_control_factory.build(
            hidden_model_config,
            self.hidden_runtime.hidden_dim,
        )

        input_model_config = projection_factory.build_input_model_config()
        output_model_config = projection_factory.build_output_model_config()
        experiment_config = ExperimentConfig(
            input_model_config=input_model_config,
            neuron_cluster_config=neuron_cluster_config,
            output_model_config=output_model_config,
        )
        return ModelConfig(
            learning_rate=self.hidden_runtime.learning_rate,
            batch_size=self.hidden_runtime.batch_size,
            input_dim=self.hidden_runtime.input_dim,
            hidden_dim=self.hidden_runtime.hidden_dim,
            output_dim=self.hidden_runtime.output_dim,
            experiment_config=experiment_config,
        )
