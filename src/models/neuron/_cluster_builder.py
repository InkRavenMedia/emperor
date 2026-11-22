import copy
from enum import Enum
from typing import Final, Protocol

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import LayerConfig, LayerStackConfig
from emperor.linears import LinearLayerConfig
from emperor.neuron import (
    AxonsConfig,
    NeuronClusterConfig,
    NeuronConfig,
    NucleusConfig,
    TerminalConfig,
    TerminalConnectionShapeOptions,
    TerminalRangeOptions,
    TerminalRoutingTreeConfig,
    TerminalRoutingTreeDepthOptions,
)
from emperor.sampler import RouterConfig, SamplerConfig
from models._expert_mixture import SamplerSettings
from models._linear_stack import LayerStackSettings
from models._residual_stack import build_residual_config
from models.neuron._hidden_block import HiddenBlockConfig


class ClusterCapacitySettings(Protocol):
    x_axis_total_neurons: Final[int]
    y_axis_total_neurons: Final[int]
    z_axis_total_neurons: Final[int]
    initial_x_axis_total_neurons: Final[int | None]
    initial_y_axis_total_neurons: Final[int | None]
    initial_z_axis_total_neurons: Final[int | None]
    max_steps: Final[int]
    growth_threshold: Final[int | None]
    beam_width: Final[int | None]
    growth_cooldown_steps: Final[int | None]
    max_total_growths: Final[int | None]
    growth_warmup_steps: Final[int | None]
    pruning_threshold: Final[int | None]
    escape_driven_growth_flag: Final[bool]
    mitosis_initialization_flag: Final[bool]


class _RoutingTree(Protocol):
    depth: Final[TerminalRoutingTreeDepthOptions]
    direction_branch_counts: Final[tuple[int, ...]]
    direction_top_k: Final[tuple[int, ...]]


class TerminalSettings(Protocol):
    xy_axis_range: Final[TerminalRangeOptions]
    z_axis_range: Final[TerminalRangeOptions]
    top_k: Final[int]
    routing_tree: Final[_RoutingTree | None]


class ClusterHaltingSettings(Protocol):
    enabled: Final[bool]
    threshold: Final[float]
    dropout: Final[float]
    hidden_state_mode: Final[HaltingHiddenStateModeOptions]
    stack_options: Final[LayerStackSettings]
    output_dim: Final[int]
    halting_option: Final[type[HaltingConfig]]


class NeuronControllerStackConfigFactory:
    def build_config(
        self,
        options: LayerStackSettings,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> LayerStackConfig:
        layer_config = self.__build_layer_config(options)
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
            layer_config=layer_config,
        )

    @staticmethod
    def __build_layer_config(
        options: LayerStackSettings,
    ) -> LayerConfig:
        residual_config = build_residual_config(
            options.residual_connection_option,
            options.residual_model_flag,
            options.residual_stack_options,
            residual_block_size=options.residual_block_size,
            residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
        )
        linear_layer_config = LinearLayerConfig(
            bias_flag=options.bias_flag,
        )
        return LayerConfig(
            activation=options.activation,
            residual_config=residual_config,
            dropout_probability=options.dropout_probability,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=linear_layer_config,
        )


class NeuronControlConfigFactory:
    hidden_block_type: type[HiddenBlockConfig]

    def __init__(
        self,
        *,
        cluster_capacity_options: ClusterCapacitySettings,
        terminal_options: TerminalSettings,
        terminal_router_options: LayerStackSettings,
        terminal_sampler_options: SamplerSettings,
        cluster_halting_options: ClusterHaltingSettings,
    ) -> None:
        self.cluster_capacity_options = cluster_capacity_options
        self.terminal_options = terminal_options
        self.terminal_router_options = terminal_router_options
        self.terminal_sampler_options = terminal_sampler_options
        self.cluster_halting_options = cluster_halting_options
        self.controller_stack_config_factory = NeuronControllerStackConfigFactory()

    def build(
        self,
        hidden_model_config: ConfigBase,
        hidden_dim: int,
    ) -> NeuronClusterConfig:
        hidden_model_copy = copy.deepcopy(hidden_model_config)
        hidden_block_config = self.hidden_block_type(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            model_config=hidden_model_copy,
        )
        terminal_sampler_config = self.__build_terminal_sampler_config(hidden_dim)
        neuron_config = self.__build_neuron_config(
            hidden_dim,
            hidden_block_config,
            terminal_sampler_config,
        )
        neuron_cluster_config = self.__build_neuron_cluster_config(
            hidden_dim,
            neuron_config,
        )
        return neuron_cluster_config

    def __build_neuron_config(
        self,
        hidden_dim: int,
        hidden_block_config: HiddenBlockConfig,
        terminal_sampler_config: SamplerConfig,
    ) -> NeuronConfig:
        terminal_options = self.terminal_options
        nucleus_config = NucleusConfig(model_config=hidden_block_config)
        axons_config = AxonsConfig(memory_config=None)
        routing_tree_config = self.__build_terminal_routing_tree_config()
        terminal_config = TerminalConfig(
            input_dim=hidden_dim,
            connection_shape=TerminalConnectionShapeOptions.BOX,
            xy_axis_range=terminal_options.xy_axis_range,
            z_axis_range=terminal_options.z_axis_range,
            sampler_config=terminal_sampler_config,
            routing_tree_config=routing_tree_config,
        )
        return NeuronConfig(
            nucleus_config=nucleus_config,
            axons_config=axons_config,
            terminal_config=terminal_config,
        )

    def __build_terminal_routing_tree_config(
        self,
    ) -> TerminalRoutingTreeConfig | None:
        routing_tree_options: _RoutingTree | None = self.terminal_options.routing_tree
        if routing_tree_options is None:
            return None
        return TerminalRoutingTreeConfig(
            depth=routing_tree_options.depth,
            direction_branch_counts=routing_tree_options.direction_branch_counts,
            direction_top_k=routing_tree_options.direction_top_k,
            direction_sampler_config=None,
        )

    def __build_neuron_cluster_config(
        self,
        hidden_dim: int,
        neuron_config: NeuronConfig,
    ) -> NeuronClusterConfig:
        capacity_options = self.cluster_capacity_options
        halting_config = self.__build_cluster_halting_config(hidden_dim)
        return NeuronClusterConfig(
            x_axis_total_neurons=capacity_options.x_axis_total_neurons,
            y_axis_total_neurons=capacity_options.y_axis_total_neurons,
            z_axis_total_neurons=capacity_options.z_axis_total_neurons,
            initial_x_axis_total_neurons=(
                capacity_options.initial_x_axis_total_neurons
            ),
            initial_y_axis_total_neurons=(
                capacity_options.initial_y_axis_total_neurons
            ),
            initial_z_axis_total_neurons=(
                capacity_options.initial_z_axis_total_neurons
            ),
            entry_sampler_config=None,
            max_steps=capacity_options.max_steps,
            beam_width=capacity_options.beam_width,
            growth_threshold=capacity_options.growth_threshold,
            growth_cooldown_steps=capacity_options.growth_cooldown_steps,
            max_total_growths=capacity_options.max_total_growths,
            growth_warmup_steps=capacity_options.growth_warmup_steps,
            pruning_threshold=capacity_options.pruning_threshold,
            escape_driven_growth_flag=(capacity_options.escape_driven_growth_flag),
            mitosis_initialization_flag=(capacity_options.mitosis_initialization_flag),
            halting_config=halting_config,
            neuron_config=neuron_config,
        )

    def __build_terminal_sampler_config(self, hidden_dim: int) -> SamplerConfig:
        terminal_options = self.terminal_options
        sampler_options = self.terminal_sampler_options
        num_experts = self.__terminal_num_experts()
        top_k = min(max(1, terminal_options.top_k), num_experts)
        router_config = self.__build_router_config(hidden_dim, num_experts)
        return SamplerConfig(
            top_k=top_k,
            threshold=sampler_options.threshold,
            filter_above_threshold=sampler_options.filter_above_threshold,
            num_topk_samples=min(sampler_options.num_topk_samples, top_k),
            normalize_probabilities_flag=(sampler_options.normalize_probabilities_flag),
            noisy_topk_flag=sampler_options.noisy_topk_flag,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=(
                sampler_options.coefficient_of_variation_loss_weight
            ),
            switch_loss_weight=sampler_options.switch_loss_weight,
            zero_centred_loss_weight=sampler_options.zero_centred_loss_weight,
            mutual_information_loss_weight=(
                sampler_options.mutual_information_loss_weight
            ),
            router_config=router_config,
        )

    def __build_router_config(
        self,
        hidden_dim: int,
        num_experts: int,
    ) -> RouterConfig:
        router_options = self.terminal_router_options
        router_hidden_dim = router_options.hidden_dim or max(
            hidden_dim,
            num_experts,
        )
        model_config = self.controller_stack_config_factory.build_config(
            router_options,
            input_dim=hidden_dim,
            hidden_dim=router_hidden_dim,
            output_dim=num_experts,
        )
        return RouterConfig(
            input_dim=hidden_dim,
            num_experts=num_experts,
            noisy_topk_flag=self.terminal_sampler_options.noisy_topk_flag,
            model_config=model_config,
        )

    def __build_cluster_halting_config(
        self,
        hidden_dim: int,
    ) -> HaltingConfig | None:
        halting_options = self.cluster_halting_options
        if not halting_options.enabled:
            return None
        stack_options = halting_options.stack_options
        halting_gate_config = self.controller_stack_config_factory.build_config(
            stack_options,
            input_dim=hidden_dim,
            hidden_dim=stack_options.hidden_dim,
            output_dim=halting_options.output_dim,
        )
        return halting_options.halting_option(
            input_dim=hidden_dim,
            threshold=halting_options.threshold,
            min_steps=1,
            ponder_cost_weight=1.0,
            dropout_probability=halting_options.dropout,
            hidden_state_mode=halting_options.hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )

    def __terminal_num_experts(self) -> int:
        terminal_options = self.terminal_options
        xy_range = self.__enum_or_int_value(terminal_options.xy_axis_range)
        z_range = self.__enum_or_int_value(terminal_options.z_axis_range)
        return (xy_range * 2 + 1) ** 2 * (z_range * 2 + 1)

    @staticmethod
    def __enum_or_int_value(value: Enum | int) -> int:
        if isinstance(value, Enum):
            return int(value.value)
        return int(getattr(value, "value", value))
