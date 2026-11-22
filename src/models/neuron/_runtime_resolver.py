from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import Field, dataclass, fields, replace
from types import ModuleType
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast

from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.neuron import TerminalRangeOptions, TerminalRoutingTreeDepthOptions


@dataclass(frozen=True, slots=True)
class NeuronDefaults:
    cluster_beam_width: int | None
    cluster_escape_driven_growth_flag: bool
    cluster_growth_cooldown_steps: int | None
    cluster_growth_threshold: int | None
    cluster_growth_warmup_steps: int | None
    cluster_halting_dropout: float
    cluster_halting_flag: bool
    cluster_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    cluster_halting_option: type[HaltingConfig]
    cluster_halting_output_dim: int
    cluster_halting_stack_activation: ActivationOptions
    cluster_halting_stack_apply_output_postprocessing_flag: bool
    cluster_halting_stack_bias_flag: bool
    cluster_halting_stack_dropout_probability: float
    cluster_halting_stack_hidden_dim: int
    cluster_halting_stack_last_layer_bias_option: LastLayerBiasOptions
    cluster_halting_stack_layer_norm_position: LayerNormPositionOptions
    cluster_halting_stack_normalization: NormalizationOptions
    cluster_halting_stack_num_layers: int
    cluster_halting_stack_residual_connection_option: type[ResidualConfig] | None
    cluster_halting_stack_residual_model_flag: bool
    cluster_halting_stack_residual_block_size: int | None
    cluster_halting_stack_residual_rms_norm_epsilon: float | None
    cluster_halting_threshold: float
    cluster_initial_x_axis_total_neurons: int
    cluster_initial_y_axis_total_neurons: int
    cluster_initial_z_axis_total_neurons: int
    cluster_max_steps: int
    cluster_max_total_growths: int | None
    cluster_mitosis_initialization_flag: bool
    cluster_pruning_threshold: int | None
    cluster_terminal_router_activation: ActivationOptions
    cluster_terminal_router_apply_output_postprocessing_flag: bool
    cluster_terminal_router_bias_flag: bool
    cluster_terminal_router_dropout_probability: float
    cluster_terminal_router_hidden_dim: int
    cluster_terminal_router_last_layer_bias_option: LastLayerBiasOptions
    cluster_terminal_router_layer_norm_position: LayerNormPositionOptions
    cluster_terminal_router_normalization: NormalizationOptions
    cluster_terminal_router_num_layers: int
    cluster_terminal_router_residual_connection_option: type[ResidualConfig] | None
    cluster_terminal_router_residual_model_flag: bool
    cluster_terminal_router_residual_block_size: int | None
    cluster_terminal_router_residual_rms_norm_epsilon: float | None
    cluster_terminal_sampler_coefficient_of_variation_loss_weight: float
    cluster_terminal_sampler_filter_above_threshold: bool
    cluster_terminal_sampler_mutual_information_loss_weight: float
    cluster_terminal_sampler_noisy_topk_flag: bool
    cluster_terminal_sampler_normalize_probabilities_flag: bool
    cluster_terminal_sampler_num_topk_samples: int
    cluster_terminal_sampler_switch_loss_weight: float
    cluster_terminal_sampler_threshold: float
    cluster_terminal_sampler_zero_centred_loss_weight: float
    cluster_terminal_top_k: int
    cluster_terminal_routing_tree_depth: TerminalRoutingTreeDepthOptions | None
    cluster_terminal_routing_tree_level_1_branch_count: int | None
    cluster_terminal_routing_tree_level_1_top_k: int | None
    cluster_terminal_routing_tree_level_2_branch_count: int | None
    cluster_terminal_routing_tree_level_2_top_k: int | None
    cluster_terminal_xy_axis_range: TerminalRangeOptions
    cluster_terminal_z_axis_range: TerminalRangeOptions
    cluster_x_axis_total_neurons: int
    cluster_y_axis_total_neurons: int
    cluster_z_axis_total_neurons: int
    gate_activation: ActivationOptions | None
    gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_gate_option: LayerGateOptions | None

    @classmethod
    def from_config(cls, config: ModuleType) -> NeuronDefaults:
        return cls(
            **{field.name: getattr(config, field.name.upper()) for field in fields(cls)}
        )


class _StackOptions(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


class _HaltingOptions(_StackOptions, Protocol):
    @property
    def stack_options(self) -> _StackOptions: ...


_Runtime = TypeVar("_Runtime")


class NeuronRuntimeResolver(ABC, Generic[_Runtime]):
    defaults: NeuronDefaults
    capacity_type: Callable[..., object]
    terminal_type: Callable[..., object]
    routing_tree_type: Callable[..., object]
    stack_type: Callable[..., _StackOptions]
    sampler_type: Callable[..., object]
    halting_type: Callable[..., _HaltingOptions]
    runtime_type: Callable[[dict[str, object]], _Runtime]

    def __init__(self, values: Mapping[str, object]) -> None:
        self.__values = dict(values or {})

    def __read(self, key: str) -> object:
        return self.__values.pop(key, getattr(self.defaults, key))

    @staticmethod
    @abstractmethod
    def _hidden_settings(values: dict[str, object]) -> tuple[object, object]: ...

    def resolve(self) -> _Runtime:
        cluster_capacity_options = self.__cluster_capacity_options()
        terminal_options = self.__terminal_options()
        terminal_router_options = self.__terminal_router_options()
        terminal_sampler_options = self.__terminal_sampler_options()
        cluster_halting_options = self.__cluster_halting_options()

        hidden_runtime, residual_stack_options = self._hidden_settings(
            self.__hidden_flat_options()
        )
        terminal_router_options = replace(
            terminal_router_options,
            residual_stack_options=residual_stack_options,
        )
        cluster_halting_options = replace(
            cluster_halting_options,
            stack_options=replace(
                cluster_halting_options.stack_options,
                residual_stack_options=residual_stack_options,
            ),
        )

        return self.runtime_type(
            {
                "hidden_runtime": hidden_runtime,
                "cluster_capacity_options": cluster_capacity_options,
                "terminal_options": terminal_options,
                "terminal_router_options": terminal_router_options,
                "terminal_sampler_options": terminal_sampler_options,
                "cluster_halting_options": cluster_halting_options,
            }
        )

    def __cluster_capacity_options(
        self,
    ) -> object:
        provided = self.__values.pop("cluster_capacity_options", None)
        x_axis_total_neurons = self.__read("cluster_x_axis_total_neurons")
        y_axis_total_neurons = self.__read("cluster_y_axis_total_neurons")
        z_axis_total_neurons = self.__read("cluster_z_axis_total_neurons")
        initial_x_axis_total_neurons = self.__read(
            "cluster_initial_x_axis_total_neurons"
        )
        initial_y_axis_total_neurons = self.__read(
            "cluster_initial_y_axis_total_neurons"
        )
        initial_z_axis_total_neurons = self.__read(
            "cluster_initial_z_axis_total_neurons"
        )
        max_steps = self.__read("cluster_max_steps")
        beam_width = self.__read("cluster_beam_width")
        growth_threshold = self.__read("cluster_growth_threshold")
        growth_cooldown_steps = self.__read("cluster_growth_cooldown_steps")
        max_total_growths = self.__read("cluster_max_total_growths")
        growth_warmup_steps = self.__read("cluster_growth_warmup_steps")
        pruning_threshold = self.__read("cluster_pruning_threshold")
        escape_driven_growth_flag = self.__read("cluster_escape_driven_growth_flag")
        mitosis_initialization_flag = self.__read("cluster_mitosis_initialization_flag")
        if provided:
            return provided
        return self.capacity_type(
            x_axis_total_neurons=x_axis_total_neurons,
            y_axis_total_neurons=y_axis_total_neurons,
            z_axis_total_neurons=z_axis_total_neurons,
            initial_x_axis_total_neurons=initial_x_axis_total_neurons,
            initial_y_axis_total_neurons=initial_y_axis_total_neurons,
            initial_z_axis_total_neurons=initial_z_axis_total_neurons,
            max_steps=max_steps,
            beam_width=beam_width,
            growth_threshold=growth_threshold,
            growth_cooldown_steps=growth_cooldown_steps,
            max_total_growths=max_total_growths,
            growth_warmup_steps=growth_warmup_steps,
            pruning_threshold=pruning_threshold,
            escape_driven_growth_flag=escape_driven_growth_flag,
            mitosis_initialization_flag=mitosis_initialization_flag,
        )

    def __terminal_options(self) -> object:
        provided = self.__values.pop("terminal_options", None)
        xy_axis_range = self.__read("cluster_terminal_xy_axis_range")
        z_axis_range = self.__read("cluster_terminal_z_axis_range")
        top_k = self.__read("cluster_terminal_top_k")
        routing_tree_depth = self.__read("cluster_terminal_routing_tree_depth")
        level_1_branch_count = self.__read(
            "cluster_terminal_routing_tree_level_1_branch_count"
        )
        level_1_top_k = self.__read("cluster_terminal_routing_tree_level_1_top_k")
        level_2_branch_count = self.__read(
            "cluster_terminal_routing_tree_level_2_branch_count"
        )
        level_2_top_k = self.__read("cluster_terminal_routing_tree_level_2_top_k")
        if provided:
            return provided
        routing_tree = self.__terminal_routing_tree_options(
            cast(TerminalRoutingTreeDepthOptions | None, routing_tree_depth),
            level_1_branch_count,
            level_1_top_k,
            level_2_branch_count,
            level_2_top_k,
        )
        return self.terminal_type(
            xy_axis_range=xy_axis_range,
            z_axis_range=z_axis_range,
            top_k=top_k,
            routing_tree=routing_tree,
        )

    def __terminal_routing_tree_options(
        self,
        depth: TerminalRoutingTreeDepthOptions | None,
        level_1_branch_count: object,
        level_1_top_k: object,
        level_2_branch_count: object,
        level_2_top_k: object,
    ) -> object | None:
        configured_levels = (
            level_1_branch_count,
            level_1_top_k,
            level_2_branch_count,
            level_2_top_k,
        )
        if depth is None:
            if any(value is not None for value in configured_levels):
                raise ValueError(
                    "cluster_terminal_routing_tree_depth must be set when routing "
                    "tree level options are provided."
                )
            return None

        required_level_count = depth.value - 1
        required_values = configured_levels[: required_level_count * 2]
        if any(value is None for value in required_values):
            raise ValueError(
                f"Terminal routing tree depth {depth.value} requires branch_count and "
                f"top_k for {required_level_count} direction level(s)."
            )
        unused_values = configured_levels[required_level_count * 2 :]
        if any(value is not None for value in unused_values):
            raise ValueError(
                f"Terminal routing tree depth {depth.value} does not accept level "
                f"{required_level_count + 1} options."
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in required_values
        ):
            raise ValueError(
                "Terminal routing tree branch counts and top-k values must be positive integers."
            )

        return self.routing_tree_type(
            depth=depth,
            direction_branch_counts=tuple(required_values[0::2]),
            direction_top_k=tuple(required_values[1::2]),
        )

    def __terminal_router_options(
        self,
    ) -> _StackOptions:
        provided = cast(
            _StackOptions | None,
            self.__values.pop("terminal_router_options", None),
        )
        hidden_dim = self.__read("cluster_terminal_router_hidden_dim")
        num_layers = self.__read("cluster_terminal_router_num_layers")
        last_layer_bias_option = self.__read(
            "cluster_terminal_router_last_layer_bias_option"
        )
        apply_output_postprocessing_flag = self.__read(
            "cluster_terminal_router_apply_output_postprocessing_flag"
        )
        activation = self.__read("cluster_terminal_router_activation")
        layer_norm_position = self.__read("cluster_terminal_router_layer_norm_position")
        normalization = self.__read("cluster_terminal_router_normalization")
        residual_connection_option = self.__read(
            "cluster_terminal_router_residual_connection_option"
        )
        residual_model_flag = self.__read("cluster_terminal_router_residual_model_flag")
        residual_block_size = self.__read("cluster_terminal_router_residual_block_size")
        residual_rms_norm_epsilon = self.__read(
            "cluster_terminal_router_residual_rms_norm_epsilon"
        )
        dropout_probability = self.__read("cluster_terminal_router_dropout_probability")
        bias_flag = self.__read("cluster_terminal_router_bias_flag")
        if provided:
            return provided
        return self.stack_type(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
            activation=activation,
            layer_norm_position=layer_norm_position,
            normalization=normalization,
            residual_connection_option=residual_connection_option,
            residual_block_size=residual_block_size,
            residual_rms_norm_epsilon=residual_rms_norm_epsilon,
            residual_model_flag=residual_model_flag,
            dropout_probability=dropout_probability,
            bias_flag=bias_flag,
        )

    def __terminal_sampler_options(
        self,
    ) -> object:
        provided = self.__values.pop("terminal_sampler_options", None)
        threshold = self.__read("cluster_terminal_sampler_threshold")
        filter_above_threshold = self.__read(
            "cluster_terminal_sampler_filter_above_threshold"
        )
        num_topk_samples = self.__read("cluster_terminal_sampler_num_topk_samples")
        normalize_probabilities_flag = self.__read(
            "cluster_terminal_sampler_normalize_probabilities_flag"
        )
        noisy_topk_flag = self.__read("cluster_terminal_sampler_noisy_topk_flag")
        coefficient_of_variation_loss_weight = self.__read(
            "cluster_terminal_sampler_coefficient_of_variation_loss_weight"
        )
        switch_loss_weight = self.__read("cluster_terminal_sampler_switch_loss_weight")
        zero_centred_loss_weight = self.__read(
            "cluster_terminal_sampler_zero_centred_loss_weight"
        )
        mutual_information_loss_weight = self.__read(
            "cluster_terminal_sampler_mutual_information_loss_weight"
        )
        if provided:
            return provided
        return self.sampler_type(
            threshold=threshold,
            filter_above_threshold=filter_above_threshold,
            num_topk_samples=num_topk_samples,
            normalize_probabilities_flag=normalize_probabilities_flag,
            noisy_topk_flag=noisy_topk_flag,
            coefficient_of_variation_loss_weight=coefficient_of_variation_loss_weight,
            switch_loss_weight=switch_loss_weight,
            zero_centred_loss_weight=zero_centred_loss_weight,
            mutual_information_loss_weight=mutual_information_loss_weight,
        )

    def __cluster_halting_options(
        self,
    ) -> _HaltingOptions:
        provided = cast(
            _HaltingOptions | None,
            self.__values.pop("cluster_halting_options", None),
        )
        enabled = self.__read("cluster_halting_flag")
        halting_option = self.__read("cluster_halting_option")
        threshold = self.__read("cluster_halting_threshold")
        dropout = self.__read("cluster_halting_dropout")
        hidden_state_mode = self.__read("cluster_halting_hidden_state_mode")
        output_dim = self.__read("cluster_halting_output_dim")
        hidden_dim = self.__read("cluster_halting_stack_hidden_dim")
        num_layers = self.__read("cluster_halting_stack_num_layers")
        last_layer_bias_option = self.__read(
            "cluster_halting_stack_last_layer_bias_option"
        )
        apply_output_postprocessing_flag = self.__read(
            "cluster_halting_stack_apply_output_postprocessing_flag"
        )
        activation = self.__read("cluster_halting_stack_activation")
        layer_norm_position = self.__read("cluster_halting_stack_layer_norm_position")
        normalization = self.__read("cluster_halting_stack_normalization")
        residual_connection_option = self.__read(
            "cluster_halting_stack_residual_connection_option"
        )
        residual_model_flag = self.__read("cluster_halting_stack_residual_model_flag")
        residual_block_size = self.__read("cluster_halting_stack_residual_block_size")
        residual_rms_norm_epsilon = self.__read(
            "cluster_halting_stack_residual_rms_norm_epsilon"
        )
        dropout_probability = self.__read("cluster_halting_stack_dropout_probability")
        bias_flag = self.__read("cluster_halting_stack_bias_flag")
        if provided:
            return provided
        return self.halting_type(
            enabled=enabled,
            halting_option=halting_option,
            threshold=threshold,
            dropout=dropout,
            hidden_state_mode=hidden_state_mode,
            stack_options=self.stack_type(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_postprocessing_flag=apply_output_postprocessing_flag,
                activation=activation,
                layer_norm_position=layer_norm_position,
                normalization=normalization,
                residual_connection_option=residual_connection_option,
                residual_block_size=residual_block_size,
                residual_rms_norm_epsilon=residual_rms_norm_epsilon,
                residual_model_flag=residual_model_flag,
                dropout_probability=dropout_probability,
                bias_flag=bias_flag,
            ),
            output_dim=output_dim,
        )

    def __hidden_flat_options(self) -> dict[str, object]:
        gate_option = self.__read("gate_option")
        gate_activation = self.__read("gate_activation")
        recurrent_gate_option = self.__read("recurrent_gate_option")
        recurrent_gate_activation = self.__read("recurrent_gate_activation")
        shared_gate_config = self.__values.pop("shared_gate_config", None)
        hidden_options = {
            "gate_option": gate_option,
            "gate_activation": gate_activation,
            "recurrent_gate_option": recurrent_gate_option,
            "recurrent_gate_activation": recurrent_gate_activation,
            **self.__values,
        }
        if shared_gate_config is not None:
            hidden_options["shared_gate_config"] = shared_gate_config
        return hidden_options
