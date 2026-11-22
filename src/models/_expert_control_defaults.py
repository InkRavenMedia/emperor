from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Generic, Protocol, TypeVar

from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from model_runtime.packages._runtime_default_fields import read_optional_stack
from models._residual_stack import residual_source_from_settings

_Result = TypeVar("_Result")


class ControlSettings(Protocol):
    expert_stack_gate_flag: Final[bool]
    expert_gate_option: Final[LayerGateOptions | None]
    expert_gate_activation: Final[ActivationOptions | None]
    expert_stack_halting_flag: Final[bool]
    expert_halting_threshold: Final[float]
    expert_halting_dropout: Final[float]
    expert_halting_hidden_state_mode: Final[HaltingHiddenStateModeOptions]
    expert_halting_output_dim: Final[int]
    expert_memory_flag: Final[bool]
    expert_memory_option: Final[type[DynamicMemoryConfig]]
    expert_memory_position_option: Final[MemoryPositionOptions]
    expert_memory_test_time_training_learning_rate: Final[float | None]
    expert_memory_test_time_training_num_inner_steps: Final[int | None]
    expert_recurrent_flag: Final[bool]
    expert_recurrent_max_steps: Final[int]
    expert_recurrent_layer_norm_position: Final[LayerNormPositionOptions]
    expert_recurrent_normalization: Final[NormalizationOptions]
    expert_recurrent_stack_gate_flag: Final[bool]
    expert_recurrent_gate_option: Final[LayerGateOptions | None]
    expert_recurrent_gate_activation: Final[ActivationOptions | None]
    expert_recurrent_stack_halting_flag: Final[bool]
    expert_recurrent_halting_threshold: Final[float]
    expert_recurrent_halting_dropout: Final[float]
    expert_recurrent_halting_hidden_state_mode: Final[HaltingHiddenStateModeOptions]
    router_noisy_topk_flag: Final[bool]
    router_stack_hidden_dim: Final[int]
    router_stack_num_layers: Final[int]
    router_stack_activation: Final[ActivationOptions]
    router_stack_residual_connection_option: Final[type[ResidualConfig] | None]
    router_stack_residual_model_flag: Final[bool]
    router_stack_residual_block_size: Final[int | None]
    router_stack_residual_rms_norm_epsilon: Final[float | None]
    router_stack_dropout_probability: Final[float]
    router_stack_layer_norm_position: Final[LayerNormPositionOptions]
    router_stack_normalization: Final[NormalizationOptions]
    router_stack_last_layer_bias_option: Final[LastLayerBiasOptions]
    router_stack_apply_output_postprocessing_flag: Final[bool]
    router_bias_flag: Final[bool]
    router_stack_gate_flag: Final[bool]
    router_gate_option: Final[LayerGateOptions | None]
    router_gate_activation: Final[ActivationOptions | None]
    router_stack_halting_flag: Final[bool]
    router_halting_threshold: Final[float]
    router_halting_dropout: Final[float]
    router_halting_hidden_state_mode: Final[HaltingHiddenStateModeOptions]
    router_halting_output_dim: Final[int]
    router_memory_flag: Final[bool]
    router_memory_option: Final[type[DynamicMemoryConfig]]
    router_memory_position_option: Final[MemoryPositionOptions]
    router_memory_test_time_training_learning_rate: Final[float | None]
    router_memory_test_time_training_num_inner_steps: Final[int | None]
    router_recurrent_flag: Final[bool]
    router_recurrent_max_steps: Final[int]
    router_recurrent_layer_norm_position: Final[LayerNormPositionOptions]
    router_recurrent_normalization: Final[NormalizationOptions]
    router_recurrent_stack_gate_flag: Final[bool]
    router_recurrent_gate_option: Final[LayerGateOptions | None]
    router_recurrent_gate_activation: Final[ActivationOptions | None]
    router_recurrent_stack_halting_flag: Final[bool]
    router_recurrent_halting_threshold: Final[float]
    router_recurrent_halting_dropout: Final[float]
    router_recurrent_halting_hidden_state_mode: Final[HaltingHiddenStateModeOptions]
    stack_gate_flag: Final[bool]
    gate_option: Final[LayerGateOptions | None]
    gate_activation: Final[ActivationOptions | None]
    stack_halting_flag: Final[bool]
    halting_threshold: Final[float]
    halting_dropout: Final[float]
    halting_hidden_state_mode: Final[HaltingHiddenStateModeOptions]
    halting_output_dim: Final[int]
    memory_flag: Final[bool]
    memory_option: Final[type[DynamicMemoryConfig]]
    memory_position_option: Final[MemoryPositionOptions]
    memory_test_time_training_learning_rate: Final[float | None]
    memory_test_time_training_num_inner_steps: Final[int | None]
    recurrent_flag: Final[bool]
    recurrent_max_steps: Final[int]
    recurrent_initial_iterations: Final[int | None]
    recurrent_gradient_transition_count: Final[int | None]
    recurrent_no_gradient_transition_count: Final[int | None]
    recurrent_iteration_increment: Final[int]
    recurrent_forward_calls_before_iteration_increment: Final[int]
    recurrent_smooth_iteration_growth_flag: Final[bool]
    recurrent_layer_norm_position: Final[LayerNormPositionOptions]
    recurrent_normalization: Final[NormalizationOptions]
    recurrent_stack_gate_flag: Final[bool]
    recurrent_gate_option: Final[LayerGateOptions | None]
    recurrent_gate_activation: Final[ActivationOptions | None]
    recurrent_stack_halting_flag: Final[bool]
    recurrent_halting_threshold: Final[float]
    recurrent_halting_dropout: Final[float]
    recurrent_halting_hidden_state_mode: Final[HaltingHiddenStateModeOptions]
    shared_gate_config: Final[GateConfig | None]
    router_options: Final[object]
    router_stack_options: Final[object]
    router_layer_controller_options: Final[object]
    router_dynamic_memory_options: Final[object]
    router_recurrent_controller_options: Final[object]
    layer_controller_options: Final[object]
    dynamic_memory_options: Final[object]
    expert_layer_controller_options: Final[object]
    expert_dynamic_memory_options: Final[object]
    expert_recurrent_controller_options: Final[object]
    recurrent_controller_options: Final[object]
    halting_option: Final[type[HaltingConfig]]
    expert_halting_option: Final[type[HaltingConfig]]
    router_halting_option: Final[type[HaltingConfig]]
    recurrent_halting_option: Final[type[HaltingConfig]]
    expert_recurrent_halting_option: Final[type[HaltingConfig]]
    router_recurrent_halting_option: Final[type[HaltingConfig]]

    residual_coefficient_activation: Final[ActivationOptions]
    residual_stack_independent_flag: Final[bool]
    residual_stack_hidden_dim: Final[int | None]
    residual_stack_layer_norm_position: Final[LayerNormPositionOptions | None]
    residual_stack_normalization: Final[NormalizationOptions | None]
    residual_stack_num_layers: Final[int | None]
    residual_stack_activation: Final[ActivationOptions | None]
    residual_stack_residual_connection_option: Final[type[ResidualConfig] | None]
    residual_stack_residual_model_flag: Final[bool]
    residual_stack_residual_block_size: Final[int | None]
    residual_stack_residual_rms_norm_epsilon: Final[float | None]
    residual_stack_dropout_probability: Final[float | None]
    residual_stack_last_layer_bias_option: Final[LastLayerBiasOptions | None]
    residual_stack_apply_output_postprocessing_flag: Final[bool | None]
    residual_stack_bias_flag: Final[bool | None]


@dataclass(frozen=True, slots=True)
class _ExpertControlDefaults:
    layer: object
    memory: object
    recurrent: object


@dataclass(frozen=True, slots=True)
class _RouterControlDefaults:
    router: object
    stack: object
    layer: object
    memory: object
    recurrent: object


@dataclass(frozen=True, slots=True)
class _LayerControlDefaults:
    layer: object
    memory: object


@dataclass(frozen=True, slots=True)
class _RecurrentDefaults:
    recurrent: object
    residual_stack: object


class ExpertControlDefaults(Generic[_Result]):
    stack_type: Callable[..., object]
    source_type: Callable[..., object]
    layer_type: Callable[..., object]
    memory_type: Callable[..., object]
    recurrent_type: Callable[..., object]
    router_type: Callable[..., object]
    defaults_type: Callable[..., _Result]
    resolve_residual: Callable[..., object]
    residual_source_type: Callable[..., object]

    def _expert_control_defaults(
        self, values: ControlSettings
    ) -> _ExpertControlDefaults:
        expert_layer_controller_options = values.expert_layer_controller_options
        expert_dynamic_memory_options = values.expert_dynamic_memory_options
        expert_recurrent_controller_options = values.expert_recurrent_controller_options
        expert_layer_controller_options = (
            expert_layer_controller_options
            or self.layer_type(
                stack_gate_flag=values.expert_stack_gate_flag,
                gate_option=values.expert_gate_option,
                gate_activation=values.expert_gate_activation,
                gate_stack_source=read_optional_stack(
                    self.source_type, values, "expert_gate_stack"
                ),
                stack_halting_flag=values.expert_stack_halting_flag,
                halting_option=values.expert_halting_option,
                halting_threshold=values.expert_halting_threshold,
                halting_dropout=values.expert_halting_dropout,
                halting_hidden_state_mode=values.expert_halting_hidden_state_mode,
                halting_stack_source=read_optional_stack(
                    self.source_type, values, "expert_halting_stack"
                ),
                halting_output_dim=values.expert_halting_output_dim,
            )
        )
        expert_dynamic_memory_options = (
            expert_dynamic_memory_options
            or self.memory_type(
                memory_flag=values.expert_memory_flag,
                memory_option=values.expert_memory_option,
                memory_position_option=values.expert_memory_position_option,
                memory_test_time_training_learning_rate=values.expert_memory_test_time_training_learning_rate,
                memory_test_time_training_num_inner_steps=values.expert_memory_test_time_training_num_inner_steps,
                memory_stack_source=read_optional_stack(
                    self.source_type, values, "expert_memory_stack"
                ),
            )
        )
        expert_recurrent_controller_options = (
            expert_recurrent_controller_options
            or self.recurrent_type(
                recurrent_flag=values.expert_recurrent_flag,
                recurrent_max_steps=values.expert_recurrent_max_steps,
                recurrent_layer_norm_position=values.expert_recurrent_layer_norm_position,
                recurrent_normalization=values.expert_recurrent_normalization,
                recurrent_stack_gate_flag=values.expert_recurrent_stack_gate_flag,
                recurrent_gate_option=values.expert_recurrent_gate_option,
                recurrent_gate_activation=values.expert_recurrent_gate_activation,
                recurrent_gate_stack_source=read_optional_stack(
                    self.source_type, values, "expert_recurrent_gate_stack"
                ),
                recurrent_stack_halting_flag=values.expert_recurrent_stack_halting_flag,
                recurrent_halting_option=values.expert_recurrent_halting_option,
                recurrent_halting_threshold=values.expert_recurrent_halting_threshold,
                recurrent_halting_dropout=values.expert_recurrent_halting_dropout,
                recurrent_halting_hidden_state_mode=values.expert_recurrent_halting_hidden_state_mode,
                recurrent_halting_stack_source=read_optional_stack(
                    self.source_type, values, "expert_recurrent_halting_stack"
                ),
            )
        )
        return _ExpertControlDefaults(
            layer=expert_layer_controller_options,
            memory=expert_dynamic_memory_options,
            recurrent=expert_recurrent_controller_options,
        )

    def _router_control_defaults(
        self, values: ControlSettings
    ) -> _RouterControlDefaults:
        return _RouterControlDefaults(
            router=self._router_options_defaults(values),
            stack=self._router_stack_defaults(values),
            layer=self._router_layer_defaults(values),
            memory=self._router_memory_defaults(values),
            recurrent=self._router_recurrent_defaults(values),
        )

    def _router_options_defaults(self, values: ControlSettings) -> object:
        provided = values.router_options
        return provided or self.router_type(
            noisy_topk_flag=values.router_noisy_topk_flag
        )

    def _router_stack_defaults(self, values: ControlSettings) -> object:
        provided = values.router_stack_options
        return provided or self.stack_type(
            hidden_dim=values.router_stack_hidden_dim,
            num_layers=values.router_stack_num_layers,
            last_layer_bias_option=values.router_stack_last_layer_bias_option,
            apply_output_postprocessing_flag=values.router_stack_apply_output_postprocessing_flag,
            activation=values.router_stack_activation,
            layer_norm_position=values.router_stack_layer_norm_position,
            normalization=values.router_stack_normalization,
            residual_connection_option=values.router_stack_residual_connection_option,
            residual_block_size=values.router_stack_residual_block_size,
            residual_rms_norm_epsilon=values.router_stack_residual_rms_norm_epsilon,
            residual_model_flag=values.router_stack_residual_model_flag,
            dropout_probability=values.router_stack_dropout_probability,
            bias_flag=values.router_bias_flag,
        )

    def _router_layer_defaults(self, values: ControlSettings) -> object:
        provided = values.router_layer_controller_options
        return provided or self.layer_type(
            stack_gate_flag=values.router_stack_gate_flag,
            gate_option=values.router_gate_option,
            gate_activation=values.router_gate_activation,
            gate_stack_source=read_optional_stack(
                self.source_type, values, "router_gate_stack"
            ),
            stack_halting_flag=values.router_stack_halting_flag,
            halting_option=values.router_halting_option,
            halting_threshold=values.router_halting_threshold,
            halting_dropout=values.router_halting_dropout,
            halting_hidden_state_mode=values.router_halting_hidden_state_mode,
            halting_stack_source=read_optional_stack(
                self.source_type, values, "router_halting_stack"
            ),
            halting_output_dim=values.router_halting_output_dim,
        )

    def _router_memory_defaults(self, values: ControlSettings) -> object:
        provided = values.router_dynamic_memory_options
        return provided or self.memory_type(
            memory_flag=values.router_memory_flag,
            memory_option=values.router_memory_option,
            memory_position_option=values.router_memory_position_option,
            memory_test_time_training_learning_rate=values.router_memory_test_time_training_learning_rate,
            memory_test_time_training_num_inner_steps=values.router_memory_test_time_training_num_inner_steps,
            memory_stack_source=read_optional_stack(
                self.source_type, values, "router_memory_stack"
            ),
        )

    def _router_recurrent_defaults(self, values: ControlSettings) -> object:
        provided = values.router_recurrent_controller_options
        return provided or self.recurrent_type(
            recurrent_flag=values.router_recurrent_flag,
            recurrent_max_steps=values.router_recurrent_max_steps,
            recurrent_layer_norm_position=values.router_recurrent_layer_norm_position,
            recurrent_normalization=values.router_recurrent_normalization,
            recurrent_stack_gate_flag=values.router_recurrent_stack_gate_flag,
            recurrent_gate_option=values.router_recurrent_gate_option,
            recurrent_gate_activation=values.router_recurrent_gate_activation,
            recurrent_gate_stack_source=read_optional_stack(
                self.source_type, values, "router_recurrent_gate_stack"
            ),
            recurrent_stack_halting_flag=values.router_recurrent_stack_halting_flag,
            recurrent_halting_option=values.router_recurrent_halting_option,
            recurrent_halting_threshold=values.router_recurrent_halting_threshold,
            recurrent_halting_dropout=values.router_recurrent_halting_dropout,
            recurrent_halting_hidden_state_mode=values.router_recurrent_halting_hidden_state_mode,
            recurrent_halting_stack_source=read_optional_stack(
                self.source_type, values, "router_recurrent_halting_stack"
            ),
        )

    def _layer_control_defaults(self, values: ControlSettings) -> _LayerControlDefaults:
        layer_controller_options = values.layer_controller_options
        dynamic_memory_options = values.dynamic_memory_options
        layer_controller_options = layer_controller_options or self.layer_type(
            stack_gate_flag=values.stack_gate_flag,
            gate_option=values.gate_option,
            gate_activation=values.gate_activation,
            gate_stack_source=read_optional_stack(
                self.source_type, values, "gate_stack"
            ),
            stack_halting_flag=values.stack_halting_flag,
            halting_option=values.halting_option,
            halting_threshold=values.halting_threshold,
            halting_dropout=values.halting_dropout,
            halting_hidden_state_mode=values.halting_hidden_state_mode,
            halting_stack_source=read_optional_stack(
                self.source_type, values, "halting_stack"
            ),
            halting_output_dim=values.halting_output_dim,
            shared_gate_config=values.shared_gate_config,
        )
        dynamic_memory_options = dynamic_memory_options or self.memory_type(
            memory_flag=values.memory_flag,
            memory_option=values.memory_option,
            memory_position_option=values.memory_position_option,
            memory_test_time_training_learning_rate=values.memory_test_time_training_learning_rate,
            memory_test_time_training_num_inner_steps=values.memory_test_time_training_num_inner_steps,
            memory_stack_source=read_optional_stack(
                self.source_type, values, "memory_stack"
            ),
        )
        return _LayerControlDefaults(
            layer=layer_controller_options, memory=dynamic_memory_options
        )

    def _recurrent_defaults(
        self, values: ControlSettings, submodule_stack_options: object
    ) -> _RecurrentDefaults:
        recurrent_controller_options = values.recurrent_controller_options
        recurrent_controller_options = (
            recurrent_controller_options
            or self.recurrent_type(
                recurrent_flag=values.recurrent_flag,
                recurrent_max_steps=values.recurrent_max_steps,
                recurrent_initial_iterations=values.recurrent_initial_iterations,
                recurrent_gradient_transition_count=values.recurrent_gradient_transition_count,
                recurrent_no_gradient_transition_count=values.recurrent_no_gradient_transition_count,
                recurrent_iteration_increment=values.recurrent_iteration_increment,
                recurrent_forward_calls_before_iteration_increment=values.recurrent_forward_calls_before_iteration_increment,
                recurrent_smooth_iteration_growth_flag=values.recurrent_smooth_iteration_growth_flag,
                recurrent_layer_norm_position=values.recurrent_layer_norm_position,
                recurrent_normalization=values.recurrent_normalization,
                recurrent_stack_gate_flag=values.recurrent_stack_gate_flag,
                recurrent_gate_option=values.recurrent_gate_option,
                recurrent_gate_activation=values.recurrent_gate_activation,
                recurrent_gate_stack_source=read_optional_stack(
                    self.source_type, values, "recurrent_gate_stack"
                ),
                recurrent_stack_halting_flag=values.recurrent_stack_halting_flag,
                recurrent_halting_option=values.recurrent_halting_option,
                recurrent_halting_threshold=values.recurrent_halting_threshold,
                recurrent_halting_dropout=values.recurrent_halting_dropout,
                recurrent_halting_hidden_state_mode=values.recurrent_halting_hidden_state_mode,
                recurrent_halting_stack_source=read_optional_stack(
                    self.source_type, values, "recurrent_halting_stack"
                ),
            )
        )
        residual_stack_options = self.resolve_residual(
            residual_source_from_settings(values, self.residual_source_type),
            submodule_stack_options,
        )
        return _RecurrentDefaults(
            recurrent=recurrent_controller_options,
            residual_stack=residual_stack_options,
        )

    def resolve_control_defaults(
        self, values: ControlSettings, submodule_stack_options: object
    ) -> _Result:
        expert = self._expert_control_defaults(values)
        router = self._router_control_defaults(values)
        layer = self._layer_control_defaults(values)
        recurrent = self._recurrent_defaults(values, submodule_stack_options)
        return self.defaults_type(
            expert_layer_controller_options=expert.layer,
            expert_dynamic_memory_options=expert.memory,
            expert_recurrent_controller_options=expert.recurrent,
            router_options=router.router,
            router_stack_options=router.stack,
            router_layer_controller_options=router.layer,
            router_dynamic_memory_options=router.memory,
            router_recurrent_controller_options=router.recurrent,
            layer_controller_options=layer.layer,
            dynamic_memory_options=layer.memory,
            recurrent_controller_options=recurrent.recurrent,
            residual_stack_options=recurrent.residual_stack,
        )
