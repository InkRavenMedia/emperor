from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from model_runtime.packages._runtime_default_fields import read_optional_stack

from ._settings import ControlSettings, ExpertControlSettings

T = TypeVar("T")
S = TypeVar("S", bound=ControlSettings)
E = TypeVar("E", bound=ExpertControlSettings)


def inherit_if_unset(value: T | None, default: T) -> T:
    if value is None:
        return default
    return value


@dataclass(frozen=True, slots=True)
class StackOptions:
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    dropout_probability: float
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    bias_flag: bool


@dataclass(frozen=True, slots=True)
class ControllerStackSource:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    activation: ActivationOptions | None
    dropout_probability: float | None
    layer_norm_position: LayerNormPositionOptions | None
    normalization: NormalizationOptions | None = field(default=None, kw_only=True)
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    bias_flag: bool | None

    def resolve(self, defaults: StackOptions) -> StackOptions:
        if not self.independent_flag:
            return defaults
        return StackOptions(
            hidden_dim=inherit_if_unset(self.hidden_dim, defaults.hidden_dim),
            num_layers=inherit_if_unset(self.num_layers, defaults.num_layers),
            activation=inherit_if_unset(self.activation, defaults.activation),
            dropout_probability=inherit_if_unset(
                self.dropout_probability, defaults.dropout_probability
            ),
            layer_norm_position=inherit_if_unset(
                self.layer_norm_position, defaults.layer_norm_position
            ),
            normalization=inherit_if_unset(self.normalization, defaults.normalization),
            residual_connection_option=inherit_if_unset(
                self.residual_connection_option, defaults.residual_connection_option
            ),
            residual_block_size=self.residual_block_size,
            residual_rms_norm_epsilon=self.residual_rms_norm_epsilon,
            residual_model_flag=self.residual_model_flag,
            last_layer_bias_option=inherit_if_unset(
                self.last_layer_bias_option, defaults.last_layer_bias_option
            ),
            apply_output_postprocessing_flag=inherit_if_unset(
                self.apply_output_postprocessing_flag,
                defaults.apply_output_postprocessing_flag,
            ),
            bias_flag=inherit_if_unset(self.bias_flag, defaults.bias_flag),
        )


@dataclass(frozen=True, slots=True)
class GateOptions:
    enabled: bool
    option: LayerGateOptions | None
    activation: ActivationOptions | None
    stack: ControllerStackSource


@dataclass(frozen=True, slots=True)
class HaltingOptions:
    enabled: bool
    implementation: type[HaltingConfig]
    threshold: float
    dropout_probability: float
    hidden_state_mode: HaltingHiddenStateModeOptions
    stack: ControllerStackSource


@dataclass(frozen=True, slots=True)
class MemoryOptions:
    enabled: bool
    implementation: type[DynamicMemoryConfig]
    position: MemoryPositionOptions
    test_time_training_learning_rate: float | None
    test_time_training_num_inner_steps: int | None
    stack: ControllerStackSource
    num_memory_slots: int = 4


H = TypeVar("H", bound=HaltingOptions)


@dataclass(frozen=True, slots=True)
class RecurrentOptions:
    enabled: bool
    max_steps: int
    gradient_transition_count: int | None
    no_gradient_transition_count: int | None
    initial_iterations: int
    iteration_increment: int
    forward_calls_before_iteration_increment: int
    smooth_iteration_growth_flag: bool
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    gate: GateOptions
    halting: HaltingOptions
    memory: MemoryOptions | None


@dataclass(frozen=True, slots=True)
class ControlOptions:
    gate: GateOptions
    halting: HaltingOptions
    memory: MemoryOptions
    recurrent: RecurrentOptions


def token_mixer_stack_options(
    runtime: ControlSettings, *, bias_flag: bool
) -> StackOptions:
    return StackOptions(
        hidden_dim=runtime.token_mixer_stack_hidden_dim,
        num_layers=runtime.token_mixer_num_layers,
        activation=runtime.token_mixer_stack_activation,
        dropout_probability=runtime.token_mixer_stack_dropout_probability,
        layer_norm_position=runtime.token_mixer_stack_layer_norm_position,
        normalization=runtime.token_mixer_stack_normalization,
        residual_connection_option=runtime.token_mixer_stack_residual_connection_option,
        residual_block_size=runtime.token_mixer_stack_residual_block_size,
        residual_rms_norm_epsilon=runtime.token_mixer_stack_residual_rms_norm_epsilon,
        residual_model_flag=runtime.token_mixer_stack_residual_model_flag,
        last_layer_bias_option=runtime.token_mixer_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=runtime.token_mixer_stack_apply_output_postprocessing_flag,
        bias_flag=bias_flag,
    )


def channel_mixer_stack_options(
    runtime: ControlSettings, *, bias_flag: bool
) -> StackOptions:
    return StackOptions(
        hidden_dim=runtime.channel_mixer_stack_hidden_dim,
        num_layers=runtime.channel_mixer_num_layers,
        activation=runtime.channel_mixer_stack_activation,
        dropout_probability=runtime.channel_mixer_stack_dropout_probability,
        layer_norm_position=runtime.channel_mixer_stack_layer_norm_position,
        normalization=runtime.channel_mixer_stack_normalization,
        residual_connection_option=runtime.channel_mixer_stack_residual_connection_option,
        residual_block_size=runtime.channel_mixer_stack_residual_block_size,
        residual_rms_norm_epsilon=runtime.channel_mixer_stack_residual_rms_norm_epsilon,
        residual_model_flag=runtime.channel_mixer_stack_residual_model_flag,
        last_layer_bias_option=runtime.channel_mixer_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=runtime.channel_mixer_stack_apply_output_postprocessing_flag,
        bias_flag=bias_flag,
    )


def submodule_stack_options(runtime: ControlSettings) -> StackOptions:
    return StackOptions(
        hidden_dim=runtime.submodule_stack_hidden_dim,
        num_layers=runtime.submodule_stack_num_layers,
        activation=runtime.submodule_stack_activation,
        dropout_probability=runtime.submodule_stack_dropout_probability,
        layer_norm_position=runtime.submodule_stack_layer_norm_position,
        normalization=runtime.submodule_stack_normalization,
        residual_connection_option=runtime.submodule_stack_residual_connection_option,
        residual_block_size=runtime.submodule_stack_residual_block_size,
        residual_rms_norm_epsilon=runtime.submodule_stack_residual_rms_norm_epsilon,
        residual_model_flag=runtime.submodule_stack_residual_model_flag,
        last_layer_bias_option=runtime.submodule_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=runtime.submodule_stack_apply_output_postprocessing_flag,
        bias_flag=runtime.submodule_stack_bias_flag,
    )


C = TypeVar("C", bound=ControlOptions)


class MixerControls(Generic[S, H, C]):
    def __init__(
        self,
        halting_type: type[H],
        control_type: type[C],
        recurrent_type: type[RecurrentOptions] = RecurrentOptions,
    ) -> None:
        self.halting_type = halting_type
        self.control_type = control_type
        self.recurrent_type = recurrent_type

    def main_control_options(self, runtime: S) -> C:
        memory = MemoryOptions(
            enabled=runtime.memory_flag,
            implementation=runtime.memory_option,
            position=runtime.memory_position_option,
            test_time_training_learning_rate=(
                runtime.memory_test_time_training_learning_rate
            ),
            num_memory_slots=runtime.memory_num_memory_slots,
            test_time_training_num_inner_steps=(
                runtime.memory_test_time_training_num_inner_steps
            ),
            stack=read_optional_stack(ControllerStackSource, runtime, "memory_stack"),
        )
        return self.control_type(
            gate=GateOptions(
                enabled=runtime.stack_gate_flag,
                option=runtime.gate_option,
                activation=runtime.gate_activation,
                stack=read_optional_stack(ControllerStackSource, runtime, "gate_stack"),
            ),
            halting=self.halting_type(
                enabled=runtime.stack_halting_flag,
                implementation=runtime.halting_option,
                threshold=runtime.halting_threshold,
                dropout_probability=runtime.halting_dropout,
                hidden_state_mode=runtime.halting_hidden_state_mode,
                stack=read_optional_stack(
                    ControllerStackSource, runtime, "halting_stack"
                ),
            ),
            memory=memory,
            recurrent=self.recurrent_type(
                enabled=runtime.recurrent_flag,
                max_steps=runtime.recurrent_max_steps,
                gradient_transition_count=runtime.recurrent_gradient_transition_count,
                no_gradient_transition_count=runtime.recurrent_no_gradient_transition_count,
                initial_iterations=runtime.recurrent_initial_iterations,
                iteration_increment=runtime.recurrent_iteration_increment,
                forward_calls_before_iteration_increment=(
                    runtime.recurrent_forward_calls_before_iteration_increment
                ),
                smooth_iteration_growth_flag=(
                    runtime.recurrent_smooth_iteration_growth_flag
                ),
                layer_norm_position=runtime.recurrent_layer_norm_position,
                normalization=runtime.recurrent_normalization,
                residual_connection_option=runtime.recurrent_residual_connection_option,
                residual_block_size=runtime.recurrent_residual_block_size,
                residual_rms_norm_epsilon=runtime.recurrent_residual_rms_norm_epsilon,
                residual_model_flag=runtime.recurrent_residual_model_flag,
                gate=GateOptions(
                    enabled=runtime.recurrent_stack_gate_flag,
                    option=runtime.recurrent_gate_option,
                    activation=runtime.recurrent_gate_activation,
                    stack=read_optional_stack(
                        ControllerStackSource, runtime, "recurrent_gate_stack"
                    ),
                ),
                halting=self.main_recurrent_halting(runtime),
                memory=MemoryOptions(
                    enabled=runtime.recurrent_memory_flag,
                    implementation=memory.implementation,
                    position=memory.position,
                    test_time_training_learning_rate=(
                        memory.test_time_training_learning_rate
                    ),
                    num_memory_slots=memory.num_memory_slots,
                    test_time_training_num_inner_steps=(
                        memory.test_time_training_num_inner_steps
                    ),
                    stack=memory.stack,
                ),
            ),
        )

    def main_recurrent_halting(self, runtime: S) -> H:
        return self.halting_type(
            enabled=runtime.recurrent_stack_halting_flag,
            implementation=runtime.recurrent_halting_option,
            threshold=runtime.recurrent_halting_threshold,
            dropout_probability=runtime.recurrent_halting_dropout,
            hidden_state_mode=runtime.recurrent_halting_hidden_state_mode,
            stack=read_optional_stack(
                ControllerStackSource, runtime, "recurrent_halting_stack"
            ),
        )

    def token_mixer_control_options(self, runtime: S) -> C:
        return self.control_type(
            gate=GateOptions(
                enabled=runtime.token_mixer_stack_gate_flag,
                option=runtime.token_mixer_gate_option,
                activation=runtime.token_mixer_gate_activation,
                stack=read_optional_stack(
                    ControllerStackSource, runtime, "token_mixer_gate_stack"
                ),
            ),
            halting=self.halting_type(
                enabled=runtime.token_mixer_stack_halting_flag,
                implementation=runtime.token_mixer_halting_option,
                threshold=runtime.token_mixer_halting_threshold,
                dropout_probability=runtime.token_mixer_halting_dropout,
                hidden_state_mode=runtime.token_mixer_halting_hidden_state_mode,
                stack=read_optional_stack(
                    ControllerStackSource, runtime, "token_mixer_halting_stack"
                ),
            ),
            memory=MemoryOptions(
                enabled=runtime.token_mixer_memory_flag,
                implementation=runtime.token_mixer_memory_option,
                position=runtime.token_mixer_memory_position_option,
                test_time_training_learning_rate=(
                    runtime.token_mixer_memory_test_time_training_learning_rate
                ),
                num_memory_slots=runtime.token_mixer_memory_num_memory_slots,
                test_time_training_num_inner_steps=(
                    runtime.token_mixer_memory_test_time_training_num_inner_steps
                ),
                stack=read_optional_stack(
                    ControllerStackSource, runtime, "token_mixer_memory_stack"
                ),
            ),
            recurrent=self.recurrent_type(
                enabled=runtime.token_mixer_recurrent_flag,
                max_steps=runtime.token_mixer_recurrent_max_steps,
                gradient_transition_count=None,
                no_gradient_transition_count=None,
                initial_iterations=runtime.token_mixer_recurrent_max_steps,
                iteration_increment=runtime.recurrent_iteration_increment,
                forward_calls_before_iteration_increment=(
                    runtime.recurrent_forward_calls_before_iteration_increment
                ),
                smooth_iteration_growth_flag=False,
                layer_norm_position=runtime.token_mixer_recurrent_layer_norm_position,
                normalization=runtime.token_mixer_recurrent_normalization,
                residual_connection_option=(
                    runtime.token_mixer_recurrent_residual_connection_option
                ),
                residual_block_size=runtime.token_mixer_recurrent_residual_block_size,
                residual_rms_norm_epsilon=runtime.token_mixer_recurrent_residual_rms_norm_epsilon,
                residual_model_flag=runtime.token_mixer_recurrent_residual_model_flag,
                gate=GateOptions(
                    enabled=runtime.token_mixer_recurrent_stack_gate_flag,
                    option=runtime.token_mixer_recurrent_gate_option,
                    activation=runtime.token_mixer_recurrent_gate_activation,
                    stack=read_optional_stack(
                        ControllerStackSource,
                        runtime,
                        "token_mixer_recurrent_gate_stack",
                    ),
                ),
                halting=self.halting_type(
                    enabled=runtime.token_mixer_recurrent_stack_halting_flag,
                    implementation=runtime.token_mixer_recurrent_halting_option,
                    threshold=runtime.token_mixer_recurrent_halting_threshold,
                    dropout_probability=runtime.token_mixer_recurrent_halting_dropout,
                    hidden_state_mode=(
                        runtime.token_mixer_recurrent_halting_hidden_state_mode
                    ),
                    stack=read_optional_stack(
                        ControllerStackSource,
                        runtime,
                        "token_mixer_recurrent_halting_stack",
                    ),
                ),
                memory=None,
            ),
        )

    def channel_mixer_control_options(self, runtime: S) -> C:
        return self.control_type(
            gate=GateOptions(
                enabled=runtime.channel_mixer_stack_gate_flag,
                option=runtime.channel_mixer_gate_option,
                activation=runtime.channel_mixer_gate_activation,
                stack=read_optional_stack(
                    ControllerStackSource, runtime, "channel_mixer_gate_stack"
                ),
            ),
            halting=self.halting_type(
                enabled=runtime.channel_mixer_stack_halting_flag,
                implementation=runtime.channel_mixer_halting_option,
                threshold=runtime.channel_mixer_halting_threshold,
                dropout_probability=runtime.channel_mixer_halting_dropout,
                hidden_state_mode=runtime.channel_mixer_halting_hidden_state_mode,
                stack=read_optional_stack(
                    ControllerStackSource, runtime, "channel_mixer_halting_stack"
                ),
            ),
            memory=MemoryOptions(
                enabled=runtime.channel_mixer_memory_flag,
                implementation=runtime.channel_mixer_memory_option,
                position=runtime.channel_mixer_memory_position_option,
                test_time_training_learning_rate=(
                    runtime.channel_mixer_memory_test_time_training_learning_rate
                ),
                num_memory_slots=runtime.channel_mixer_memory_num_memory_slots,
                test_time_training_num_inner_steps=(
                    runtime.channel_mixer_memory_test_time_training_num_inner_steps
                ),
                stack=read_optional_stack(
                    ControllerStackSource, runtime, "channel_mixer_memory_stack"
                ),
            ),
            recurrent=self.recurrent_type(
                enabled=runtime.channel_mixer_recurrent_flag,
                max_steps=runtime.channel_mixer_recurrent_max_steps,
                gradient_transition_count=None,
                no_gradient_transition_count=None,
                initial_iterations=runtime.channel_mixer_recurrent_max_steps,
                iteration_increment=runtime.recurrent_iteration_increment,
                forward_calls_before_iteration_increment=(
                    runtime.recurrent_forward_calls_before_iteration_increment
                ),
                smooth_iteration_growth_flag=False,
                layer_norm_position=runtime.channel_mixer_recurrent_layer_norm_position,
                normalization=runtime.channel_mixer_recurrent_normalization,
                residual_connection_option=(
                    runtime.channel_mixer_recurrent_residual_connection_option
                ),
                residual_block_size=runtime.channel_mixer_recurrent_residual_block_size,
                residual_rms_norm_epsilon=runtime.channel_mixer_recurrent_residual_rms_norm_epsilon,
                residual_model_flag=(
                    runtime.channel_mixer_recurrent_residual_model_flag
                ),
                gate=GateOptions(
                    enabled=runtime.channel_mixer_recurrent_stack_gate_flag,
                    option=runtime.channel_mixer_recurrent_gate_option,
                    activation=runtime.channel_mixer_recurrent_gate_activation,
                    stack=read_optional_stack(
                        ControllerStackSource,
                        runtime,
                        "channel_mixer_recurrent_gate_stack",
                    ),
                ),
                halting=self.halting_type(
                    enabled=runtime.channel_mixer_recurrent_stack_halting_flag,
                    implementation=runtime.channel_mixer_recurrent_halting_option,
                    threshold=runtime.channel_mixer_recurrent_halting_threshold,
                    dropout_probability=(
                        runtime.channel_mixer_recurrent_halting_dropout
                    ),
                    hidden_state_mode=(
                        runtime.channel_mixer_recurrent_halting_hidden_state_mode
                    ),
                    stack=read_optional_stack(
                        ControllerStackSource,
                        runtime,
                        "channel_mixer_recurrent_halting_stack",
                    ),
                ),
                memory=None,
            ),
        )


class ExpertMixerControls(MixerControls[E, H, C]):
    def expert_control_options(self, runtime: E) -> C:
        return self.control_type(
            gate=GateOptions(
                enabled=runtime.expert_stack_gate_flag,
                option=runtime.expert_gate_option,
                activation=runtime.expert_gate_activation,
                stack=read_optional_stack(
                    ControllerStackSource, runtime, "expert_gate_stack"
                ),
            ),
            halting=self.halting_type(
                enabled=runtime.expert_stack_halting_flag,
                implementation=runtime.expert_halting_option,
                threshold=runtime.expert_halting_threshold,
                dropout_probability=runtime.expert_halting_dropout,
                hidden_state_mode=runtime.expert_halting_hidden_state_mode,
                stack=read_optional_stack(
                    ControllerStackSource, runtime, "expert_halting_stack"
                ),
            ),
            memory=MemoryOptions(
                enabled=runtime.expert_memory_flag,
                implementation=runtime.expert_memory_option,
                position=runtime.expert_memory_position_option,
                test_time_training_learning_rate=(
                    runtime.expert_memory_test_time_training_learning_rate
                ),
                num_memory_slots=runtime.expert_memory_num_memory_slots,
                test_time_training_num_inner_steps=(
                    runtime.expert_memory_test_time_training_num_inner_steps
                ),
                stack=read_optional_stack(
                    ControllerStackSource, runtime, "expert_memory_stack"
                ),
            ),
            recurrent=self.recurrent_type(
                enabled=runtime.expert_recurrent_flag,
                max_steps=runtime.expert_recurrent_max_steps,
                gradient_transition_count=None,
                no_gradient_transition_count=None,
                initial_iterations=runtime.expert_recurrent_max_steps,
                iteration_increment=runtime.recurrent_iteration_increment,
                forward_calls_before_iteration_increment=(
                    runtime.recurrent_forward_calls_before_iteration_increment
                ),
                smooth_iteration_growth_flag=False,
                layer_norm_position=runtime.expert_recurrent_layer_norm_position,
                normalization=runtime.expert_recurrent_normalization,
                residual_connection_option=(
                    runtime.expert_recurrent_residual_connection_option
                ),
                residual_block_size=runtime.expert_recurrent_residual_block_size,
                residual_rms_norm_epsilon=runtime.expert_recurrent_residual_rms_norm_epsilon,
                residual_model_flag=runtime.expert_recurrent_residual_model_flag,
                gate=GateOptions(
                    enabled=runtime.expert_recurrent_stack_gate_flag,
                    option=runtime.expert_recurrent_gate_option,
                    activation=runtime.expert_recurrent_gate_activation,
                    stack=read_optional_stack(
                        ControllerStackSource, runtime, "expert_recurrent_gate_stack"
                    ),
                ),
                halting=self.halting_type(
                    enabled=runtime.expert_recurrent_stack_halting_flag,
                    implementation=runtime.expert_recurrent_halting_option,
                    threshold=runtime.expert_recurrent_halting_threshold,
                    dropout_probability=runtime.expert_recurrent_halting_dropout,
                    hidden_state_mode=runtime.expert_recurrent_halting_hidden_state_mode,
                    stack=read_optional_stack(
                        ControllerStackSource, runtime, "expert_recurrent_halting_stack"
                    ),
                ),
                memory=None,
            ),
        )
