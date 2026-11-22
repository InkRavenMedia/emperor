from collections.abc import Callable
from types import ModuleType
from typing import Generic, TypeVar

from model_runtime.packages._runtime_default_fields import read_optional_stack

_Stack = TypeVar("_Stack")
_Defaults = TypeVar("_Defaults")


class PlainExpertControlDefaults(Generic[_Stack, _Defaults]):
    stack_type: Callable[..., _Stack]
    defaults_type: Callable[..., _Defaults]
    source_type: Callable[..., object]
    layer_type: Callable[..., object]
    memory_type: Callable[..., object]
    recurrent_type: Callable[..., object]

    def router_stack_defaults(self, config: ModuleType) -> _Stack:
        return self.stack_type(
            hidden_dim=config.ROUTER_STACK_HIDDEN_DIM,
            num_layers=config.ROUTER_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ROUTER_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.ROUTER_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.ROUTER_STACK_ACTIVATION,
            layer_norm_position=config.ROUTER_STACK_LAYER_NORM_POSITION,
            normalization=config.ROUTER_STACK_NORMALIZATION,
            residual_connection_option=config.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_block_size=config.ROUTER_STACK_RESIDUAL_BLOCK_SIZE,
            residual_rms_norm_epsilon=config.ROUTER_STACK_RESIDUAL_RMS_NORM_EPSILON,
            residual_model_flag=config.ROUTER_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ROUTER_BIAS_FLAG,
        )

    def main_control_defaults(self, config: ModuleType) -> _Defaults:
        return self.defaults_type(
            layer=self.layer_type(
                stack_gate_flag=config.STACK_GATE_FLAG,
                gate_option=config.GATE_OPTION,
                gate_activation=config.GATE_ACTIVATION,
                gate_stack_source=self._stack_source(config, "gate_stack"),
                stack_halting_flag=config.STACK_HALTING_FLAG,
                halting_option=config.HALTING_OPTION,
                halting_threshold=config.HALTING_THRESHOLD,
                halting_dropout=config.HALTING_DROPOUT,
                halting_hidden_state_mode=config.HALTING_HIDDEN_STATE_MODE,
                halting_stack_source=self._stack_source(config, "halting_stack"),
                halting_output_dim=config.HALTING_OUTPUT_DIM,
            ),
            memory=self.memory_type(
                memory_flag=config.MEMORY_FLAG,
                memory_option=config.MEMORY_OPTION,
                memory_position_option=config.MEMORY_POSITION_OPTION,
                memory_test_time_training_learning_rate=config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
                memory_test_time_training_num_inner_steps=config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
                memory_stack_source=self._stack_source(config, "memory_stack"),
            ),
            recurrent=self.recurrent_type(
                recurrent_flag=config.RECURRENT_FLAG,
                recurrent_max_steps=config.RECURRENT_MAX_STEPS,
                recurrent_initial_iterations=config.RECURRENT_INITIAL_ITERATIONS,
                recurrent_gradient_transition_count=config.RECURRENT_GRADIENT_TRANSITION_COUNT,
                recurrent_no_gradient_transition_count=config.RECURRENT_NO_GRADIENT_TRANSITION_COUNT,
                recurrent_iteration_increment=config.RECURRENT_ITERATION_INCREMENT,
                recurrent_forward_calls_before_iteration_increment=config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT,
                recurrent_smooth_iteration_growth_flag=config.RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG,
                recurrent_layer_norm_position=config.RECURRENT_LAYER_NORM_POSITION,
                recurrent_normalization=config.RECURRENT_NORMALIZATION,
                recurrent_stack_gate_flag=config.RECURRENT_STACK_GATE_FLAG,
                recurrent_gate_option=config.RECURRENT_GATE_OPTION,
                recurrent_gate_activation=config.RECURRENT_GATE_ACTIVATION,
                recurrent_gate_stack_source=self._stack_source(
                    config, "recurrent_gate_stack"
                ),
                recurrent_stack_halting_flag=config.RECURRENT_STACK_HALTING_FLAG,
                recurrent_halting_option=config.RECURRENT_HALTING_OPTION,
                recurrent_halting_threshold=config.RECURRENT_HALTING_THRESHOLD,
                recurrent_halting_dropout=config.RECURRENT_HALTING_DROPOUT,
                recurrent_halting_hidden_state_mode=config.RECURRENT_HALTING_HIDDEN_STATE_MODE,
                recurrent_halting_stack_source=self._stack_source(
                    config, "recurrent_halting_stack"
                ),
            ),
        )

    def expert_control_defaults(self, config: ModuleType) -> _Defaults:
        return self.defaults_type(
            layer=self.layer_type(
                stack_gate_flag=config.EXPERT_STACK_GATE_FLAG,
                gate_option=config.EXPERT_GATE_OPTION,
                gate_activation=config.EXPERT_GATE_ACTIVATION,
                gate_stack_source=self._stack_source(config, "expert_gate_stack"),
                stack_halting_flag=config.EXPERT_STACK_HALTING_FLAG,
                halting_option=config.EXPERT_HALTING_OPTION,
                halting_threshold=config.EXPERT_HALTING_THRESHOLD,
                halting_dropout=config.EXPERT_HALTING_DROPOUT,
                halting_hidden_state_mode=config.EXPERT_HALTING_HIDDEN_STATE_MODE,
                halting_stack_source=self._stack_source(config, "expert_halting_stack"),
                halting_output_dim=config.EXPERT_HALTING_OUTPUT_DIM,
            ),
            memory=self.memory_type(
                memory_flag=config.EXPERT_MEMORY_FLAG,
                memory_option=config.EXPERT_MEMORY_OPTION,
                memory_position_option=config.EXPERT_MEMORY_POSITION_OPTION,
                memory_test_time_training_learning_rate=config.EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
                memory_test_time_training_num_inner_steps=config.EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
                memory_stack_source=self._stack_source(config, "expert_memory_stack"),
            ),
            recurrent=self.recurrent_type(
                recurrent_flag=config.EXPERT_RECURRENT_FLAG,
                recurrent_max_steps=config.EXPERT_RECURRENT_MAX_STEPS,
                recurrent_initial_iterations=2,
                recurrent_gradient_transition_count=None,
                recurrent_no_gradient_transition_count=None,
                recurrent_iteration_increment=1,
                recurrent_forward_calls_before_iteration_increment=1,
                recurrent_smooth_iteration_growth_flag=False,
                recurrent_layer_norm_position=config.EXPERT_RECURRENT_LAYER_NORM_POSITION,
                recurrent_normalization=config.EXPERT_RECURRENT_NORMALIZATION,
                recurrent_stack_gate_flag=config.EXPERT_RECURRENT_STACK_GATE_FLAG,
                recurrent_gate_option=config.EXPERT_RECURRENT_GATE_OPTION,
                recurrent_gate_activation=config.EXPERT_RECURRENT_GATE_ACTIVATION,
                recurrent_gate_stack_source=self._stack_source(
                    config, "expert_recurrent_gate_stack"
                ),
                recurrent_stack_halting_flag=config.EXPERT_RECURRENT_STACK_HALTING_FLAG,
                recurrent_halting_option=config.EXPERT_RECURRENT_HALTING_OPTION,
                recurrent_halting_threshold=config.EXPERT_RECURRENT_HALTING_THRESHOLD,
                recurrent_halting_dropout=config.EXPERT_RECURRENT_HALTING_DROPOUT,
                recurrent_halting_hidden_state_mode=config.EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE,
                recurrent_halting_stack_source=self._stack_source(
                    config, "expert_recurrent_halting_stack"
                ),
            ),
        )

    def _stack_source(self, config: ModuleType, prefix: str) -> object:
        return read_optional_stack(
            self.source_type, config, prefix, config_constants=True
        )
