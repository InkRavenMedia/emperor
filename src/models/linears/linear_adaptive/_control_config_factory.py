from typing import ClassVar

from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import DynamicMemoryConfig
from models import _controller_stack
from models._recurrent_block import RecurrentSchedule, build_recurrent_block
from models.linears.linear_adaptive._residual import (
    build_residual_config,
    inherit_if_unset,
)
from models.linears.linear_adaptive.runtime_options import (
    GateOptions,
    HaltingOptions,
    MemoryOptions,
    RuntimeOptions,
)


class ControlConfigFactory:
    __STICK_BREAKING_GATE_OUTPUT_DIM: ClassVar[int] = 2

    def __init__(self, runtime: RuntimeOptions) -> None:
        self._runtime = runtime

    def build_gate_config(self, *, recurrent: bool = False) -> GateConfig | None:
        options = self._runtime.gate
        if recurrent:
            options = self._runtime.recurrence.gate
        if not options.enabled:
            return None
        gate_model_config = self.__controller_stack(options)
        return GateConfig(
            model_config=gate_model_config,
            option=options.option,
            activation=options.activation,
        )

    def build_halting_config(
        self,
        *,
        recurrent: bool = False,
    ) -> HaltingConfig | None:
        options = self._runtime.halting
        halting_option = self._runtime.halting_option
        if recurrent:
            options = self._runtime.recurrence.halting
            halting_option = inherit_if_unset(
                self._runtime.recurrent_halting_option, halting_option
            )
        if not options.enabled:
            return None
        halting_gate_config = self.__controller_stack(options, halting_gate=True)
        return halting_option(
            threshold=options.threshold,
            min_steps=1,
            ponder_cost_weight=1.0,
            dropout_probability=options.dropout_probability,
            hidden_state_mode=options.hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )

    def build_memory_config(self) -> DynamicMemoryConfig | None:
        options = self._runtime.memory
        if not options.enabled:
            return None
        memory_model_config = self.__controller_stack(options)
        return options.option(
            input_dim=self._runtime.hidden_dim,
            output_dim=self._runtime.hidden_dim,
            memory_position_option=options.position,
            test_time_training_learning_rate=(options.test_time_training_learning_rate),
            test_time_training_num_inner_steps=(
                options.test_time_training_num_inner_steps
            ),
            model_config=memory_model_config,
        )

    def apply_recurrence(
        self,
        block_config: LayerStackConfig,
    ) -> LayerStackConfig | RecurrentLayerConfig:
        recurrence = self._runtime.recurrence
        if not recurrence.enabled:
            return block_config
        gate_config = self.build_gate_config(recurrent=True)
        halting_config = self.build_halting_config(recurrent=True)
        schedule = RecurrentSchedule.from_options(recurrence)
        return build_recurrent_block(
            schedule,
            block_config,
            smooth_iteration_growth_flag=recurrence.smooth_iteration_growth_flag,
            layer_norm_position=recurrence.layer_norm_position,
            normalization=recurrence.normalization,
            gate_config=gate_config,
            halting_config=halting_config,
        )

    def __controller_stack(
        self,
        options: GateOptions | HaltingOptions | MemoryOptions,
        *,
        halting_gate: bool = False,
    ) -> LayerStackConfig:
        stack = options.stack
        hidden_dim = stack.hidden_dim
        output_dim = None
        if halting_gate:
            hidden_dim = hidden_dim or self._runtime.output_dim
            output_dim = self.__STICK_BREAKING_GATE_OUTPUT_DIM

        residual_config = build_residual_config(
            stack.residual_connection_option,
            stack.residual_model_flag,
            self._runtime.residual_stack,
            residual_block_size=stack.residual_block_size,
            residual_rms_norm_epsilon=stack.residual_rms_norm_epsilon,
        )
        linear_config = LinearLayerConfig(bias_flag=stack.bias_flag)
        return _controller_stack.assemble_controller_stack(
            stack,
            linear_config,
            residual_config,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
