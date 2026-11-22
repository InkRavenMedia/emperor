from typing import ClassVar

from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.memory import DynamicMemoryConfig
from models import _controller_stack
from models._recurrent_block import RecurrentSchedule, build_recurrent_block
from models.neuron.linear._hidden.runtime_options import (
    GateOptions,
    HaltingOptions,
    MemoryOptions,
    RuntimeOptions,
)


class ControlConfigFactory:
    __STICK_BREAKING_GATE_OUTPUT_DIM: ClassVar[int] = 2

    def __init__(self, runtime: RuntimeOptions) -> None:
        self.runtime = runtime

    def build_gate_config(self) -> GateConfig | None:
        gate_config = self.__gate_config(recurrent=False)
        return gate_config

    def build_halting_config(self) -> HaltingConfig | None:
        halting_config = self.__halting_config(recurrent=False)
        return halting_config

    def build_memory_config(self) -> DynamicMemoryConfig | None:
        options = self.runtime.memory
        if not options.enabled:
            return None
        memory_model_config = self.__controller_stack(options)
        memory_config = options.implementation(
            input_dim=self.runtime.hidden_dim,
            output_dim=self.runtime.hidden_dim,
            memory_position_option=options.position,
            test_time_training_learning_rate=(options.test_time_training_learning_rate),
            test_time_training_num_inner_steps=(
                options.test_time_training_num_inner_steps
            ),
            model_config=memory_model_config,
        )
        return memory_config

    def wrap_recurrent(
        self,
        block_config: LayerStackConfig,
    ) -> LayerStackConfig | RecurrentLayerConfig:
        options = self.runtime.recurrence
        if not options.enabled:
            return block_config
        recurrent_gate_config = self.__gate_config(recurrent=True)
        recurrent_halting_config = self.__halting_config(recurrent=True)
        schedule = RecurrentSchedule.from_options(options)
        return build_recurrent_block(
            schedule,
            block_config,
            smooth_iteration_growth_flag=options.smooth_iteration_growth_flag,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            gate_config=recurrent_gate_config,
            halting_config=recurrent_halting_config,
        )

    def __gate_config(self, *, recurrent: bool) -> GateConfig | None:
        options = self.runtime.gate
        if recurrent:
            options = self.runtime.recurrence.gate
        if not options.enabled:
            return None
        gate_model_config = self.__controller_stack(options)
        return GateConfig(
            model_config=gate_model_config,
            option=options.option,
            activation=options.activation,
        )

    def __halting_config(self, *, recurrent: bool) -> HaltingConfig | None:
        options = self.runtime.halting
        option = self.runtime.halting_option
        if recurrent:
            options = self.runtime.recurrence.halting
            option = self.runtime.recurrent_halting_option
        if not options.enabled:
            return None
        halting_gate_config = self.__controller_stack(options, halting_gate=True)
        return option(
            threshold=options.threshold,
            min_steps=1,
            ponder_cost_weight=1.0,
            dropout_probability=options.dropout_probability,
            hidden_state_mode=options.hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )

    def __controller_stack(
        self,
        options: GateOptions | HaltingOptions | MemoryOptions,
        *,
        halting_gate: bool = False,
    ) -> LayerStackConfig:
        stack = options.stack
        output_dim = None
        if halting_gate:
            output_dim = self.__STICK_BREAKING_GATE_OUTPUT_DIM

        return _controller_stack.build_controller_stack_config(
            stack, output_dim=output_dim
        )
