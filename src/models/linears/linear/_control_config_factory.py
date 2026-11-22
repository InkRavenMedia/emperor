from typing import ClassVar

from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import DynamicMemoryConfig
from models._recurrent_block import RecurrentSchedule, build_recurrent_block
from models.linears.linear._residual import build_residual_config
from models.linears.linear.runtime_options import (
    ControllerStackOptions,
    GateOptions,
    HaltingOptions,
    LinearBuildSettings,
    MainStackOptions,
    MemoryOptions,
)


class ControlConfigFactory:
    __STICK_BREAKING_GATE_OUTPUT_DIM: ClassVar[int] = 2

    def __init__(self, build_settings: LinearBuildSettings) -> None:
        self.build_settings = build_settings

    def build_hidden_layer_and_shared_memory(
        self,
    ) -> tuple[LayerConfig, DynamicMemoryConfig | None]:
        gate_config = self.__gate_config(recurrent=False)
        halting_config = self.__halting_config(recurrent=False)
        memory_config = self.__memory_config()
        options = self.build_settings.stack
        residual_config = build_residual_config(
            options.residual_connection_option,
            options.residual_model_flag,
            self.build_settings.residual_stack,
            residual_block_size=options.residual_block_size,
            residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
            selector_field="STACK_RESIDUAL_CONNECTION_OPTION",
            model_flag_field="STACK_RESIDUAL_MODEL_FLAG",
        )
        hidden_layer_config = self.__build_stack_layer_config(
            options,
            residual_config=residual_config,
            gate_config=gate_config,
            halting_config=halting_config,
        )
        return hidden_layer_config, memory_config

    def wrap_recurrent(
        self,
        block_config: LayerStackConfig,
    ) -> LayerStackConfig | RecurrentLayerConfig:
        options = self.build_settings.recurrence
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
        options = self.build_settings.gate
        if recurrent:
            options = self.build_settings.recurrence.gate
        if not options.enabled:
            return None
        gate_model_config = self.__controller_stack(options)
        return GateConfig(
            model_config=gate_model_config,
            option=options.option,
            activation=options.activation,
        )

    def __halting_config(self, *, recurrent: bool) -> HaltingConfig | None:
        selection = self.build_settings.layer_halting
        if recurrent:
            selection = self.build_settings.recurrent_halting
        options = selection.options
        if not options.enabled:
            return None
        halting_gate_config = self.__controller_stack(options)
        halting_config = selection.implementation(
            threshold=options.threshold,
            min_steps=1,
            ponder_cost_weight=1.0,
            dropout_probability=options.dropout_probability,
            hidden_state_mode=options.hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )
        return halting_config

    def __memory_config(self) -> DynamicMemoryConfig | None:
        options = self.build_settings.memory
        if not options.enabled:
            return None
        memory_model_config = self.__controller_stack(options)
        memory_config = options.implementation(
            input_dim=self.build_settings.hidden_dim,
            output_dim=self.build_settings.hidden_dim,
            memory_position_option=options.position,
            test_time_training_learning_rate=options.test_time_training_learning_rate,
            test_time_training_num_inner_steps=(
                options.test_time_training_num_inner_steps
            ),
            model_config=memory_model_config,
        )
        return memory_config

    def __controller_stack(
        self,
        options: GateOptions | HaltingOptions | MemoryOptions,
    ) -> LayerStackConfig:
        stack = options.stack
        output_dim = None
        if isinstance(options, HaltingOptions):
            output_dim = self.__STICK_BREAKING_GATE_OUTPUT_DIM

        controller_layer_config = self.__controller_layer(stack)
        return LayerStackConfig(
            hidden_dim=stack.hidden_dim,
            output_dim=output_dim,
            num_layers=stack.num_layers,
            last_layer_bias_option=stack.last_layer_bias_option,
            apply_output_postprocessing_flag=stack.apply_output_postprocessing_flag,
            layer_config=controller_layer_config,
        )

    def __controller_layer(self, options: ControllerStackOptions) -> LayerConfig:
        residual_config = build_residual_config(
            options.residual_connection_option,
            options.residual_model_flag,
            self.build_settings.residual_stack,
            residual_block_size=options.residual_block_size,
            residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
        )
        controller_layer_config = self.__build_stack_layer_config(
            options, residual_config=residual_config
        )
        return controller_layer_config

    @staticmethod
    def __build_stack_layer_config(
        options: MainStackOptions | ControllerStackOptions,
        *,
        residual_config: ResidualConfig | None,
        gate_config: GateConfig | None = None,
        halting_config: HaltingConfig | None = None,
    ) -> LayerConfig:
        linear_config = LinearLayerConfig(bias_flag=options.bias_flag)
        return LayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            dropout_probability=options.dropout_probability,
            layer_model_config=linear_config,
            residual_config=residual_config,
            gate_config=gate_config,
            halting_config=halting_config,
        )
