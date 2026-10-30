import math
from numbers import Real

from emperor.layers import (
    LayerStackConfig,
    RecurrentLayerConfig,
)
from models.vit.expert_linear_adaptive._gate_config_factory import GateConfigFactory
from models.vit.expert_linear_adaptive._halting_config_factory import (
    HaltingConfigFactory,
)
from models.vit.expert_linear_adaptive.runtime_options import RecurrentControllerOptions

from ._residual import build_residual_config


class RecurrentConfigFactory:
    def __init__(
        self,
        *,
        recurrent_controller_options: RecurrentControllerOptions,
        gate_config_factory: GateConfigFactory,
        halting_config_factory: HaltingConfigFactory,
    ) -> None:
        self.recurrent_controller_options = recurrent_controller_options
        self.gate_config_factory = gate_config_factory
        self.halting_config_factory = halting_config_factory

    def build_config(
        self,
        block_config: LayerStackConfig,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ) -> LayerStackConfig | RecurrentLayerConfig:
        self.__validate_recurrent_halting_configuration()
        if not self.recurrent_controller_options.recurrent_flag:
            return block_config
        gate_config = self.gate_config_factory.build_recurrent_gate_config()
        halting_config = self.halting_config_factory.build_recurrent_halting_config()
        return RecurrentLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            max_steps=self.recurrent_controller_options.recurrent_max_steps,
            gradient_transition_count=(
                self.recurrent_controller_options.recurrent_gradient_transition_count
            ),
            no_gradient_transition_count=(
                self.recurrent_controller_options.recurrent_no_gradient_transition_count
            ),
            initial_iterations=self.recurrent_controller_options.recurrent_initial_iterations,
            iteration_increment=self.recurrent_controller_options.recurrent_iteration_increment,
            forward_calls_before_iteration_increment=(
                self.recurrent_controller_options.recurrent_forward_calls_before_iteration_increment
            ),
            smooth_iteration_growth_flag=(
                self.recurrent_controller_options.recurrent_smooth_iteration_growth_flag
            ),
            recurrent_layer_norm_position=(
                self.recurrent_controller_options.recurrent_layer_norm_position
            ),
            recurrent_normalization=(
                self.recurrent_controller_options.recurrent_normalization
            ),
            block_config=block_config,
            gate_config=gate_config,
            residual_config=build_residual_config(
                self.recurrent_controller_options.recurrent_residual_connection_option,
                self.recurrent_controller_options.recurrent_residual_model_flag,
                self.gate_config_factory.submodule_stack_options.residual_stack_options,
                residual_block_size=self.recurrent_controller_options.recurrent_residual_block_size,
                residual_rms_norm_epsilon=self.recurrent_controller_options.recurrent_residual_rms_norm_epsilon,
            ),
            halting_config=halting_config,
        )

    def __validate_recurrent_halting_configuration(self) -> None:
        self.__validate_minimum_steps()
        self.__validate_ponder_cost_weight()
        controller = self.recurrent_controller_options
        if (
            controller.recurrent_min_steps == 1
            and controller.recurrent_ponder_cost_weight == 1.0
        ):
            return
        configured_field = self.__configured_halting_field_name()
        if not controller.recurrent_flag:
            raise ValueError(
                f"{configured_field} requires recurrent_flag=True when configured away from its default."
            )
        if not controller.recurrent_stack_halting_flag:
            raise ValueError(
                f"{configured_field} requires recurrent_stack_halting_flag=True when configured away from its default."
            )

    def __validate_minimum_steps(self) -> None:
        controller = self.recurrent_controller_options
        minimum_steps = controller.recurrent_min_steps
        if isinstance(minimum_steps, bool) or not isinstance(minimum_steps, int):
            raise TypeError("recurrent_min_steps must be an integer.")
        if minimum_steps < 1:
            raise ValueError("recurrent_min_steps must be greater than or equal to 1.")
        maximum_steps = controller.recurrent_max_steps
        if type(maximum_steps) is int and minimum_steps > maximum_steps:
            raise ValueError(
                "recurrent_min_steps must be less than or equal to recurrent_max_steps."
            )

    def __validate_ponder_cost_weight(self) -> None:
        ponder_cost_weight = (
            self.recurrent_controller_options.recurrent_ponder_cost_weight
        )
        if isinstance(ponder_cost_weight, bool) or not isinstance(
            ponder_cost_weight, Real
        ):
            raise TypeError("recurrent_ponder_cost_weight must be a number.")
        if not math.isfinite(float(ponder_cost_weight)) or ponder_cost_weight < 0:
            raise ValueError(
                "recurrent_ponder_cost_weight must be finite and greater than or equal to 0."
            )

    def __configured_halting_field_name(self) -> str:
        if self.recurrent_controller_options.recurrent_min_steps != 1:
            return "recurrent_min_steps"
        return "recurrent_ponder_cost_weight"
