import math
from dataclasses import dataclass
from numbers import Real
from typing import Generic, Protocol, TypeVar

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig
from models._residual_stack import ResidualStackSettings, build_residual_config

_BlockT = TypeVar("_BlockT", bound=ConfigBase)


class _Schedule(Protocol):
    @property
    def max_steps(self) -> int: ...

    @property
    def gradient_transition_count(self) -> int | None: ...

    @property
    def no_gradient_transition_count(self) -> int | None: ...

    @property
    def initial_iterations(self) -> int | None: ...

    @property
    def iteration_increment(self) -> int: ...

    @property
    def forward_calls_before_iteration_increment(self) -> int: ...


class _ControllerSchedule(Protocol):
    @property
    def recurrent_max_steps(self) -> int: ...

    @property
    def recurrent_gradient_transition_count(self) -> int | None: ...

    @property
    def recurrent_no_gradient_transition_count(self) -> int | None: ...

    @property
    def recurrent_initial_iterations(self) -> int: ...

    @property
    def recurrent_iteration_increment(self) -> int: ...

    @property
    def recurrent_forward_calls_before_iteration_increment(self) -> int: ...


class _ControllerOptions(_ControllerSchedule, Protocol):
    @property
    def recurrent_flag(self) -> bool: ...

    @property
    def recurrent_layer_norm_position(self) -> LayerNormPositionOptions: ...

    @property
    def recurrent_normalization(self) -> NormalizationOptions: ...


class _GrowthOptions(_ControllerOptions, Protocol):
    @property
    def recurrent_smooth_iteration_growth_flag(self) -> bool: ...


_ControllerT = TypeVar("_ControllerT", bound=_ControllerOptions)


class _RecurrentGateFactory(Protocol):
    def build_recurrent_gate_config(self) -> GateConfig | None: ...


class _RecurrentHaltingFactory(Protocol):
    def build_recurrent_halting_config(self) -> HaltingConfig | None: ...


@dataclass(frozen=True)
class RecurrentSchedule:
    max_steps: int
    gradient_transition_count: int | None
    no_gradient_transition_count: int | None
    initial_iterations: int | None
    iteration_increment: int
    forward_calls_before_iteration_increment: int

    @classmethod
    def from_options(cls, options: _Schedule) -> "RecurrentSchedule":
        return cls(
            max_steps=options.max_steps,
            gradient_transition_count=options.gradient_transition_count,
            no_gradient_transition_count=options.no_gradient_transition_count,
            initial_iterations=options.initial_iterations,
            iteration_increment=options.iteration_increment,
            forward_calls_before_iteration_increment=options.forward_calls_before_iteration_increment,
        )

    @classmethod
    def from_controller(cls, options: _ControllerSchedule) -> "RecurrentSchedule":
        return cls(
            max_steps=options.recurrent_max_steps,
            gradient_transition_count=options.recurrent_gradient_transition_count,
            no_gradient_transition_count=options.recurrent_no_gradient_transition_count,
            initial_iterations=options.recurrent_initial_iterations,
            iteration_increment=options.recurrent_iteration_increment,
            forward_calls_before_iteration_increment=options.recurrent_forward_calls_before_iteration_increment,
        )


def build_recurrent_block(
    schedule: _Schedule,
    block_config: ConfigBase,
    *,
    layer_norm_position: LayerNormPositionOptions,
    normalization: NormalizationOptions,
    smooth_iteration_growth_flag: bool | None = None,
    input_dim: int | None = None,
    output_dim: int | None = None,
    gate_config: GateConfig | None = None,
    halting_config: HaltingConfig | None = None,
    residual_config: ResidualConfig | None = None,
    memory_config: DynamicMemoryConfig | None = None,
) -> RecurrentLayerConfig:
    return RecurrentLayerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        max_steps=schedule.max_steps,
        gradient_transition_count=schedule.gradient_transition_count,
        no_gradient_transition_count=schedule.no_gradient_transition_count,
        initial_iterations=schedule.initial_iterations,
        iteration_increment=schedule.iteration_increment,
        forward_calls_before_iteration_increment=schedule.forward_calls_before_iteration_increment,
        smooth_iteration_growth_flag=smooth_iteration_growth_flag,
        recurrent_layer_norm_position=layer_norm_position,
        recurrent_normalization=normalization,
        block_config=block_config,
        gate_config=gate_config,
        halting_config=halting_config,
        residual_config=residual_config,
        memory_config=memory_config,
    )


class RecurrentBlockFactory(Generic[_ControllerT]):
    def __init__(
        self,
        *,
        recurrent_controller_options: _ControllerT,
        gate_config_factory: _RecurrentGateFactory,
        halting_config_factory: _RecurrentHaltingFactory,
    ) -> None:
        self.recurrent_controller_options = recurrent_controller_options
        self.gate_config_factory = gate_config_factory
        self.halting_config_factory = halting_config_factory

    def build_config(self, block_config: _BlockT) -> _BlockT | RecurrentLayerConfig:
        return self._wrap(block_config)

    def _smooth_growth_flag(self) -> bool | None:
        return None

    def _wrap(
        self,
        block_config: _BlockT,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ) -> _BlockT | RecurrentLayerConfig:
        if not self.recurrent_controller_options.recurrent_flag:
            return block_config
        gate_config = self.gate_config_factory.build_recurrent_gate_config()
        halting_config = self.halting_config_factory.build_recurrent_halting_config()
        controller = self.recurrent_controller_options
        schedule = RecurrentSchedule.from_controller(controller)
        return build_recurrent_block(
            schedule,
            block_config,
            input_dim=input_dim,
            output_dim=output_dim,
            smooth_iteration_growth_flag=self._smooth_growth_flag(),
            layer_norm_position=controller.recurrent_layer_norm_position,
            normalization=controller.recurrent_normalization,
            gate_config=gate_config,
            halting_config=halting_config,
        )


class ScheduledRecurrentBlockFactory(RecurrentBlockFactory[_GrowthOptions]):
    def _smooth_growth_flag(self) -> bool:
        return self.recurrent_controller_options.recurrent_smooth_iteration_growth_flag


class RecurrentConfigFactory(ScheduledRecurrentBlockFactory):
    def build_config(
        self,
        block_config: _BlockT,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ) -> _BlockT | RecurrentLayerConfig:
        return self._wrap(block_config, input_dim=input_dim, output_dim=output_dim)


class _ResidualHaltingOptions(_GrowthOptions, Protocol):
    @property
    def recurrent_stack_halting_flag(self) -> bool: ...
    @property
    def recurrent_min_steps(self) -> object: ...
    @property
    def recurrent_ponder_cost_weight(self) -> float: ...
    @property
    def recurrent_residual_connection_option(self) -> type[ResidualConfig] | None: ...
    @property
    def recurrent_residual_model_flag(self) -> bool: ...
    @property
    def recurrent_residual_block_size(self) -> int | None: ...
    @property
    def recurrent_residual_rms_norm_epsilon(self) -> float | None: ...


class _ResidualSource(Protocol):
    @property
    def residual_stack_options(self) -> ResidualStackSettings | None: ...


class _ExpertResidualOptions(_ResidualSource, _ResidualHaltingOptions, Protocol):
    pass


class _ProjectionGate(_RecurrentGateFactory, Protocol):
    @property
    def submodule_stack_options(self) -> _ResidualSource: ...


_ResidualOptions = TypeVar("_ResidualOptions", bound=_ResidualHaltingOptions)
_ExpertOptions = TypeVar("_ExpertOptions", bound=_ExpertResidualOptions)
_Gate = TypeVar("_Gate", bound=_RecurrentGateFactory)
_ProjectedGate = TypeVar("_ProjectedGate", bound=_ProjectionGate)
_Halting = TypeVar("_Halting", bound=_RecurrentHaltingFactory)


class _HaltingRecurrentFactory(Generic[_ResidualOptions, _Gate, _Halting]):
    def __init__(
        self,
        *,
        recurrent_controller_options: _ResidualOptions,
        gate_config_factory: _Gate,
        halting_config_factory: _Halting,
    ) -> None:
        self.recurrent_controller_options = recurrent_controller_options
        self.gate_config_factory = gate_config_factory
        self.halting_config_factory = halting_config_factory

    def _validate_halting(self) -> None:
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


class ProjectedResidualRecurrentFactory(
    _HaltingRecurrentFactory[_ResidualOptions, _ProjectedGate, _Halting]
):
    def build_config(
        self,
        block_config: _BlockT,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ) -> _BlockT | RecurrentLayerConfig:
        self._validate_halting()
        if not self.recurrent_controller_options.recurrent_flag:
            return block_config
        gate_config = self.gate_config_factory.build_recurrent_gate_config()
        halting_config = self.halting_config_factory.build_recurrent_halting_config()
        residual_config = build_residual_config(
            self.recurrent_controller_options.recurrent_residual_connection_option,
            self.recurrent_controller_options.recurrent_residual_model_flag,
            self.gate_config_factory.submodule_stack_options.residual_stack_options,
            residual_block_size=self.recurrent_controller_options.recurrent_residual_block_size,
            residual_rms_norm_epsilon=self.recurrent_controller_options.recurrent_residual_rms_norm_epsilon,
        )
        controller = self.recurrent_controller_options
        schedule = RecurrentSchedule.from_controller(controller)
        return build_recurrent_block(
            schedule,
            block_config,
            input_dim=input_dim,
            output_dim=output_dim,
            smooth_iteration_growth_flag=controller.recurrent_smooth_iteration_growth_flag,
            layer_norm_position=controller.recurrent_layer_norm_position,
            normalization=controller.recurrent_normalization,
            gate_config=gate_config,
            residual_config=residual_config,
            halting_config=halting_config,
        )


class ExpertResidualRecurrentFactory(
    _HaltingRecurrentFactory[_ExpertOptions, _Gate, _Halting]
):
    def build_config(self, block_config: _BlockT) -> _BlockT | RecurrentLayerConfig:
        self._validate_halting()
        if not self.recurrent_controller_options.recurrent_flag:
            return block_config
        gate_config = self.gate_config_factory.build_recurrent_gate_config()
        residual_config = build_residual_config(
            self.recurrent_controller_options.recurrent_residual_connection_option,
            self.recurrent_controller_options.recurrent_residual_model_flag,
            self.recurrent_controller_options.residual_stack_options,
            residual_block_size=self.recurrent_controller_options.recurrent_residual_block_size,
            residual_rms_norm_epsilon=self.recurrent_controller_options.recurrent_residual_rms_norm_epsilon,
        )
        halting_config = self.halting_config_factory.build_recurrent_halting_config()
        controller = self.recurrent_controller_options
        schedule = RecurrentSchedule.from_controller(controller)
        return build_recurrent_block(
            schedule,
            block_config,
            smooth_iteration_growth_flag=controller.recurrent_smooth_iteration_growth_flag,
            layer_norm_position=controller.recurrent_layer_norm_position,
            normalization=controller.recurrent_normalization,
            gate_config=gate_config,
            residual_config=residual_config,
            halting_config=halting_config,
        )
