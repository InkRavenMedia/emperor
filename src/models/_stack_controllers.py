from collections.abc import Callable
from dataclasses import Field, dataclass, replace
from typing import Any, ClassVar, Generic, Protocol, TypeVar

from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerStackConfig,
)
from emperor.memory import (
    AttentionDynamicMemoryConfig,
    DynamicMemoryConfig,
    MemoryPositionOptions,
)


class _Width(Protocol):
    @property
    def hidden_dim(self) -> int | None: ...


class _Stack(_Width, Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def last_layer_bias_option(self) -> LastLayerBiasOptions: ...


_SourceT = TypeVar("_SourceT")
_Source_co = TypeVar("_Source_co", covariant=True)
_StackT = TypeVar("_StackT")
_HaltingStackT = TypeVar("_HaltingStackT", bound=_Stack)
_Stack_contra = TypeVar("_Stack_contra", contravariant=True)


class _BuildStack(Protocol[_Stack_contra]):
    def __call__(
        self,
        options: _Stack_contra,
        /,
        *,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
    ) -> LayerStackConfig: ...


@dataclass(frozen=True)
class ControllerStacks(Generic[_SourceT, _StackT]):
    resolve: Callable[[_SourceT, _StackT], _StackT]
    build: _BuildStack[_StackT]


class _LayerGate(Protocol[_Source_co]):
    @property
    def stack_gate_flag(self) -> bool: ...

    @property
    def gate_stack_source(self) -> _Source_co: ...

    @property
    def gate_option(self) -> LayerGateOptions | None: ...

    @property
    def gate_activation(self) -> ActivationOptions | None: ...


class _LayerHalting(Protocol[_Source_co]):
    @property
    def stack_halting_flag(self) -> bool: ...

    @property
    def halting_stack_source(self) -> _Source_co: ...

    @property
    def halting_option(self) -> type[HaltingConfig]: ...

    @property
    def halting_threshold(self) -> float | None: ...

    @property
    def halting_dropout(self) -> float: ...

    @property
    def halting_hidden_state_mode(self) -> HaltingHiddenStateModeOptions: ...


class _RecurrentGate(Protocol[_Source_co]):
    @property
    def recurrent_stack_gate_flag(self) -> bool: ...

    @property
    def recurrent_gate_stack_source(self) -> _Source_co: ...

    @property
    def recurrent_gate_option(self) -> LayerGateOptions | None: ...

    @property
    def recurrent_gate_activation(self) -> ActivationOptions | None: ...


class _RecurrentHalting(Protocol[_Source_co]):
    @property
    def recurrent_stack_halting_flag(self) -> bool: ...

    @property
    def recurrent_halting_stack_source(self) -> _Source_co: ...

    @property
    def recurrent_halting_option(self) -> type[HaltingConfig]: ...

    @property
    def recurrent_halting_threshold(self) -> float | None: ...

    @property
    def recurrent_halting_dropout(self) -> float: ...

    @property
    def recurrent_halting_hidden_state_mode(self) -> HaltingHiddenStateModeOptions: ...


class _Memory(Protocol[_Source_co]):
    @property
    def memory_flag(self) -> bool: ...

    @property
    def memory_stack_source(self) -> _Source_co: ...

    @property
    def memory_option(self) -> type[DynamicMemoryConfig]: ...

    @property
    def memory_position_option(self) -> MemoryPositionOptions: ...

    @property
    def memory_test_time_training_learning_rate(self) -> float | None: ...

    @property
    def memory_test_time_training_num_inner_steps(self) -> int | None: ...


class _SlottedMemory(_Memory[_Source_co], Protocol[_Source_co]):
    @property
    def memory_num_memory_slots(self) -> int: ...


class _ExpertLayerHalting(_LayerHalting[_Source_co], Protocol[_Source_co]):
    @property
    def halting_output_dim(self) -> int: ...


class GateConfigFactory(Generic[_SourceT, _StackT]):
    stacks: ControllerStacks[_SourceT, _StackT]

    def __init__(
        self,
        *,
        layer_controller_options: _LayerGate[_SourceT],
        recurrent_controller_options: _RecurrentGate[_SourceT],
        submodule_stack_options: _StackT,
        recurrent_stack_inherits_gate_stack: bool = True,
    ) -> None:
        self.layer_controller_options = layer_controller_options
        self.recurrent_controller_options = recurrent_controller_options
        self.submodule_stack_options = submodule_stack_options
        self.recurrent_stack_inherits_gate_stack = recurrent_stack_inherits_gate_stack

    def build_gate_config(self) -> GateConfig | None:
        layer = self.layer_controller_options
        if not layer.stack_gate_flag:
            return None
        options = self.stacks.resolve(
            layer.gate_stack_source, self.submodule_stack_options
        )
        model_config = self.stacks.build(options)
        return GateConfig(
            model_config=model_config,
            option=layer.gate_option,
            activation=layer.gate_activation,
        )

    def build_recurrent_gate_config(self) -> GateConfig | None:
        recurrent = self.recurrent_controller_options
        if not recurrent.recurrent_stack_gate_flag:
            return None
        defaults = self.submodule_stack_options
        if self.recurrent_stack_inherits_gate_stack:
            defaults = self.stacks.resolve(
                self.layer_controller_options.gate_stack_source, defaults
            )
        options = self.stacks.resolve(recurrent.recurrent_gate_stack_source, defaults)
        model_config = self.stacks.build(options)
        return GateConfig(
            model_config=model_config,
            option=recurrent.recurrent_gate_option,
            activation=recurrent.recurrent_gate_activation,
        )


class HaltingConfigFactory(Generic[_SourceT, _HaltingStackT]):
    stacks: ControllerStacks[_SourceT, _HaltingStackT]
    _STICK_BREAKING_GATE_OUTPUT_DIM = 2

    def __init__(
        self,
        *,
        layer_controller_options: _LayerHalting[_SourceT],
        recurrent_controller_options: _RecurrentHalting[_SourceT],
        submodule_stack_options: _HaltingStackT,
        output_dim: int,
        halting_stack_defaults: _HaltingStackT | None = None,
        recurrent_stack_inherits_halting_stack: bool = True,
    ) -> None:
        self.layer_controller_options = layer_controller_options
        self.recurrent_controller_options = recurrent_controller_options
        self.submodule_stack_options = submodule_stack_options
        self.output_dim = output_dim
        self.halting_stack_defaults = halting_stack_defaults
        self.recurrent_stack_inherits_halting_stack = (
            recurrent_stack_inherits_halting_stack
        )

    def build_halting_config(self) -> HaltingConfig | None:
        layer = self.layer_controller_options
        if not layer.stack_halting_flag:
            return None
        defaults = self.__stack_defaults()
        options = self.stacks.resolve(layer.halting_stack_source, defaults)
        gate_config = self.__build_gate(options)
        return layer.halting_option(
            threshold=layer.halting_threshold,
            min_steps=1,
            ponder_cost_weight=1.0,
            dropout_probability=layer.halting_dropout,
            hidden_state_mode=layer.halting_hidden_state_mode,
            halting_gate_config=gate_config,
        )

    def build_recurrent_halting_config(self) -> HaltingConfig | None:
        recurrent = self.recurrent_controller_options
        if not recurrent.recurrent_stack_halting_flag:
            return None
        defaults = self.__stack_defaults()
        if self.recurrent_stack_inherits_halting_stack:
            defaults = self.stacks.resolve(
                self.layer_controller_options.halting_stack_source, defaults
            )
        options = self.stacks.resolve(
            recurrent.recurrent_halting_stack_source, defaults
        )
        gate_config = self.__build_gate(options)
        return recurrent.recurrent_halting_option(
            threshold=recurrent.recurrent_halting_threshold,
            min_steps=self._minimum_recurrent_steps(),
            ponder_cost_weight=self._recurrent_ponder_cost_weight(),
            dropout_probability=recurrent.recurrent_halting_dropout,
            hidden_state_mode=recurrent.recurrent_halting_hidden_state_mode,
            halting_gate_config=gate_config,
        )

    def __stack_defaults(self) -> _HaltingStackT:
        if self.halting_stack_defaults is not None:
            return self.halting_stack_defaults
        return replace(
            self.submodule_stack_options,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
        )

    def __build_gate(self, options: _HaltingStackT) -> LayerStackConfig:
        return self.stacks.build(
            options,
            hidden_dim=options.hidden_dim or self.output_dim,
            output_dim=self._gate_output_dim(),
        )

    def _gate_output_dim(self) -> int:
        return self._STICK_BREAKING_GATE_OUTPUT_DIM

    def _minimum_recurrent_steps(self) -> int:
        return 1

    def _recurrent_ponder_cost_weight(self) -> float:
        return 1.0


class ExpertHaltingConfigFactory(HaltingConfigFactory[_SourceT, _HaltingStackT]):
    def __init__(
        self,
        *,
        layer_controller_options: _ExpertLayerHalting[_SourceT],
        recurrent_controller_options: _RecurrentHalting[_SourceT],
        submodule_stack_options: _HaltingStackT,
        output_dim: int,
        halting_stack_defaults: _HaltingStackT | None = None,
        recurrent_stack_inherits_halting_stack: bool = True,
    ) -> None:
        super().__init__(
            layer_controller_options=layer_controller_options,
            recurrent_controller_options=recurrent_controller_options,
            submodule_stack_options=submodule_stack_options,
            output_dim=output_dim,
            halting_stack_defaults=halting_stack_defaults,
            recurrent_stack_inherits_halting_stack=recurrent_stack_inherits_halting_stack,
        )
        self.layer_controller_options = layer_controller_options

    def _gate_output_dim(self) -> int:
        return self.layer_controller_options.halting_output_dim


class _MemoryFactory(Generic[_SourceT, _StackT]):
    stacks: ControllerStacks[_SourceT, _StackT]
    submodule_stack_options: _StackT

    def _build_memory_config(
        self, memory: _Memory[_SourceT]
    ) -> DynamicMemoryConfig | None:
        if not memory.memory_flag:
            return None
        options = self.stacks.resolve(
            memory.memory_stack_source, self.submodule_stack_options
        )
        model_config = self.stacks.build(options)
        width = self._memory_width()
        memory_config = memory.memory_option(
            input_dim=width,
            output_dim=width,
            memory_position_option=memory.memory_position_option,
            test_time_training_learning_rate=memory.memory_test_time_training_learning_rate,
            test_time_training_num_inner_steps=memory.memory_test_time_training_num_inner_steps,
            model_config=model_config,
        )
        if isinstance(memory_config, AttentionDynamicMemoryConfig):
            self._apply_slot_count(memory_config)
        return memory_config

    def _memory_width(self) -> int | None:
        raise NotImplementedError

    def _apply_slot_count(self, memory: AttentionDynamicMemoryConfig) -> None:
        pass


class MemoryConfigFactory(_MemoryFactory[_SourceT, _StackT]):
    def __init__(
        self,
        *,
        hidden_dim: int,
        stack_options: object,
        dynamic_memory_options: _SlottedMemory[_SourceT],
        submodule_stack_options: _StackT,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.stack_options = stack_options
        self.dynamic_memory_options = dynamic_memory_options
        self.submodule_stack_options = submodule_stack_options

    def build_memory_config(self) -> DynamicMemoryConfig | None:
        return self._build_memory_config(self.dynamic_memory_options)

    def _memory_width(self) -> int:
        return self.hidden_dim

    def _apply_slot_count(self, memory: AttentionDynamicMemoryConfig) -> None:
        memory.num_memory_slots = self.dynamic_memory_options.memory_num_memory_slots


class ExpertMemoryConfigFactory(_MemoryFactory[_SourceT, _StackT]):
    def __init__(
        self,
        *,
        stack_options: _Width,
        dynamic_memory_options: _Memory[_SourceT],
        submodule_stack_options: _StackT,
    ) -> None:
        self.stack_options = stack_options
        self.dynamic_memory_options = dynamic_memory_options
        self.submodule_stack_options = submodule_stack_options

    def build_memory_config(self) -> DynamicMemoryConfig | None:
        return self._build_memory_config(self.dynamic_memory_options)

    def _memory_width(self) -> int | None:
        return self.stack_options.hidden_dim


class SlottedExpertMemoryConfigFactory(ExpertMemoryConfigFactory[_SourceT, _StackT]):
    def __init__(
        self,
        *,
        stack_options: _Width,
        dynamic_memory_options: _SlottedMemory[_SourceT],
        submodule_stack_options: _StackT,
    ) -> None:
        super().__init__(
            stack_options=stack_options,
            dynamic_memory_options=dynamic_memory_options,
            submodule_stack_options=submodule_stack_options,
        )
        self.dynamic_memory_options = dynamic_memory_options

    def _apply_slot_count(self, memory: AttentionDynamicMemoryConfig) -> None:
        memory.num_memory_slots = self.dynamic_memory_options.memory_num_memory_slots


_Built = TypeVar("_Built", covariant=True)


class GateConstructor(Protocol[_Built]):
    def __call__(
        self,
        *,
        layer_controller_options: Any,
        recurrent_controller_options: Any,
        submodule_stack_options: Any,
        recurrent_stack_inherits_gate_stack: bool = True,
    ) -> _Built: ...


class HaltingConstructor(Protocol[_Built]):
    def __call__(
        self,
        *,
        layer_controller_options: Any,
        recurrent_controller_options: Any,
        submodule_stack_options: Any,
        output_dim: int,
        halting_stack_defaults: Any = None,
        recurrent_stack_inherits_halting_stack: bool = True,
    ) -> _Built: ...


class MemoryConstructor(Protocol[_Built]):
    def __call__(
        self,
        *,
        hidden_dim: int,
        stack_options: Any,
        dynamic_memory_options: Any,
        submodule_stack_options: Any,
    ) -> _Built: ...


class RecurrentConstructor(Protocol[_Built]):
    def __call__(
        self,
        *,
        recurrent_controller_options: Any,
        gate_config_factory: Any,
        halting_config_factory: Any,
    ) -> _Built: ...


class ExpertMemoryConstructor(Protocol[_Built]):
    def __call__(
        self,
        *,
        stack_options: Any,
        dynamic_memory_options: Any,
        submodule_stack_options: Any,
    ) -> _Built: ...
