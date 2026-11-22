from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Final, Generic, Protocol, TypeVar

from models._controller_stack import ControllerStackSettings
from models._stack_controllers import (
    GateConstructor,
    HaltingConstructor,
    MemoryConstructor,
    RecurrentConstructor,
)


class _LayerControls(Protocol):
    stack_gate_flag: Final[bool]
    stack_halting_flag: Final[bool]


class _RecurrentControls(Protocol):
    recurrent_flag: Final[bool]


class _MemoryControls(Protocol):
    memory_flag: Final[bool]


class _ModelStack(Protocol):
    num_layers: Final[int]


class ProjectionPaths(Protocol):
    hidden_dim: Final[int]
    attention_projection_stack_options: Final[ControllerStackSettings]
    attention_projection_layer_controller_options: Final[_LayerControls | None]
    attention_projection_recurrent_controller_options: Final[_RecurrentControls | None]
    attention_projection_dynamic_memory_options: Final[_MemoryControls | None]
    feed_forward_stack_options: Final[ControllerStackSettings]
    feed_forward_layer_controller_options: Final[_LayerControls | None]
    feed_forward_recurrent_controller_options: Final[_RecurrentControls | None]
    feed_forward_dynamic_memory_options: Final[_MemoryControls | None]


class EncoderPaths(ProjectionPaths, Protocol):
    encoder_stack_options: Final[_ModelStack]
    encoder_submodule_stack_options: Final[ControllerStackSettings | None]
    encoder_layer_controller_options: Final[_LayerControls | None]
    encoder_recurrent_controller_options: Final[_RecurrentControls | None]
    encoder_dynamic_memory_options: Final[_MemoryControls | None]


class DecoderPaths(ProjectionPaths, Protocol):
    decoder_stack_options: Final[_ModelStack]
    decoder_submodule_stack_options: Final[ControllerStackSettings | None]
    decoder_layer_controller_options: Final[_LayerControls | None]
    decoder_recurrent_controller_options: Final[_RecurrentControls | None]
    decoder_dynamic_memory_options: Final[_MemoryControls | None]


class _ProjectionFactory(Protocol):
    def build_attention_projection_factory(self) -> object: ...
    def build_feed_forward_factory(self) -> object: ...


class _EncoderFactory(_ProjectionFactory, Protocol):
    def build_encoder_factory(self) -> object: ...


class _DecoderFactory(_ProjectionFactory, Protocol):
    def build_decoder_factory(self) -> object: ...


_Paths = TypeVar("_Paths", bound=ProjectionPaths)
_Encoder = TypeVar("_Encoder", bound=EncoderPaths)
_Decoder = TypeVar("_Decoder", bound=DecoderPaths)
_Factory = TypeVar("_Factory")
_Gate = TypeVar("_Gate", bound=_ProjectionFactory)
_Halting = TypeVar("_Halting", bound=_ProjectionFactory)
_EncoderGate = TypeVar("_EncoderGate", bound=_EncoderFactory)
_EncoderHalting = TypeVar("_EncoderHalting", bound=_EncoderFactory)
_DecoderGate = TypeVar("_DecoderGate", bound=_DecoderFactory)
_DecoderHalting = TypeVar("_DecoderHalting", bound=_DecoderFactory)


class _PathFactory(Generic[_Paths, _Factory]):
    def __init__(self, dependencies: _Paths) -> None:
        self.dependencies = dependencies


class _GatePaths(_PathFactory[_Paths, _Factory]):
    _factory: GateConstructor[_Factory]

    def build_attention_projection_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if (
            dependencies.attention_projection_layer_controller_options is None
            or dependencies.attention_projection_recurrent_controller_options is None
        ):
            return None
        return self._factory(
            layer_controller_options=dependencies.attention_projection_layer_controller_options,
            recurrent_controller_options=dependencies.attention_projection_recurrent_controller_options,
            submodule_stack_options=dependencies.attention_projection_stack_options,
            recurrent_stack_inherits_gate_stack=False,
        )

    def build_feed_forward_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if (
            dependencies.feed_forward_layer_controller_options is None
            or dependencies.feed_forward_recurrent_controller_options is None
        ):
            return None
        return self._factory(
            layer_controller_options=dependencies.feed_forward_layer_controller_options,
            recurrent_controller_options=dependencies.feed_forward_recurrent_controller_options,
            submodule_stack_options=dependencies.feed_forward_stack_options,
            recurrent_stack_inherits_gate_stack=False,
        )


class EncoderGateFactories(_GatePaths[_Encoder, _Factory]):
    def build_encoder_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if (
            dependencies.encoder_layer_controller_options is None
            or dependencies.encoder_recurrent_controller_options is None
            or dependencies.encoder_submodule_stack_options is None
        ):
            return None
        return self._factory(
            layer_controller_options=dependencies.encoder_layer_controller_options,
            recurrent_controller_options=dependencies.encoder_recurrent_controller_options,
            submodule_stack_options=dependencies.encoder_submodule_stack_options,
        )


class DecoderGateFactories(_GatePaths[_Decoder, _Factory]):
    def build_decoder_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if (
            dependencies.decoder_layer_controller_options is None
            or dependencies.decoder_recurrent_controller_options is None
            or dependencies.decoder_submodule_stack_options is None
        ):
            return None
        return self._factory(
            layer_controller_options=dependencies.decoder_layer_controller_options,
            recurrent_controller_options=dependencies.decoder_recurrent_controller_options,
            submodule_stack_options=dependencies.decoder_submodule_stack_options,
        )


class _HaltingPaths(_PathFactory[_Paths, _Factory]):
    _factory: HaltingConstructor[_Factory]

    def build_attention_projection_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if (
            dependencies.attention_projection_layer_controller_options is None
            or dependencies.attention_projection_recurrent_controller_options is None
        ):
            return None
        stack_options = dependencies.attention_projection_stack_options
        return self._factory(
            layer_controller_options=dependencies.attention_projection_layer_controller_options,
            recurrent_controller_options=dependencies.attention_projection_recurrent_controller_options,
            submodule_stack_options=stack_options,
            output_dim=dependencies.hidden_dim,
            halting_stack_defaults=stack_options,
            recurrent_stack_inherits_halting_stack=False,
        )

    def build_feed_forward_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if (
            dependencies.feed_forward_layer_controller_options is None
            or dependencies.feed_forward_recurrent_controller_options is None
        ):
            return None
        stack_options = dependencies.feed_forward_stack_options
        return self._factory(
            layer_controller_options=dependencies.feed_forward_layer_controller_options,
            recurrent_controller_options=dependencies.feed_forward_recurrent_controller_options,
            submodule_stack_options=stack_options,
            output_dim=dependencies.hidden_dim,
            halting_stack_defaults=stack_options,
            recurrent_stack_inherits_halting_stack=False,
        )


class EncoderHaltingFactories(_HaltingPaths[_Encoder, _Factory]):
    def build_encoder_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if (
            dependencies.encoder_layer_controller_options is None
            or dependencies.encoder_recurrent_controller_options is None
            or dependencies.encoder_submodule_stack_options is None
        ):
            return None
        return self._factory(
            layer_controller_options=dependencies.encoder_layer_controller_options,
            recurrent_controller_options=dependencies.encoder_recurrent_controller_options,
            submodule_stack_options=dependencies.encoder_submodule_stack_options,
            output_dim=dependencies.hidden_dim,
        )


class DecoderHaltingFactories(_HaltingPaths[_Decoder, _Factory]):
    def build_decoder_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if (
            dependencies.decoder_layer_controller_options is None
            or dependencies.decoder_recurrent_controller_options is None
            or dependencies.decoder_submodule_stack_options is None
        ):
            return None
        return self._factory(
            layer_controller_options=dependencies.decoder_layer_controller_options,
            recurrent_controller_options=dependencies.decoder_recurrent_controller_options,
            submodule_stack_options=dependencies.decoder_submodule_stack_options,
            output_dim=dependencies.hidden_dim,
        )


class _MemoryPaths(_PathFactory[_Paths, _Factory], ABC):
    _factory: MemoryConstructor[_Factory]

    @property
    @abstractmethod
    def _main_stack(self) -> _ModelStack: ...

    def build_attention_projection_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if dependencies.attention_projection_dynamic_memory_options is None:
            return None
        return self._factory(
            hidden_dim=dependencies.hidden_dim,
            stack_options=self._main_stack,
            dynamic_memory_options=dependencies.attention_projection_dynamic_memory_options,
            submodule_stack_options=dependencies.attention_projection_stack_options,
        )

    def build_feed_forward_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if dependencies.feed_forward_dynamic_memory_options is None:
            return None
        return self._factory(
            hidden_dim=dependencies.hidden_dim,
            stack_options=self._main_stack,
            dynamic_memory_options=dependencies.feed_forward_dynamic_memory_options,
            submodule_stack_options=dependencies.feed_forward_stack_options,
        )


class EncoderMemoryFactories(_MemoryPaths[_Encoder, _Factory]):
    @property
    def _main_stack(self) -> _ModelStack:
        return self.dependencies.encoder_stack_options

    def build_encoder_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if (
            dependencies.encoder_dynamic_memory_options is None
            or dependencies.encoder_submodule_stack_options is None
        ):
            return None
        return self._factory(
            hidden_dim=dependencies.hidden_dim,
            stack_options=dependencies.encoder_stack_options,
            dynamic_memory_options=dependencies.encoder_dynamic_memory_options,
            submodule_stack_options=dependencies.encoder_submodule_stack_options,
        )


class DecoderMemoryFactories(_MemoryPaths[_Decoder, _Factory]):
    @property
    def _main_stack(self) -> _ModelStack:
        return self.dependencies.decoder_stack_options

    def build_decoder_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if (
            dependencies.decoder_dynamic_memory_options is None
            or dependencies.decoder_submodule_stack_options is None
        ):
            return None
        return self._factory(
            hidden_dim=dependencies.hidden_dim,
            stack_options=dependencies.decoder_stack_options,
            dynamic_memory_options=dependencies.decoder_dynamic_memory_options,
            submodule_stack_options=dependencies.decoder_submodule_stack_options,
        )


class _RecurrentPaths(
    _PathFactory[_Paths, _Factory], Generic[_Paths, _Factory, _Gate, _Halting]
):
    _factory: RecurrentConstructor[_Factory]

    _gate_factory: Callable[[_Paths], _Gate]
    _halting_factory: Callable[[_Paths], _Halting]

    def __init__(self, dependencies: _Paths) -> None:
        super().__init__(dependencies)
        self.gate_factory = type(self)._gate_factory(dependencies)
        self.halting_factory = type(self)._halting_factory(dependencies)

    def build_attention_projection_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if dependencies.attention_projection_recurrent_controller_options is None:
            return None
        gate_factory = self.gate_factory.build_attention_projection_factory()
        halting_factory = self.halting_factory.build_attention_projection_factory()
        if gate_factory is None or halting_factory is None:
            return None
        return self._factory(
            recurrent_controller_options=dependencies.attention_projection_recurrent_controller_options,
            gate_config_factory=gate_factory,
            halting_config_factory=halting_factory,
        )

    def build_feed_forward_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if dependencies.feed_forward_recurrent_controller_options is None:
            return None
        gate_factory = self.gate_factory.build_feed_forward_factory()
        halting_factory = self.halting_factory.build_feed_forward_factory()
        if gate_factory is None or halting_factory is None:
            return None
        return self._factory(
            recurrent_controller_options=dependencies.feed_forward_recurrent_controller_options,
            gate_config_factory=gate_factory,
            halting_config_factory=halting_factory,
        )


class EncoderRecurrentFactories(
    _RecurrentPaths[_Encoder, _Factory, _EncoderGate, _EncoderHalting]
):
    def build_encoder_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if dependencies.encoder_recurrent_controller_options is None:
            return None
        gate_factory = self.gate_factory.build_encoder_factory()
        halting_factory = self.halting_factory.build_encoder_factory()
        if gate_factory is None or halting_factory is None:
            return None
        return self._factory(
            recurrent_controller_options=dependencies.encoder_recurrent_controller_options,
            gate_config_factory=gate_factory,
            halting_config_factory=halting_factory,
        )


class DecoderRecurrentFactories(
    _RecurrentPaths[_Decoder, _Factory, _DecoderGate, _DecoderHalting]
):
    def build_decoder_factory(self) -> _Factory | None:
        dependencies = self.dependencies
        if dependencies.decoder_recurrent_controller_options is None:
            return None
        gate_factory = self.gate_factory.build_decoder_factory()
        halting_factory = self.halting_factory.build_decoder_factory()
        if gate_factory is None or halting_factory is None:
            return None
        return self._factory(
            recurrent_controller_options=dependencies.decoder_recurrent_controller_options,
            gate_config_factory=gate_factory,
            halting_config_factory=halting_factory,
        )
