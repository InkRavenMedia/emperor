from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, Protocol, TypeVar

_Value = TypeVar("_Value")
_Role = TypeVar("_Role")


class _StackReader(Protocol[_Value]):
    def __call__(
        self,
        *,
        option_key: str,
        provided: _Value | None,
        defaults_factory: Callable[[], _Value],
        flat_prefix: str,
    ) -> _Value | None: ...


class _LayerReader(Protocol[_Value]):
    def __call__(
        self,
        *,
        option_key: str,
        provided: _Value | None,
        defaults_factory: Callable[[], _Value],
        flat_prefix: str,
        gate_stack_prefix: str,
        halting_stack_prefix: str,
    ) -> _Value | None: ...


class _MemoryReader(Protocol[_Value]):
    def __call__(
        self,
        *,
        option_key: str,
        provided: _Value | None,
        defaults_factory: Callable[[], _Value],
        flat_prefix: str,
        memory_stack_prefix: str,
    ) -> _Value | None: ...


_Stack = TypeVar("_Stack")

_Layer = TypeVar("_Layer")

_Memory = TypeVar("_Memory")

_Recurrent = TypeVar("_Recurrent")


@dataclass(frozen=True, slots=True)
class ControlOverrides(Generic[_Role, _Stack, _Layer, _Memory, _Recurrent]):
    supplied: dict[str, Any]
    resolved: dict[str, Any]
    stack: _StackReader[_Stack]
    layer: _LayerReader[_Layer]
    memory: _MemoryReader[_Memory]
    recurrent: _LayerReader[_Recurrent]
    default_stack: Callable[[_Role], _Stack]
    default_layer: Callable[[_Role], _Layer]
    default_memory: Callable[[_Role], _Memory]
    default_recurrent: Callable[[_Role], _Recurrent]

    def __apply(
        self, role: _Role, *, group: str, prefix: str, stack_key: str, stack_prefix: str
    ) -> None:
        stack = self.stack(
            option_key=stack_key,
            provided=self.supplied.get(stack_key),
            defaults_factory=partial(self.default_stack, role),
            flat_prefix=stack_prefix,
        )
        self.__store(stack_key, stack)

        layer_key = f"{group}layer_controller_options"
        layer = self.layer(
            option_key=layer_key,
            provided=self.supplied.get(layer_key),
            defaults_factory=partial(self.default_layer, role),
            flat_prefix=prefix.removesuffix("_"),
            gate_stack_prefix=f"{prefix}gate_stack",
            halting_stack_prefix=f"{prefix}halting_stack",
        )
        self.__store(layer_key, layer)

        memory_key = f"{group}dynamic_memory_options"
        memory = self.memory(
            option_key=memory_key,
            provided=self.supplied.get(memory_key),
            defaults_factory=partial(self.default_memory, role),
            flat_prefix=prefix.removesuffix("_"),
            memory_stack_prefix=f"{prefix}memory_stack",
        )
        self.__store(memory_key, memory)

        recurrent_key = f"{group}recurrent_controller_options"
        recurrent = self.recurrent(
            option_key=recurrent_key,
            provided=self.supplied.get(recurrent_key),
            defaults_factory=partial(self.default_recurrent, role),
            flat_prefix=f"{prefix}recurrent",
            gate_stack_prefix=f"{prefix}recurrent_gate_stack",
            halting_stack_prefix=f"{prefix}recurrent_halting_stack",
        )
        self.__store(recurrent_key, recurrent)

    def apply_roles(
        self, *, main: _Role, attention: _Role, feed_forward: _Role
    ) -> None:
        self.__apply(
            main,
            group="",
            prefix="",
            stack_key="submodule_stack_options",
            stack_prefix="submodule_stack",
        )
        self.__apply(
            attention,
            group="attention_projection_",
            prefix="attn_",
            stack_key="attention_projection_stack_options",
            stack_prefix="attn_stack",
        )
        self.__apply(
            feed_forward,
            group="feed_forward_",
            prefix="ff_",
            stack_key="feed_forward_stack_options",
            stack_prefix="ff_stack",
        )

    def __store(self, key: str, value: object) -> None:
        if value is not None:
            self.resolved[key] = value
