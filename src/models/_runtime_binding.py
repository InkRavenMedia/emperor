import inspect
from collections.abc import Callable
from dataclasses import Field, replace
from functools import wraps
from types import ModuleType
from typing import Any, ClassVar, ParamSpec, Protocol, TypeVar

from model_runtime.packages.runtime_values import runtime_default_mapping


class _RuntimeRecord(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


_Args = ParamSpec("_Args")
_Runtime = TypeVar("_Runtime", bound=_RuntimeRecord)


def bind_runtime_options(
    config: ModuleType,
    read_updates: Callable[[dict[str, Any], ModuleType, Any], dict[str, Any]],
) -> Callable[[Callable[_Args, _Runtime]], Callable[_Args, _Runtime]]:
    def decorate(bind: Callable[_Args, _Runtime]) -> Callable[_Args, _Runtime]:
        signature = inspect.signature(bind)
        values_parameter = next(iter(signature.parameters))

        @wraps(bind)
        def with_options(*args: _Args.args, **kwargs: _Args.kwargs) -> _Runtime:
            bound = signature.bind(*args, **kwargs)
            source = bound.arguments.get("config_module", config)
            values = runtime_default_mapping(
                bound.arguments.get(values_parameter),
                package=config.__name__.rpartition(".")[0],
            )
            updates = read_updates(values, source, bound.arguments.get("base"))
            bound.arguments[values_parameter] = values
            runtime = bind(*bound.args, **bound.kwargs)
            if all(getattr(runtime, key) is value for key, value in updates.items()):
                return runtime
            return replace(runtime, **updates)

        return with_options

    return decorate
