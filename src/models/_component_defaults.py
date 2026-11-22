"""Register flat controls for optional components declared by a package."""

from collections.abc import Callable
from typing import ParamSpec, TypeVar

from models._learned_grouping import declare_grouping_model_defaults
from models._projection_model import declare_projection_model_defaults
from models._sequence_expansion_controls import declare_sequence_expansion_defaults


def declare_component_defaults(namespace: dict[str, object]) -> None:
    declare_projection_model_defaults(namespace)
    declare_grouping_model_defaults(namespace)
    declare_sequence_expansion_defaults(namespace)


_OptionsT = TypeVar("_OptionsT")
_DefaultArgs = ParamSpec("_DefaultArgs")


def resolve_options(
    provided: _OptionsT | None,
    default_factory: Callable[_DefaultArgs, _OptionsT],
    /,
    *args: _DefaultArgs.args,
    **kwargs: _DefaultArgs.kwargs,
) -> _OptionsT:
    if provided is None:
        return default_factory(*args, **kwargs)
    return provided
