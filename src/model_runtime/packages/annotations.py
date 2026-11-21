"""Resolve Runtime Defaults annotations shared by validation, parsing and Inspection."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping
from functools import cache
from types import MappingProxyType, ModuleType, NoneType, UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints


@cache
def resolved_annotations(module: ModuleType) -> Mapping[str, Any]:
    return MappingProxyType(get_type_hints(module, include_extras=True))


class ModuleAnnotations(Mapping[str, Any]):
    """Read annotations on demand so unknown keys retain validation precedence."""

    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def __repr__(self) -> str:
        return f"ModuleAnnotations({self._module.__name__!r})"

    def __getitem__(self, key: str) -> Any:
        return resolved_annotations(self._module)[key]

    def __iter__(self) -> Iterator[str]:
        return iter(resolved_annotations(self._module))

    def __len__(self) -> int:
        return len(resolved_annotations(self._module))


def transparent_annotation(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def annotation_options(annotation: Any) -> tuple[Any, ...]:
    annotation = transparent_annotation(annotation)
    if get_origin(annotation) in {UnionType, Union}:
        return tuple(
            item for arg in get_args(annotation) for item in annotation_options(arg)
        )
    return (annotation,)


def annotation_classes(annotation: Any) -> list[type[Any]]:
    classes: list[type[Any]] = []
    for option in annotation_options(annotation):
        if option is NoneType:
            continue
        arguments = get_args(option)
        if get_origin(option) is type and arguments:
            option = transparent_annotation(arguments[0])
        if inspect.isclass(option):
            classes.append(option)
    return classes


def annotation_is_nullable(annotation: Any) -> bool:
    return NoneType in annotation_options(annotation)
