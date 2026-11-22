"""Resolve Runtime Defaults annotations shared by validation, parsing and Inspection."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType, ModuleType, NoneType, UnionType
from typing import Annotated, Any, Union, cast, get_args, get_origin, get_type_hints


def snapshot_section_path(path: object) -> tuple[str, ...]:
    if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
        raise ValueError("Runtime Defaults metadata section path must be a sequence.")
    snapshot = tuple(cast(Sequence[object], path))
    if any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in snapshot
    ):
        raise ValueError(
            "Runtime Defaults metadata section path items must be non-empty "
            "trimmed strings."
        )
    return cast(tuple[str, ...], snapshot)


@dataclass(frozen=True, slots=True)
class Section:
    path: tuple[str, ...]

    def __post_init__(self) -> None:
        path = snapshot_section_path(self.path)
        if not path:
            raise ValueError("A configuration section needs a non-empty path.")
        object.__setattr__(self, "path", path)

    def child(self, title: str) -> Section:
        return Section((*self.path, title))


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
        return without_inspection_section(resolved_annotations(self._module)[key])

    def __iter__(self) -> Iterator[str]:
        return iter(resolved_annotations(self._module))

    def __len__(self) -> int:
        return len(resolved_annotations(self._module))


def without_inspection_section(annotation: Any) -> Any:
    if get_origin(annotation) is not Annotated:
        return annotation
    value_type, *metadata = get_args(annotation)
    remaining = tuple(item for item in metadata if not isinstance(item, Section))
    if len(remaining) == len(metadata):
        return annotation
    if remaining:
        return Annotated[value_type, *remaining]
    return value_type


def transparent_annotation(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def inherit_alias_annotations(
    annotations: dict[str, Any], aliases: Mapping[str, str]
) -> None:
    for key, owner in aliases.items():
        if key not in annotations and owner in annotations:
            annotations[key] = without_inspection_section(annotations[owner])


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
