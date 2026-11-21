from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_type_hints

from models.catalog import discover_model_packages


def _allows_none(annotation: object) -> bool:
    return annotation in (Any, object) or type(None) in get_args(annotation)


def _none_annotation_mismatches(root: object) -> list[str]:
    mismatches: list[str] = []
    pending = [("runtime", root)]
    visited: set[int] = set()
    while pending:
        path, value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        if isinstance(value, Mapping):
            pending.extend((f"{path}[{key!r}]", item) for key, item in value.items())
        elif isinstance(value, (tuple, list)):
            pending.extend((f"{path}[{i}]", item) for i, item in enumerate(value))
        if not is_dataclass(value) or isinstance(value, type):
            continue
        annotations = get_type_hints(type(value))
        for dataclass_field in fields(value):
            field_value = getattr(value, dataclass_field.name)
            field_path = f"{path}.{dataclass_field.name}"
            annotation = annotations[dataclass_field.name]
            if field_value is None and not _allows_none(annotation):
                mismatches.append(f"{field_path}: {annotation}")
            else:
                pending.append((field_path, field_value))
    return mismatches


def test_runtime_option_annotations_accept_every_default_none_value() -> None:
    mismatches: list[str] = []
    for package in discover_model_packages():
        for mismatch in _none_annotation_mismatches(package.bind_runtime_defaults()):
            mismatches.append(f"{package.catalog_key}: {mismatch}")

    assert not mismatches, "Runtime annotation mismatches:\n" + "\n".join(mismatches)


def test_optional_generator_stacks_resolve_to_the_package_source_record() -> None:
    from importlib import import_module

    packages = (
        "bert.linear_adaptive",
        "bert.expert_linear_adaptive",
        "gpt.linear_adaptive",
        "gpt.expert_linear_adaptive",
        "vit.linear_adaptive",
        "vit.expert_linear_adaptive",
        "experts.linear_adaptive",
        "neuron.expert_linear_adaptive",
    )
    for package in packages:
        generation = import_module(f"models.{package}._generation")
        source = generation.AdaptiveGeneratorStackSource
        for options in (
            generation.WeightGenerationOptions,
            generation.BiasGenerationOptions,
        ):
            assert (
                get_type_hints(options)["mixture_router_generator_stack"]
                == source | None
            )
