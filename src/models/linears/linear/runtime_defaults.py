from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final

from emperor.halting import HaltingConfig
from emperor.layers import LastLayerBiasOptions
from models.linears.linear._residual import ResidualStackOptions
from models.linears.linear._runtime_default_values import (
    GATE_STACK_FIELDS,
    HALTING_STACK_FIELDS,
    MEMORY_STACK_FIELDS,
    RECURRENT_GATE_STACK_FIELDS,
    RECURRENT_HALTING_STACK_FIELDS,
    RuntimeOverrideReader,
    read_controller_stack,
    read_dimensions,
    read_gate_options,
    read_halting_options,
    read_main_stack,
    read_memory_options,
    read_recurrence_options,
    read_residual_stack,
    read_submodule_stack,
)
from models.linears.linear.runtime_options import (
    ControllerStackOptions,
    LinearBuildSettings,
    MainStackOptions,
)


def build_settings_from_flat(
    overrides: Mapping[str, object] | None = None,
) -> LinearBuildSettings:
    reader = RuntimeOverrideReader({} if overrides is None else overrides)
    reader.reject_unknown(_ACCEPTED_KEYS)
    return _read_build_settings(reader)


def _read_build_settings(reader: RuntimeOverrideReader) -> LinearBuildSettings:
    dimensions = read_dimensions(reader)
    stacks = _read_stacks(reader)
    return LinearBuildSettings(
        batch_size=dimensions.batch_size,
        learning_rate=dimensions.learning_rate,
        input_dim=dimensions.input_dim,
        hidden_dim=dimensions.hidden_dim,
        output_dim=dimensions.output_dim,
        stack=stacks.main,
        submodule_stack=stacks.submodule,
        residual_stack=stacks.residual,
        gate=read_gate_options(reader, stacks.gate),
        halting=read_halting_options(reader, stacks.halting),
        memory=read_memory_options(reader, stacks.memory),
        recurrence=read_recurrence_options(
            reader, stacks.recurrent_gate, stacks.recurrent_halting
        ),
        halting_option=reader.implementation("halting_option", HaltingConfig),
        recurrent_halting_option=reader.implementation(
            "recurrent_halting_option", HaltingConfig
        ),
    )


@dataclass(frozen=True, slots=True)
class _LayerStacks:
    main: MainStackOptions
    submodule: ControllerStackOptions
    residual: ResidualStackOptions
    gate: ControllerStackOptions
    halting: ControllerStackOptions
    memory: ControllerStackOptions
    recurrent_gate: ControllerStackOptions
    recurrent_halting: ControllerStackOptions


def _read_stacks(reader: RuntimeOverrideReader) -> _LayerStacks:
    main = read_main_stack(reader)
    submodule = read_submodule_stack(reader)
    residual = read_residual_stack(reader, submodule)
    gate = read_controller_stack(reader, GATE_STACK_FIELDS, submodule)
    halting_defaults = replace(
        submodule, last_layer_bias_option=LastLayerBiasOptions.DISABLED
    )
    halting = read_controller_stack(reader, HALTING_STACK_FIELDS, halting_defaults)
    memory = read_controller_stack(reader, MEMORY_STACK_FIELDS, submodule)
    return _LayerStacks(
        main=main,
        submodule=submodule,
        residual=residual,
        gate=gate,
        halting=halting,
        memory=memory,
        recurrent_gate=read_controller_stack(reader, RECURRENT_GATE_STACK_FIELDS, gate),
        recurrent_halting=read_controller_stack(
            reader, RECURRENT_HALTING_STACK_FIELDS, halting
        ),
    )


def _package_defaults() -> tuple[LinearBuildSettings, frozenset[str]]:
    reader = RuntimeOverrideReader({})
    settings = _read_build_settings(reader)
    return settings, frozenset(reader.accepted)


DEFAULT_BUILD_SETTINGS, _ACCEPTED_KEYS = _package_defaults()

runtime_from_flat = build_settings_from_flat
DEFAULT_RUNTIME: Final = DEFAULT_BUILD_SETTINGS
