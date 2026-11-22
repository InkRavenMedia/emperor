from __future__ import annotations

from dataclasses import dataclass, replace

from .. import _control_options
from .._control_options import (
    ControllerStackSource,
    GateOptions,
    MemoryOptions,
    StackOptions,
    submodule_stack_options,
)
from .runtime_options import RuntimeOptions


@dataclass(frozen=True, slots=True)
class HaltingOptions(_control_options.HaltingOptions):
    min_steps: int = 1
    ponder_cost_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class RecurrentOptions(_control_options.RecurrentOptions):
    halting: HaltingOptions


@dataclass(frozen=True, slots=True)
class ControlOptions(_control_options.ControlOptions):
    halting: HaltingOptions
    recurrent: RecurrentOptions


class _Controls(
    _control_options.ExpertMixerControls[RuntimeOptions, HaltingOptions, ControlOptions]
):
    def main_recurrent_halting(self, runtime: RuntimeOptions) -> HaltingOptions:
        halting = super().main_recurrent_halting(runtime)
        return replace(
            halting,
            min_steps=runtime.recurrent_min_steps,
            ponder_cost_weight=runtime.recurrent_ponder_cost_weight,
        )


def _control_reader() -> _Controls:
    return _Controls(HaltingOptions, ControlOptions, RecurrentOptions)


def token_mixer_control_options(
    runtime: RuntimeOptions,
) -> ControlOptions:
    controls = _control_reader()
    return controls.token_mixer_control_options(runtime)


def channel_mixer_control_options(
    runtime: RuntimeOptions,
) -> ControlOptions:
    controls = _control_reader()
    return controls.channel_mixer_control_options(runtime)


def main_control_options(runtime: RuntimeOptions) -> ControlOptions:
    controls = _control_reader()
    return controls.main_control_options(runtime)


def expert_control_options(runtime: RuntimeOptions) -> ControlOptions:
    controls = _control_reader()
    return controls.expert_control_options(runtime)


def token_mixer_stack_options(runtime: RuntimeOptions) -> StackOptions:
    return _control_options.token_mixer_stack_options(
        runtime, bias_flag=runtime.stack_bias_flag
    )


def channel_mixer_stack_options(runtime: RuntimeOptions) -> StackOptions:
    return _control_options.channel_mixer_stack_options(
        runtime, bias_flag=runtime.stack_bias_flag
    )


__all__ = [
    "ControlOptions",
    "ControllerStackSource",
    "GateOptions",
    "HaltingOptions",
    "MemoryOptions",
    "RecurrentOptions",
    "StackOptions",
    "channel_mixer_control_options",
    "channel_mixer_stack_options",
    "expert_control_options",
    "main_control_options",
    "submodule_stack_options",
    "token_mixer_control_options",
    "token_mixer_stack_options",
]
