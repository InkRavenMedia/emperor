from __future__ import annotations

from .. import _control_options
from .._control_options import (
    ControllerStackSource,
    ControlOptions,
    GateOptions,
    HaltingOptions,
    MemoryOptions,
    RecurrentOptions,
    StackOptions,
    submodule_stack_options,
)
from .runtime_options import RuntimeOptions


def _control_reader() -> _control_options.MixerControls[
    RuntimeOptions, HaltingOptions, ControlOptions
]:
    return _control_options.MixerControls(HaltingOptions, ControlOptions)


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


def token_mixer_stack_options(runtime: RuntimeOptions) -> StackOptions:
    return _control_options.token_mixer_stack_options(
        runtime, bias_flag=runtime.token_mixer_bias_flag
    )


def channel_mixer_stack_options(runtime: RuntimeOptions) -> StackOptions:
    return _control_options.channel_mixer_stack_options(
        runtime, bias_flag=runtime.channel_mixer_bias_flag
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
    "main_control_options",
    "submodule_stack_options",
    "token_mixer_control_options",
    "token_mixer_stack_options",
]
