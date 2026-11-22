from dataclasses import dataclass
from types import ModuleType

from models._plain_expert_defaults import PlainExpertControlDefaults
from models.experts.linear.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
)


@dataclass(frozen=True, slots=True)
class ControlDefaults:
    layer: ExpertsLayerControllerOptions
    memory: ExpertsDynamicMemoryOptions
    recurrent: ExpertsRecurrentControllerOptions


class _Defaults(
    PlainExpertControlDefaults[ExpertsSubmoduleStackOptions, ControlDefaults]
):
    stack_type = ExpertsSubmoduleStackOptions
    defaults_type = ControlDefaults
    source_type = ExpertsSubmoduleStackSource
    layer_type = ExpertsLayerControllerOptions
    memory_type = ExpertsDynamicMemoryOptions
    recurrent_type = ExpertsRecurrentControllerOptions


def router_stack_defaults(config: ModuleType) -> ExpertsSubmoduleStackOptions:
    return _Defaults().router_stack_defaults(config)


def main_control_defaults(config: ModuleType) -> ControlDefaults:
    return _Defaults().main_control_defaults(config)


def expert_control_defaults(config: ModuleType) -> ControlDefaults:
    return _Defaults().expert_control_defaults(config)
