from dataclasses import dataclass

import models.bert.expert_linear.config as config
from models._configured_experts import LanguageExpertFactory
from models.bert.expert_linear import _config_defaults as config_defaults
from models.bert.expert_linear._expert_control_support import (
    ExpertsGateConfigFactory,
    ExpertsHaltingConfigFactory,
    ExpertsMemoryConfigFactory,
    ExpertsRecurrentConfigFactory,
)
from models.bert.expert_linear.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
)


@dataclass(frozen=True)
class ControlConfigDependencies:
    hidden_dim: int
    stack_options: ExpertsStackOptions | None
    submodule_stack_options: ExpertsSubmoduleStackOptions | None
    mixture_options: ExpertsMixtureOptions | None
    expert_stack_options: ExpertsSubmoduleStackOptions | None
    sampler_options: ExpertsSamplerOptions | None
    router_options: ExpertsRouterOptions | None
    router_stack_options: ExpertsSubmoduleStackOptions | None
    layer_controller_options: ExpertsLayerControllerOptions | None
    dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_layer_controller_options: ExpertsLayerControllerOptions | None
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    output_dim: int


class ControlConfigFactory(LanguageExpertFactory[ControlConfigDependencies]):
    _config = config
    _defaults = config_defaults
    _gate_type = ExpertsGateConfigFactory
    _halting_type = ExpertsHaltingConfigFactory
    _memory_type = ExpertsMemoryConfigFactory
    _recurrent_type = ExpertsRecurrentConfigFactory
