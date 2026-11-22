from dataclasses import dataclass

import models.vit.expert_linear.config as config
from emperor.layers import LayerStackConfig
from models._expert_mixture import (
    ExpertSettings,
    ExpertStackFactory,
)
from models.vit.expert_linear import _config_defaults as config_defaults
from models.vit.expert_linear._controller_stack_config import (
    build_controller_stack_config,
)
from models.vit.expert_linear._expert_control_support import (
    ExpertsGateConfigFactory,
    ExpertsHaltingConfigFactory,
    ExpertsMemoryConfigFactory,
    ExpertsRecurrentConfigFactory,
)
from models.vit.expert_linear.runtime_options import (
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


class ControlConfigFactory(ExpertStackFactory):
    _gate_type = ExpertsGateConfigFactory
    _halting_type = ExpertsHaltingConfigFactory
    _memory_type = ExpertsMemoryConfigFactory
    _recurrent_type = ExpertsRecurrentConfigFactory

    def __init__(self, dependencies: ControlConfigDependencies) -> None:
        self.stack_options = config_defaults.resolve_options(
            dependencies.stack_options,
            config_defaults.experts_stack_options,
            config,
            hidden_dim=dependencies.hidden_dim,
        )
        self.submodule_stack_options = config_defaults.resolve_options(
            dependencies.submodule_stack_options,
            config_defaults.experts_submodule_stack_options,
            config,
            config_defaults.ExpertStackRole.MAIN,
        )
        self.mixture_options = config_defaults.resolve_options(
            dependencies.mixture_options,
            config_defaults.experts_mixture_options,
            config,
        )
        self.expert_stack_options = config_defaults.resolve_options(
            dependencies.expert_stack_options,
            config_defaults.experts_expert_stack_options,
            config,
            self.submodule_stack_options,
        )
        self.sampler_options = config_defaults.resolve_options(
            dependencies.sampler_options,
            config_defaults.experts_sampler_options,
            config,
        )
        self.router_options = config_defaults.resolve_options(
            dependencies.router_options, config_defaults.experts_router_options, config
        )
        self.router_stack_options = config_defaults.resolve_options(
            dependencies.router_stack_options,
            config_defaults.experts_submodule_stack_options,
            config,
            config_defaults.ExpertStackRole.ROUTER,
        )
        self.layer_controller_options = config_defaults.resolve_options(
            dependencies.layer_controller_options,
            config_defaults.experts_layer_controller_options,
            config,
            config_defaults.ExpertControlRole.MAIN,
        )
        self.dynamic_memory_options = config_defaults.resolve_options(
            dependencies.dynamic_memory_options,
            config_defaults.experts_dynamic_memory_options,
            config,
            config_defaults.ExpertControlRole.MAIN,
        )
        self.recurrent_controller_options = config_defaults.resolve_options(
            dependencies.recurrent_controller_options,
            config_defaults.experts_recurrent_controller_options,
            config,
            config_defaults.ExpertControlRole.MAIN,
        )
        self.expert_layer_controller_options = config_defaults.resolve_options(
            dependencies.expert_layer_controller_options,
            config_defaults.experts_layer_controller_options,
            config,
            config_defaults.ExpertControlRole.EXPERT,
        )
        self.expert_dynamic_memory_options = config_defaults.resolve_options(
            dependencies.expert_dynamic_memory_options,
            config_defaults.experts_dynamic_memory_options,
            config,
            config_defaults.ExpertControlRole.EXPERT,
        )
        self.expert_recurrent_controller_options = config_defaults.resolve_options(
            dependencies.expert_recurrent_controller_options,
            config_defaults.experts_recurrent_controller_options,
            config,
            config_defaults.ExpertControlRole.EXPERT,
        )
        self.hidden_dim = self.stack_options.hidden_dim
        self.output_dim = dependencies.output_dim
        self._initialize_controllers()

    @property
    def _expert_settings(self) -> ExpertSettings:
        return self

    def _build_router_model(self) -> LayerStackConfig:
        return build_controller_stack_config(self.router_stack_options)
