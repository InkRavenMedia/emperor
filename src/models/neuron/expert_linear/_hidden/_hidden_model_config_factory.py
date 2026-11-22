from emperor.config import ConfigBase
from emperor.layers import (
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from models import _controller_stack, _recurrent_block, _stack_controllers
from models._expert_mixture import (
    ExpertSettings,
    ExpertStackFactory,
)
from models.neuron.expert_linear._hidden.runtime_options import (
    ExpertsAdaptiveGeneratorStackOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
    RuntimeOptions,
    resolve_experts_controller_stack_options,
)


def build_linear_controller_stack(
    options: ExpertsSubmoduleStackOptions,
    *,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    linear_layer_config = LinearLayerConfig(bias_flag=options.bias_flag)
    linear_controller_stack_config = build_controller_stack(
        options,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        layer_model_config=linear_layer_config,
    )
    return linear_controller_stack_config


def build_controller_stack(
    options: ExpertsSubmoduleStackOptions | ExpertsAdaptiveGeneratorStackOptions,
    *,
    layer_model_config: ConfigBase,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    return _controller_stack.build_controller_stack_with_model(
        options,
        layer_model_config=layer_model_config,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )


def expert_controller_stacks() -> _stack_controllers.ControllerStacks[
    ExpertsSubmoduleStackSource, ExpertsSubmoduleStackOptions
]:
    return _stack_controllers.ControllerStacks(
        resolve=resolve_experts_controller_stack_options,
        build=build_linear_controller_stack,
    )


class ExpertsRecurrentConfigFactory(_recurrent_block.ScheduledRecurrentBlockFactory):
    pass


class ExpertsGateConfigFactory(
    _stack_controllers.GateConfigFactory[
        ExpertsSubmoduleStackSource, ExpertsSubmoduleStackOptions
    ]
):
    stacks = expert_controller_stacks()


class ExpertsHaltingConfigFactory(
    _stack_controllers.ExpertHaltingConfigFactory[
        ExpertsSubmoduleStackSource, ExpertsSubmoduleStackOptions
    ]
):
    stacks = expert_controller_stacks()


class ExpertsMemoryConfigFactory(
    _stack_controllers.ExpertMemoryConfigFactory[
        ExpertsSubmoduleStackSource, ExpertsSubmoduleStackOptions
    ]
):
    stacks = expert_controller_stacks()


class HiddenModelConfigFactory(ExpertStackFactory):
    _gate_type = ExpertsGateConfigFactory
    _halting_type = ExpertsHaltingConfigFactory
    _memory_type = ExpertsMemoryConfigFactory
    _recurrent_type = ExpertsRecurrentConfigFactory

    def __init__(self, runtime: RuntimeOptions) -> None:
        self.stack_options = runtime.stack_options
        self.submodule_stack_options = runtime.submodule_stack_options
        self.mixture_options = runtime.mixture_options
        self.expert_stack_options = runtime.expert_stack_options
        self.sampler_options = runtime.sampler_options
        self.router_options = runtime.router_options
        self.router_stack_options = runtime.router_stack_options
        self.layer_controller_options = runtime.layer_controller_options
        self.dynamic_memory_options = runtime.dynamic_memory_options
        self.recurrent_controller_options = runtime.recurrent_controller_options
        self.expert_layer_controller_options = runtime.expert_layer_controller_options
        self.expert_dynamic_memory_options = runtime.expert_dynamic_memory_options
        self.expert_recurrent_controller_options = (
            runtime.expert_recurrent_controller_options
        )
        self.hidden_dim = self.stack_options.hidden_dim
        self.output_dim = runtime.output_dim
        self._initialize_controllers()

    @property
    def _expert_settings(self) -> ExpertSettings:
        return self

    def _build_router_model(self) -> LayerStackConfig:
        return build_linear_controller_stack(self.router_stack_options)
