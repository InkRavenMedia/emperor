from emperor.config import ConfigBase
from emperor.layers import (
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from models import _controller_stack, _recurrent_block, _stack_controllers
from models.experts.linear_adaptive.runtime_options import (
    ExpertsAdaptiveGeneratorStackOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
    resolve_experts_controller_stack_options,
)


def build_linear_controller_stack(
    options: ExpertsSubmoduleStackOptions,
    *,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    linear_config = LinearLayerConfig(bias_flag=options.bias_flag)
    linear_controller_stack_config = build_controller_stack(
        options,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        layer_model_config=linear_config,
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
