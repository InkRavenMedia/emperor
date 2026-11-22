from emperor.config import ConfigBase
from emperor.layers import LayerStackConfig
from models import _controller_stack, _stack_controllers
from models.bert.expert_linear.runtime_options import (
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
    SubmoduleStackOptions,
    SubmoduleStackSource,
    resolve_controller_stack_options,
    resolve_experts_controller_stack_options,
)


def build_controller_stack_config(
    options: SubmoduleStackOptions | ExpertsSubmoduleStackOptions,
    *,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
    layer_model_config: ConfigBase | None = None,
) -> LayerStackConfig:
    """Build a regular or expert controller stack for this Model Package."""

    return _controller_stack.build_controller_stack_config(
        options,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        layer_model_config=layer_model_config,
    )


def expert_controller_stacks() -> _stack_controllers.ControllerStacks[
    ExpertsSubmoduleStackSource, ExpertsSubmoduleStackOptions
]:
    return _stack_controllers.ControllerStacks(
        resolve=resolve_experts_controller_stack_options,
        build=build_controller_stack_config,
    )


def controller_stacks() -> _stack_controllers.ControllerStacks[
    SubmoduleStackSource, SubmoduleStackOptions
]:
    return _stack_controllers.ControllerStacks(
        resolve=resolve_controller_stack_options,
        build=build_controller_stack_config,
    )
