from emperor.layers import LayerStackConfig
from models import _controller_stack, _stack_controllers
from models.gpt.linear_adaptive.runtime_options import (
    SubmoduleStackOptions,
    SubmoduleStackSource,
    resolve_controller_stack_options,
)


def build_controller_stack_config(
    options: SubmoduleStackOptions,
    *,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    """Build the package-local stack used by controller models."""

    return _controller_stack.build_controller_stack_config(
        options, hidden_dim=hidden_dim, output_dim=output_dim
    )


def controller_stacks() -> _stack_controllers.ControllerStacks[
    SubmoduleStackSource, SubmoduleStackOptions
]:
    return _stack_controllers.ControllerStacks(
        resolve=resolve_controller_stack_options,
        build=build_controller_stack_config,
    )
