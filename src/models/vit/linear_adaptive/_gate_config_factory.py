from models import _stack_controllers
from models.vit.linear_adaptive.runtime_options import (
    SubmoduleStackOptions,
    SubmoduleStackSource,
)

from ._controller_stack_config import controller_stacks


class GateConfigFactory(
    _stack_controllers.GateConfigFactory[SubmoduleStackSource, SubmoduleStackOptions]
):
    stacks = controller_stacks()
