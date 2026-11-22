from models import _stack_controllers
from models.gpt.expert_linear.runtime_options import (
    SubmoduleStackOptions,
    SubmoduleStackSource,
)

from ._controller_stack_config import controller_stacks


class MemoryConfigFactory(
    _stack_controllers.MemoryConfigFactory[SubmoduleStackSource, SubmoduleStackOptions]
):
    stacks = controller_stacks()
