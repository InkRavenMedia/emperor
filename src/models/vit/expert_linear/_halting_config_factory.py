from models import _stack_controllers
from models.vit.expert_linear.runtime_options import (
    SubmoduleStackOptions,
    SubmoduleStackSource,
)

from ._controller_stack_config import controller_stacks


class HaltingConfigFactory(
    _stack_controllers.HaltingConfigFactory[SubmoduleStackSource, SubmoduleStackOptions]
):
    stacks = controller_stacks()
