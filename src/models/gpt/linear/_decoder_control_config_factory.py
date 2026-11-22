from models import _stack_controllers
from models._recurrent_block import RecurrentConfigFactory as RecurrentConfigFactory
from models.gpt.linear.runtime_options import (
    SubmoduleStackOptions,
    SubmoduleStackSource,
)

from ._controller_stack_config import controller_stacks


class GateConfigFactory(
    _stack_controllers.GateConfigFactory[SubmoduleStackSource, SubmoduleStackOptions]
):
    stacks = controller_stacks()


class HaltingConfigFactory(
    _stack_controllers.HaltingConfigFactory[SubmoduleStackSource, SubmoduleStackOptions]
):
    stacks = controller_stacks()


class MemoryConfigFactory(
    _stack_controllers.MemoryConfigFactory[SubmoduleStackSource, SubmoduleStackOptions]
):
    stacks = controller_stacks()
