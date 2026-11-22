from models import _recurrent_block, _stack_controllers
from models.vit.expert_linear_adaptive_cluster.runtime_options import (
    ExpertsRecurrentControllerOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
)

from ._controller_stack_config import expert_controller_stacks


class ExpertsRecurrentConfigFactory(
    _recurrent_block.RecurrentBlockFactory[ExpertsRecurrentControllerOptions]
):
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
    _stack_controllers.SlottedExpertMemoryConfigFactory[
        ExpertsSubmoduleStackSource, ExpertsSubmoduleStackOptions
    ]
):
    stacks = expert_controller_stacks()
