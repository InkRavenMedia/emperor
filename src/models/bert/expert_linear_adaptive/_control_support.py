from models import _stack_controllers
from models._recurrent_block import (
    ExpertResidualRecurrentFactory,
)
from models.bert.expert_linear_adaptive.runtime_options import (
    ExpertsRecurrentControllerOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
)

from ._controller_stack_config import (
    expert_controller_stacks,
)


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

    recurrent_controller_options: ExpertsRecurrentControllerOptions

    def _minimum_recurrent_steps(self) -> int:
        return self.recurrent_controller_options.recurrent_min_steps

    def _recurrent_ponder_cost_weight(self) -> float:
        return self.recurrent_controller_options.recurrent_ponder_cost_weight


class ExpertsMemoryConfigFactory(
    _stack_controllers.SlottedExpertMemoryConfigFactory[
        ExpertsSubmoduleStackSource, ExpertsSubmoduleStackOptions
    ]
):
    stacks = expert_controller_stacks()


class ExpertsRecurrentConfigFactory(
    ExpertResidualRecurrentFactory[
        ExpertsRecurrentControllerOptions,
        ExpertsGateConfigFactory,
        ExpertsHaltingConfigFactory,
    ]
):
    pass
