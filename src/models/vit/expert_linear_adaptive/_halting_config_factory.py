from models import _stack_controllers
from models.vit.expert_linear_adaptive.runtime_options import (
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    SubmoduleStackSource,
)

from ._controller_stack_config import controller_stacks


class HaltingConfigFactory(
    _stack_controllers.HaltingConfigFactory[SubmoduleStackSource, SubmoduleStackOptions]
):
    stacks = controller_stacks()

    recurrent_controller_options: RecurrentControllerOptions

    def _minimum_recurrent_steps(self) -> int:
        return self.recurrent_controller_options.recurrent_min_steps

    def _recurrent_ponder_cost_weight(self) -> float:
        return self.recurrent_controller_options.recurrent_ponder_cost_weight
