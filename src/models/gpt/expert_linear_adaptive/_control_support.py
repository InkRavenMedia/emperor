import math
from numbers import Real
from typing import cast

from models import _stack_controllers
from models._recurrent_block import (
    ExpertResidualRecurrentFactory,
)
from models.gpt.expert_linear_adaptive.runtime_options import (
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
    def _validate_halting(self) -> None:
        controller = self.recurrent_controller_options
        min_steps = cast(object, controller.recurrent_min_steps)
        if isinstance(min_steps, bool) or not isinstance(min_steps, int):
            raise TypeError(
                "recurrent_min_steps must be an integer, "
                f"received {type(min_steps).__name__}."
            )
        if min_steps < 1:
            raise ValueError(
                "recurrent_min_steps must be greater than or equal to 1, "
                f"received {min_steps}."
            )
        max_steps = cast(object, controller.recurrent_max_steps)
        if (
            isinstance(max_steps, int)
            and not isinstance(max_steps, bool)
            and min_steps > max_steps
        ):
            raise ValueError(
                "recurrent_min_steps must be less than or equal to "
                f"recurrent_max_steps, received {min_steps} and {max_steps}."
            )

        ponder_cost_weight = controller.recurrent_ponder_cost_weight
        if isinstance(ponder_cost_weight, bool) or not isinstance(
            ponder_cost_weight,
            Real,
        ):
            raise TypeError(
                "recurrent_ponder_cost_weight must be a number, "
                f"received {type(ponder_cost_weight).__name__}."
            )
        if not math.isfinite(float(ponder_cost_weight)) or ponder_cost_weight < 0:
            raise ValueError(
                "recurrent_ponder_cost_weight must be finite and greater than or "
                f"equal to 0, received {ponder_cost_weight}."
            )

        minimum_is_default = min_steps == 1
        ponder_weight_is_default = ponder_cost_weight == 1.0
        if minimum_is_default and ponder_weight_is_default:
            return
        if not controller.recurrent_flag:
            configured_field = "recurrent_ponder_cost_weight"
            if not minimum_is_default:
                configured_field = "recurrent_min_steps"
            raise ValueError(
                f"{configured_field} requires recurrent_flag=True when configured "
                "away from its default."
            )
        if not controller.recurrent_stack_halting_flag:
            configured_field = "recurrent_ponder_cost_weight"
            if not minimum_is_default:
                configured_field = "recurrent_min_steps"
            raise ValueError(
                f"{configured_field} requires recurrent_stack_halting_flag=True "
                "when configured away from its default."
            )
