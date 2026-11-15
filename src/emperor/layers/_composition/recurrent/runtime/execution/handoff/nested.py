from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from emperor.layers._composition.recurrent.runtime.execution.handoff.results import (
    accumulate_caller_loss_once,
    blend_recurrent_branch_hidden,
    blend_recurrent_branch_losses,
)
from emperor.layers._composition.recurrent.runtime.execution.interface import (
    RecurrentExecutionResult,
    RecurrentStateT,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from emperor.layers._composition.recurrent.runtime.execution import (
        RecurrentExecution,
    )
    from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
        RecurrentBranchExecutionPlan,
        RecurrentNestedSmoothHandoffExecutionPlan,
    )
    from emperor.layers._state import LayerState


class RecurrentNestedHandoff(Generic[RecurrentStateT]):
    def __init__(self, execution: RecurrentExecution) -> None:
        self.__execution = execution

    def execute(
        self,
        recurrent_state: RecurrentStateT,
        layer_state: LayerState,
        execution_plan: RecurrentNestedSmoothHandoffExecutionPlan,
        run_branch: Callable[
            [RecurrentStateT, RecurrentBranchExecutionPlan, bool], RecurrentStateT
        ],
    ) -> RecurrentExecutionResult:
        recurrent_state = run_branch(
            recurrent_state, execution_plan.source_branch, True
        )
        source_result = self.__execution.finalize(recurrent_state)
        if recurrent_state.all_items_halted:
            return accumulate_caller_loss_once(layer_state.loss, source_result)

        transition_index_before_target = recurrent_state.transition_index
        recurrent_state = run_branch(
            recurrent_state, execution_plan.target_branch, True
        )
        if recurrent_state.transition_index == transition_index_before_target:
            return accumulate_caller_loss_once(layer_state.loss, source_result)

        target_result = self.__execution.finalize(recurrent_state)
        transition_weight = execution_plan.transition_weight
        blended_result = RecurrentExecutionResult(
            hidden=blend_recurrent_branch_hidden(
                source_result, target_result, transition_weight
            ),
            loss=blend_recurrent_branch_losses(
                source_result.loss,
                source_result.loss,
                target_result.loss,
                transition_weight,
            ),
            context_state=(
                source_result if transition_weight == 0 else target_result
            ).context_state,
        )
        return accumulate_caller_loss_once(layer_state.loss, blended_result)
