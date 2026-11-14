"""Full-gradient smooth depth handoff along one evolving recurrent state."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Generic

from emperor.layers._composition.recurrent.runtime.execution.branch import (
    RecurrentBranchRunner,
)
from emperor.layers._composition.recurrent.runtime.execution.handoff.results import (
    accumulate_caller_loss_once,
    blend_recurrent_branch_hidden,
)
from emperor.layers._composition.recurrent.runtime.execution.interface import (
    RecurrentExecutionAdapter,
    RecurrentExecutionResult,
    _StateT,
)
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentNestedSmoothHandoffExecutionPlan,
)
from emperor.layers._state import LayerState


class RecurrentNestedHandoff(Generic[_StateT]):
    """Extend the source run to target depth without replaying its shared prefix."""

    def __init__(self, branch_runner: RecurrentBranchRunner[_StateT]) -> None:
        self.__branch_runner = branch_runner

    def execute(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        layer_state: LayerState,
        execution_plan: RecurrentNestedSmoothHandoffExecutionPlan,
    ) -> RecurrentExecutionResult:
        recurrent_state = adapter._initialize_recurrent_execution_state(
            layer_state,
            branch_base_loss=None,
        )
        recurrent_state = self.__branch_runner.run(
            adapter,
            recurrent_state,
            execution_plan.source_branch,
            observe_transitions=True,
        )
        source_result = self.__branch_runner.finalize(adapter, recurrent_state)
        if recurrent_state.all_items_halted:
            return accumulate_caller_loss_once(
                adapter,
                layer_state.loss,
                source_result,
            )

        transition_index_before_target = recurrent_state.transition_index
        recurrent_state = self.__branch_runner.run(
            adapter,
            recurrent_state,
            execution_plan.target_branch,
            observe_transitions=True,
        )
        if recurrent_state.transition_index == transition_index_before_target:
            return accumulate_caller_loss_once(
                adapter,
                layer_state.loss,
                source_result,
            )

        target_result = self.__branch_runner.finalize(adapter, recurrent_state)
        transition_weight = execution_plan.transition_weight
        blended_result = RecurrentExecutionResult(
            hidden=blend_recurrent_branch_hidden(
                source_result,
                target_result,
                transition_weight,
            ),
            loss=adapter._blend_recurrent_branch_losses(
                source_result.loss,
                source_result.loss,
                target_result.loss,
                transition_weight,
            ),
            context_state=(
                source_result if transition_weight == 0 else target_result
            ).context_state,
        )
        return accumulate_caller_loss_once(
            adapter,
            layer_state.loss,
            blended_result,
        )
