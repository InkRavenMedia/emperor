from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Generic

from emperor.layers._composition.recurrent.runtime.execution.handoff.results import (
    accumulate_caller_loss_once,
    blend_recurrent_branch_hidden,
    blend_recurrent_branch_losses,
)
from emperor.layers._composition.recurrent.runtime.execution.interface import (
    RecurrentExecutionResult,
    RecurrentStateT,
    RecurrentTransitionCandidate,
    RecurrentTransitionResult,
)
from emperor.layers._composition.recurrent.runtime.execution.runtime_state import (
    RecurrentRuntimeStateGuard,
)
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentBranchExecutionPlan,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import Tensor, nn

    from emperor.layers._composition.recurrent.runtime.execution import (
        PreparedRecurrentTransition,
        RecurrentExecution,
    )
    from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
        RecurrentSmoothHandoffExecutionPlan,
    )
    from emperor.layers._composition.residual.base import ResidualState
    from emperor.layers._state import LayerState


class RecurrentBranchedHandoff(Generic[RecurrentStateT]):
    def __init__(self, execution: RecurrentExecution) -> None:
        self.__execution = execution
        self.__runtime_state_guard = RecurrentRuntimeStateGuard()

    def execute(
        self,
        recurrent_state: RecurrentStateT,
        layer_state: LayerState,
        execution_plan: RecurrentSmoothHandoffExecutionPlan,
        run_branch: Callable[
            [RecurrentStateT, RecurrentBranchExecutionPlan, bool], RecurrentStateT
        ],
        run_boundary: Callable[
            [RecurrentStateT, RecurrentBranchExecutionPlan],
            tuple[RecurrentExecutionResult, RecurrentStateT],
        ],
    ) -> RecurrentExecutionResult:
        common_transition_count = execution_plan.common_prefix_transition_count
        common_state = run_branch(
            recurrent_state,
            RecurrentBranchExecutionPlan(
                transition_count=common_transition_count,
                no_gradient_transition_count=common_transition_count,
            ),
            True,
        )
        source_result, target_state = run_boundary(
            common_state, execution_plan.source_branch
        )
        target_state = run_branch(target_state, execution_plan.target_branch, True)
        target_result = self.__execution.finalize(target_state)
        transition_weight = execution_plan.transition_weight
        blended_result = RecurrentExecutionResult(
            hidden=blend_recurrent_branch_hidden(
                source_result, target_result, transition_weight
            ),
            loss=blend_recurrent_branch_losses(
                common_state.branch_loss,
                source_result.loss,
                target_result.loss,
                transition_weight,
            ),
            context_state=(
                source_result if transition_weight == 0 else target_result
            ).context_state,
        )
        return accumulate_caller_loss_once(layer_state.loss, blended_result)

    def run_shared_transition(
        self,
        recurrent_state: RecurrentStateT,
        prepared_transition: PreparedRecurrentTransition,
        run_source_branch: Callable[
            [RecurrentTransitionResult], RecurrentExecutionResult
        ],
        runtime_modules: Iterable[nn.Module],
        validate_output: Callable[[object, Tensor, int], None],
        target_residual_state: ResidualState | None = None,
        transition_index: int = 0,
    ) -> tuple[RecurrentExecutionResult, RecurrentTransitionResult]:
        transition_candidate = self.__execution.compute_transition(
            recurrent_state,
            prepared_transition,
            validate_output,
            transition_index,
        )
        self.__execution.utilities.advance_detached_target_residual_state(
            transition_candidate.residual_input,
            prepared_transition.previous_evolving_hidden,
            target_residual_state,
            transition_index,
        )
        with self.__runtime_state_guard.isolate_provisional_branch(runtime_modules):
            source_transition = self.__execution.complete_transition(
                recurrent_state,
                transition_candidate,
                previous_evolving_hidden=prepared_transition.previous_evolving_hidden,
                halting_update_enabled=prepared_transition.halting_update_enabled,
                observe_transition=False,
            )
            source_result = run_source_branch(source_transition)
        target_candidate = self.__detach_transition_candidate(transition_candidate)
        target_transition = self.__execution.complete_transition(
            recurrent_state,
            target_candidate,
            previous_evolving_hidden=prepared_transition.previous_evolving_hidden,
            halting_update_enabled=False,
            observe_transition=True,
        )
        return source_result, target_transition

    def __detach_transition_candidate(
        self, transition_candidate: RecurrentTransitionCandidate
    ) -> RecurrentTransitionCandidate:
        return replace(
            transition_candidate,
            hidden=transition_candidate.hidden.detach(),
            loss=(
                None
                if transition_candidate.loss is None
                else transition_candidate.loss.detach()
            ),
            residual_input=transition_candidate.residual_input.detach(),
            context_state=transition_candidate.context_state.detach_recurrent_context(),
        )
