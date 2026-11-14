"""Shared recurrent branch loop, gradient handling, and finalization."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import replace
from typing import Generic

from emperor.layers._composition.recurrent.runtime.execution.interface import (
    RecurrentExecutionAdapter,
    RecurrentExecutionResult,
    _StateT,
)
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentBranchExecutionPlan,
)


class RecurrentBranchRunner(Generic[_StateT]):
    """Continue a branch to its planned total transitions or until all items halt."""

    def run(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        recurrent_state: _StateT,
        branch_plan: RecurrentBranchExecutionPlan,
        observe_transitions: bool,
    ) -> _StateT:
        while (
            recurrent_state.transition_index < branch_plan.transition_count
            and not recurrent_state.all_items_halted
        ):
            recurrent_state = self.__run_planned_transition(
                adapter,
                recurrent_state,
                branch_plan,
                observe_transition=observe_transitions,
            )
        return recurrent_state

    def __run_planned_transition(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        recurrent_state: _StateT,
        branch_plan: RecurrentBranchExecutionPlan,
        observe_transition: bool,
    ) -> _StateT:
        recurrent_state = self.detach_at_gradient_boundary(
            adapter,
            recurrent_state,
            branch_plan,
        )
        transition_index = recurrent_state.transition_index
        tracks_gradients = branch_plan.tracks_gradients(transition_index)
        with branch_plan.gradient_context(transition_index):
            prepared_transition = adapter._prepare_recurrent_transition(
                recurrent_state,
                tracks_gradients=tracks_gradients,
            )
            transition_result = adapter._run_recurrent_transition(
                recurrent_state,
                prepared_transition,
                transition_index=transition_index,
                observe_transition=observe_transition,
            )
            return adapter._apply_recurrent_transition_result(
                recurrent_state,
                transition_result,
            )

    def detach_at_gradient_boundary(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        recurrent_state: _StateT,
        branch_plan: RecurrentBranchExecutionPlan,
    ) -> _StateT:
        if not branch_plan.starts_gradient_suffix(recurrent_state.transition_index):
            return recurrent_state
        return adapter._detach_recurrent_execution_state(recurrent_state)

    def finalize(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        recurrent_state: _StateT,
    ) -> RecurrentExecutionResult:
        finalized_hidden, finalized_loss = adapter._finalize_recurrent_halting(
            recurrent_state.output_hidden,
            adapter._recurrent_branch_loss(recurrent_state),
            recurrent_state.halting_state,
        )
        return RecurrentExecutionResult(
            finalized_hidden,
            finalized_loss,
            replace(recurrent_state.context_state, hidden=finalized_hidden),
        )
