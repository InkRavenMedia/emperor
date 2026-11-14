"""Smooth depth handoff with provisional source and detached target branches."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Generic

import torch

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
    RecurrentBranchExecutionPlan,
    RecurrentSmoothHandoffExecutionPlan,
)
from emperor.layers._state import LayerState

if TYPE_CHECKING:
    from emperor.layers._composition.recurrent.base import RecurrentTransitionResult


class RecurrentBranchedHandoff(Generic[_StateT]):
    """Run the shared prefix, provisional source, and observed target in order."""

    def __init__(self, branch_runner: RecurrentBranchRunner[_StateT]) -> None:
        self.__branch_runner = branch_runner

    def execute(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        layer_state: LayerState,
        execution_plan: RecurrentSmoothHandoffExecutionPlan,
    ) -> RecurrentExecutionResult:
        common_state = adapter._initialize_recurrent_execution_state(
            layer_state,
            branch_base_loss=None,
        )
        common_transition_count = execution_plan.common_prefix_transition_count
        common_state = self.__branch_runner.run(
            adapter,
            common_state,
            RecurrentBranchExecutionPlan(
                transition_count=common_transition_count,
                no_gradient_transition_count=common_transition_count,
            ),
            observe_transitions=True,
        )

        source_result, target_state = (
            self.__advance_handoff_branches_through_shared_boundary(
                adapter,
                common_state,
                execution_plan.source_branch,
            )
        )

        target_state = self.__branch_runner.run(
            adapter,
            target_state,
            execution_plan.target_branch,
            observe_transitions=True,
        )
        target_result = self.__branch_runner.finalize(adapter, target_state)
        transition_weight = execution_plan.transition_weight
        blended_hidden = blend_recurrent_branch_hidden(
            source_result,
            target_result,
            transition_weight,
        )
        blended_recurrent_loss = adapter._blend_recurrent_branch_losses(
            adapter._recurrent_branch_loss(common_state),
            source_result.loss,
            target_result.loss,
            transition_weight,
        )
        context_state = (
            source_result if transition_weight == 0 else target_result
        ).context_state
        return accumulate_caller_loss_once(
            adapter,
            layer_state.loss,
            RecurrentExecutionResult(
                hidden=blended_hidden,
                loss=blended_recurrent_loss,
                context_state=context_state,
            ),
        )

    def __advance_handoff_branches_through_shared_boundary(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        common_state: _StateT,
        source_plan: RecurrentBranchExecutionPlan,
    ) -> tuple[RecurrentExecutionResult, _StateT]:
        source_boundary_state = self.__branch_runner.detach_at_gradient_boundary(
            adapter,
            common_state,
            source_plan,
        )
        transition_index = source_boundary_state.transition_index
        source_tracks_gradients = source_plan.tracks_gradients(transition_index)
        target_state = None
        target_residual_state = None
        with source_plan.gradient_context(transition_index):
            prepared_transition = adapter._prepare_recurrent_transition(
                source_boundary_state,
                tracks_gradients=source_tracks_gradients,
            )
            if prepared_transition.residual_state is not None:
                target_state = adapter._fork_recurrent_handoff_state(
                    source_boundary_state
                )
                with torch.no_grad():
                    target_prepared_transition = adapter._prepare_recurrent_transition(
                        target_state,
                        tracks_gradients=False,
                    )
                target_residual_state = target_prepared_transition.residual_state
            provisional_source_continuation = partial(
                self.__continue_and_finalize_provisional_source_branch,
                adapter,
                source_boundary_state,
                source_plan,
            )

            source_result, target_transition_result = (
                adapter._run_shared_handoff_boundary_transition(
                    source_boundary_state,
                    prepared_transition,
                    run_provisional_source_branch=provisional_source_continuation,
                    target_residual_state=target_residual_state,
                    transition_index=transition_index,
                )
            )

        if target_state is None:
            target_state = adapter._fork_recurrent_handoff_state(source_boundary_state)
        target_state = adapter._apply_recurrent_transition_result(
            target_state,
            target_transition_result,
        )
        return source_result, target_state

    def __continue_and_finalize_provisional_source_branch(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        source_boundary_state: _StateT,
        source_plan: RecurrentBranchExecutionPlan,
        source_transition_result: RecurrentTransitionResult,
    ) -> RecurrentExecutionResult:
        source_state = adapter._fork_recurrent_handoff_state(source_boundary_state)
        source_state = adapter._apply_recurrent_transition_result(
            source_state,
            source_transition_result,
        )
        source_state = self.__branch_runner.run(
            adapter,
            source_state,
            source_plan,
            observe_transitions=False,
        )
        return self.__branch_runner.finalize(adapter, source_state)
