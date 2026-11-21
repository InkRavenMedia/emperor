from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Generic

import torch

from emperor.layers._composition.recurrent.runtime.handoff.results import (
    accumulate_caller_loss_once,
    blend_recurrent_branch_hidden,
    blend_recurrent_branch_losses,
)
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentNestedSmoothHandoffExecutionPlan,
    RecurrentSmoothHandoffExecutionPlan,
)
from emperor.layers._composition.recurrent.runtime.runtime_state import (
    RecurrentRuntimeStateGuard,
)
from emperor.layers._composition.recurrent.runtime.step import (
    RecurrentStateT,
    RecurrentStep,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn

    from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
        RecurrentIterationSchedule,
    )
    from emperor.layers._state import LayerState


class RecurrentDepthHandoff(Generic[RecurrentStateT]):
    def __init__(
        self, step: RecurrentStep[RecurrentStateT], schedule: RecurrentIterationSchedule
    ) -> None:
        self.step = step
        self.schedule = schedule
        self.__runtime_state_guard = RecurrentRuntimeStateGuard()

    def run_and_blend_depth_results(
        self,
        state: LayerState,
        runtime_modules: Callable[[], Iterable[nn.Module]],
    ) -> LayerState:
        execution_plan = self.schedule.execution_plan()
        original_state = replace(state)
        try:
            with self.__runtime_state_guard.rollback_handoff_on_failure(
                runtime_modules()
            ):
                recurrent_state = self.step.initialize_recurrent_state(
                    state, include_caller_loss=False
                )
                if isinstance(
                    execution_plan, RecurrentNestedSmoothHandoffExecutionPlan
                ):
                    result = self.__run_nested_depths(recurrent_state, execution_plan)
                elif isinstance(execution_plan, RecurrentSmoothHandoffExecutionPlan):
                    result = self.__run_branched_depths(
                        recurrent_state, execution_plan, runtime_modules
                    )
                else:
                    raise ValueError(
                        "a depth handoff requires a transitioning iteration schedule"
                    )
                result = accumulate_caller_loss_once(state.loss, result)
                state.hidden = result.hidden
                state.loss = result.loss
                state.commit_recurrent_context(result)
                self.schedule.record_successful_forward()
                return state
        except BaseException:
            state.hidden = original_state.hidden
            state.loss = original_state.loss
            state.commit_recurrent_context(original_state)
            raise

    def __run_steps(
        self,
        recurrent_state: RecurrentStateT,
        step_limit: int,
        no_gradient_steps: int,
        observe_transitions: bool = True,
    ) -> RecurrentStateT:
        while self.__has_remaining_steps_and_active_items(recurrent_state, step_limit):
            if self.__is_first_step_after_no_gradient_prefix(
                recurrent_state, no_gradient_steps
            ):
                recurrent_state = recurrent_state.detach()
            tracks_gradients = recurrent_state.transition_index >= no_gradient_steps
            with self.__gradient_context_for_step(tracks_gradients):
                recurrent_state = self.step(
                    recurrent_state, tracks_gradients, observe_transitions
                )
        return recurrent_state

    def __has_remaining_steps_and_active_items(
        self, recurrent_state: RecurrentStateT, step_limit: int
    ) -> bool:
        return (
            recurrent_state.transition_index < step_limit
            and not recurrent_state.all_items_halted
        )

    def __is_first_step_after_no_gradient_prefix(
        self, recurrent_state: RecurrentStateT, no_gradient_steps: int
    ) -> bool:
        return (
            no_gradient_steps > 0
            and recurrent_state.transition_index == no_gradient_steps
        )

    def __gradient_context_for_step(
        self, tracks_gradients: bool
    ) -> AbstractContextManager[None]:
        if tracks_gradients:
            return nullcontext()
        return torch.no_grad()

    def __run_nested_depths(
        self,
        recurrent_state: RecurrentStateT,
        plan: RecurrentNestedSmoothHandoffExecutionPlan,
    ) -> LayerState:
        recurrent_state = self.__run_steps(
            recurrent_state,
            plan.source_branch.transition_count,
            plan.source_branch.no_gradient_transition_count,
        )
        source_result = self.step.finalize(recurrent_state)
        if recurrent_state.all_items_halted:
            return source_result
        source_transition_index = recurrent_state.transition_index
        recurrent_state = self.__run_steps(
            recurrent_state,
            plan.target_branch.transition_count,
            plan.target_branch.no_gradient_transition_count,
        )
        if recurrent_state.transition_index == source_transition_index:
            return source_result
        target_result = self.step.finalize(recurrent_state)
        return replace(
            source_result if plan.transition_weight == 0 else target_result,
            hidden=blend_recurrent_branch_hidden(
                source_result, target_result, plan.transition_weight
            ),
            loss=blend_recurrent_branch_losses(
                source_result.loss,
                source_result.loss,
                target_result.loss,
                plan.transition_weight,
            ),
        )

    def __run_branched_depths(
        self,
        recurrent_state: RecurrentStateT,
        plan: RecurrentSmoothHandoffExecutionPlan,
        runtime_modules: Callable[[], Iterable[nn.Module]],
    ) -> LayerState:
        common_state = self.__run_steps(
            recurrent_state,
            plan.common_prefix_transition_count,
            plan.common_prefix_transition_count,
        )
        source_result, target_state = self.__run_shared_boundary(
            common_state, plan, runtime_modules
        )
        target_state = self.__run_steps(
            target_state,
            plan.target_branch.transition_count,
            plan.target_branch.no_gradient_transition_count,
        )
        target_result = self.step.finalize(target_state)
        return replace(
            source_result if plan.transition_weight == 0 else target_result,
            hidden=blend_recurrent_branch_hidden(
                source_result, target_result, plan.transition_weight
            ),
            loss=blend_recurrent_branch_losses(
                common_state.branch_loss,
                source_result.loss,
                target_result.loss,
                plan.transition_weight,
            ),
        )

    def __run_shared_boundary(
        self,
        recurrent_state: RecurrentStateT,
        plan: RecurrentSmoothHandoffExecutionPlan,
        runtime_modules: Callable[[], Iterable[nn.Module]],
    ) -> tuple[LayerState, RecurrentStateT]:
        no_gradient_steps = plan.source_branch.no_gradient_transition_count
        if self.__is_first_step_after_no_gradient_prefix(
            recurrent_state, no_gradient_steps
        ):
            recurrent_state = recurrent_state.detach()
        tracks_gradients = recurrent_state.transition_index >= no_gradient_steps
        with self.__gradient_context_for_step(tracks_gradients):
            target_state = self.step.fork(recurrent_state)
            output_state = self.step.compute_output(
                recurrent_state, target_state=target_state
            )
            with self.__runtime_state_guard.isolate_provisional_branch(
                runtime_modules()
            ):
                source_state = self.step.update_state(
                    recurrent_state,
                    output_state,
                    tracks_gradients,
                    observe_transition=False,
                )
                source_state = self.step.fork(source_state)
                source_state = self.__run_steps(
                    source_state,
                    plan.source_branch.transition_count,
                    no_gradient_steps,
                    observe_transitions=False,
                )
                source_result = self.step.finalize(source_state)
            detached_output = replace(
                output_state.detach_recurrent_context(),
                hidden=output_state.hidden.detach(),
                loss=None if output_state.loss is None else output_state.loss.detach(),
            )
            target_state = self.step.update_state(
                target_state, detached_output, tracks_gradients=False
            )
        return source_result, target_state
