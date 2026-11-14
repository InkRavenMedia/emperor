"""Coordinate recurrent execution delegates and commit one forward result."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Generic, cast

from torch.nn import Module

from emperor.layers._composition.recurrent.runtime.execution.branch import (
    RecurrentBranchRunner,
)
from emperor.layers._composition.recurrent.runtime.execution.handoff import (
    RecurrentBranchedHandoff,
    RecurrentNestedHandoff,
)
from emperor.layers._composition.recurrent.runtime.execution.interface import (
    RecurrentExecutionAdapter,
    RecurrentExecutionResult,
    _StateT,
)
from emperor.layers._composition.recurrent.runtime.execution.runtime_state import (
    RecurrentRuntimeStateGuard,
)
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentIterationExecutionPlan,
    RecurrentIterationSchedule,
    RecurrentNestedSmoothHandoffExecutionPlan,
    RecurrentSmoothHandoffExecutionPlan,
)
from emperor.layers._composition.recurrent.validation import (
    RecurrentExecutionValidator,
)
from emperor.layers._state import LayerState
from emperor.nn import once_per_recurrent_run

if TYPE_CHECKING:
    from collections.abc import Callable


class RecurrentExecution(Generic[_StateT]):
    """Select an execution delegate and commit its result for every variant."""

    VALIDATOR = RecurrentExecutionValidator
    __runtime_state_guard = RecurrentRuntimeStateGuard()

    def __init__(self) -> None:
        self.__branch_runner: RecurrentBranchRunner[_StateT] = RecurrentBranchRunner()
        self.__branched_handoff = RecurrentBranchedHandoff(self.__branch_runner)
        self.__nested_handoff = RecurrentNestedHandoff(self.__branch_runner)

    @once_per_recurrent_run()
    def execute(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        layer_state: LayerState,
        iteration_schedule: RecurrentIterationSchedule,
    ) -> LayerState:
        self.VALIDATOR.validate_adapter_is_module(adapter)
        execution_plan = iteration_schedule.execution_plan()
        if isinstance(execution_plan, RecurrentNestedSmoothHandoffExecutionPlan):
            run_handoff = partial(
                self.__nested_handoff.execute, adapter, layer_state, execution_plan
            )
        elif isinstance(execution_plan, RecurrentSmoothHandoffExecutionPlan):
            run_handoff = partial(
                self.__branched_handoff.execute, adapter, layer_state, execution_plan
            )
        else:
            execution_result = self.__execute_stable_plan(
                adapter,
                layer_state,
                execution_plan,
            )
            return self.__commit_execution_result(
                layer_state,
                iteration_schedule,
                execution_result,
            )
        return self.__execute_handoff(
            adapter,
            layer_state,
            iteration_schedule,
            run_handoff,
        )

    def __execute_handoff(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        layer_state: LayerState,
        iteration_schedule: RecurrentIterationSchedule,
        run_handoff: Callable[[], RecurrentExecutionResult],
    ) -> LayerState:
        """Keep handoff computation and commit inside one failure-restoration scope."""
        original_state = replace(layer_state)
        try:
            with self.__runtime_state_guard.rollback_handoff_on_failure(
                cast(Module, adapter)
            ):
                execution_result = run_handoff()
                return self.__commit_execution_result(
                    layer_state,
                    iteration_schedule,
                    execution_result,
                )
        except BaseException:
            layer_state.hidden = original_state.hidden
            layer_state.loss = original_state.loss
            layer_state.commit_recurrent_context(original_state)
            raise

    def __execute_stable_plan(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        layer_state: LayerState,
        execution_plan: RecurrentIterationExecutionPlan,
    ) -> RecurrentExecutionResult:
        recurrent_state = adapter._initialize_recurrent_execution_state(
            layer_state,
            branch_base_loss=layer_state.loss,
        )
        recurrent_state = self.__branch_runner.run(
            adapter,
            recurrent_state,
            execution_plan.target_branch,
            observe_transitions=True,
        )
        return self.__branch_runner.finalize(adapter, recurrent_state)

    @staticmethod
    def __commit_execution_result(
        layer_state: LayerState,
        iteration_schedule: RecurrentIterationSchedule,
        execution_result: RecurrentExecutionResult,
    ) -> LayerState:
        layer_state.hidden = execution_result.hidden
        layer_state.loss = execution_result.loss
        if execution_result.context_state is not None:
            layer_state.commit_recurrent_context(execution_result.context_state)
        iteration_schedule.record_successful_forward()
        return layer_state
