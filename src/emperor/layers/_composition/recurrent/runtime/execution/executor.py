# pyright: reportPrivateUsage=false

from __future__ import annotations

from contextlib import contextmanager
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
    PreparedRecurrentTransition,
    RecurrentExecutionAdapter,
    RecurrentExecutionResult,
    RecurrentExecutionState,
    RecurrentTransitionCandidate,
    RecurrentTransitionResult,
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
from emperor.memory import MemoryPositionOptions
from emperor.nn import once_per_recurrent_run

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from torch import Tensor

    from emperor.layers._composition.recurrent.utilities import RecurrentUtilities


class RecurrentTransitionExecution:
    def __init__(self, utilities: RecurrentUtilities, output_dim: int) -> None:
        self.utilities = utilities
        self.output_dim = output_dim
        self.diagnostic_observer: Callable[[Tensor, Tensor], None] | None = None
        self.__diagnostic_suppression_depth = 0

    def set_diagnostic_observer(
        self, observer: Callable[[Tensor, Tensor], None] | None
    ) -> None:
        self.diagnostic_observer = observer

    @property
    def diagnostic_observation_enabled(self) -> bool:
        return self.__diagnostic_suppression_depth == 0

    @contextmanager
    def __diagnostic_observation_context(
        self, enabled: bool
    ) -> Generator[None, None, None]:
        if enabled:
            yield
            return
        self.__diagnostic_suppression_depth += 1
        try:
            yield
        finally:
            self.__diagnostic_suppression_depth -= 1

    def run_transition(
        self,
        recurrent_state: RecurrentExecutionState,
        prepared_transition: PreparedRecurrentTransition,
        validate_output: Callable[[object, Tensor, int], None],
        transition_index: int = 0,
        observe_transition: bool = True,
    ) -> RecurrentTransitionResult:
        with self.__diagnostic_observation_context(observe_transition):
            transition_candidate = self.compute_transition(
                recurrent_state,
                prepared_transition,
                validate_output,
                transition_index,
            )
        return self.complete_transition(
            recurrent_state,
            transition_candidate,
            previous_evolving_hidden=prepared_transition.previous_evolving_hidden,
            halting_update_enabled=prepared_transition.halting_update_enabled,
            observe_transition=observe_transition,
        )

    def compute_transition(
        self,
        recurrent_state: RecurrentExecutionState,
        prepared_transition: PreparedRecurrentTransition,
        validate_output: Callable[[object, Tensor, int], None],
        transition_index: int,
    ) -> RecurrentTransitionCandidate:
        transition_state = replace(
            recurrent_state.context_state,
            hidden=prepared_transition.transition_input,
            loss=prepared_transition.loss,
            halting_state=None,
        )
        transition_state = self.utilities.normalize_before_model(transition_state)
        transition_model_input = self.utilities.apply_memory(
            transition_state.hidden,
            MemoryPositionOptions.BEFORE_AFFINE,
        )
        transition_state.hidden = transition_model_input
        output_state = prepared_transition.run_transition(transition_state)
        validate_output(
            output_state,
            transition_model_input,
            self.output_dim,
        )
        candidate_hidden = self.utilities.apply_memory(
            output_state.hidden,
            MemoryPositionOptions.AFTER_AFFINE,
        )
        candidate_state = replace(output_state, hidden=candidate_hidden)
        candidate_state = self.utilities.normalize_after_model(candidate_state)
        candidate_state.hidden = self.utilities.apply_gate(candidate_state.hidden)
        residual_input = candidate_state.hidden
        aligned_previous_hidden = output_state.align_recurrent_hidden(
            prepared_transition.previous_evolving_hidden,
            source_state=recurrent_state.context_state,
        )
        candidate_state.hidden = self.utilities.apply_residual(
            candidate_state.hidden,
            aligned_previous_hidden,
            prepared_transition.residual_state,
            transition_index,
        )
        candidate_state = self.utilities.normalize_after_residual(candidate_state)
        return RecurrentTransitionCandidate(
            hidden=candidate_state.hidden,
            loss=output_state.loss,
            residual_input=residual_input,
            context_state=candidate_state,
        )

    def complete_transition(
        self,
        recurrent_state: RecurrentExecutionState,
        transition_candidate: RecurrentTransitionCandidate,
        previous_evolving_hidden: Tensor,
        halting_update_enabled: bool,
        observe_transition: bool,
    ) -> RecurrentTransitionResult:
        halting_state, output_hidden = self.utilities.apply_halting(
            recurrent_state.halting_state,
            transition_candidate.hidden,
            update_enabled=halting_update_enabled,
        )
        if observe_transition:
            aligned_previous_hidden = (
                transition_candidate.context_state.align_recurrent_hidden(
                    previous_evolving_hidden,
                    source_state=recurrent_state.context_state,
                )
            )
            if self.diagnostic_observer is not None:
                self.diagnostic_observer(aligned_previous_hidden, output_hidden)
        return RecurrentTransitionResult(
            hidden=output_hidden,
            loss=transition_candidate.loss,
            halting_state=halting_state,
            all_items_halted=self.utilities.all_items_halted(halting_state),
            context_state=replace(
                transition_candidate.context_state, hidden=output_hidden
            ),
        )


class RecurrentExecution(Generic[_StateT]):
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
