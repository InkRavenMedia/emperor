from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

from emperor.layers._composition.recurrent.runtime.execution.interface import (
    PreparedRecurrentTransition,
    RecurrentExecutionResult,
    RecurrentExecutionState,
    RecurrentTransitionCandidate,
    RecurrentTransitionResult,
)
from emperor.layers._composition.recurrent.runtime.execution.runtime_state import (
    RecurrentRuntimeStateGuard,
)
from emperor.memory import MemoryPositionOptions

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from torch import Tensor, nn

    from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
        RecurrentIterationSchedule,
    )
    from emperor.layers._composition.recurrent.utilities import RecurrentUtilities
    from emperor.layers._state import LayerState


class RecurrentExecution:
    def __init__(self, utilities: RecurrentUtilities, output_dim: int) -> None:
        self.utilities = utilities
        self.output_dim = output_dim
        self.diagnostic_observer: Callable[[Tensor, Tensor], None] | None = None
        self.__diagnostic_suppression_depth = 0
        self.__runtime_state_guard = RecurrentRuntimeStateGuard()

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

    def finalize(
        self, recurrent_state: RecurrentExecutionState
    ) -> RecurrentExecutionResult:
        finalized_hidden, finalized_loss = self.utilities.finalize_halting(
            recurrent_state.output_hidden,
            recurrent_state.branch_loss,
            recurrent_state.halting_state,
        )
        return RecurrentExecutionResult(
            finalized_hidden,
            finalized_loss,
            replace(recurrent_state.context_state, hidden=finalized_hidden),
        )

    def commit(
        self,
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

    @contextmanager
    def handoff(
        self, layer_state: LayerState, runtime_modules: Iterable[nn.Module]
    ) -> Generator[None, None, None]:
        original_state = replace(layer_state)
        try:
            with self.__runtime_state_guard.rollback_handoff_on_failure(
                runtime_modules
            ):
                yield
        except BaseException:
            layer_state.hidden = original_state.hidden
            layer_state.loss = original_state.loss
            layer_state.commit_recurrent_context(original_state)
            raise
