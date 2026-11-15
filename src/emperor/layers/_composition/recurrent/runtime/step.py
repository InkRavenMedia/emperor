from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from emperor.layers._composition.recurrent.validation import (
    HierarchicalReasoningModelRecurrentValidator,
    RecurrentLayerValidator,
    TinyRecursiveModelRecurrentValidator,
)
from emperor.memory import MemoryPositionOptions

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from torch import Tensor

    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.recurrent.utilities import RecurrentUtilities
    from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model.state import (
        HierarchicalReasoningModelState,
    )
    from emperor.layers._composition.recurrent.variants.standard.state import (
        StandardRecurrentState,
    )
    from emperor.layers._composition.recurrent.variants.tiny_recursive_model.state import (
        TinyRecursiveModelState,
    )
    from emperor.layers._composition.residual.base import ResidualState
    from emperor.layers._state import LayerState


RecurrentStateT = TypeVar(
    "RecurrentStateT",
    bound="StandardRecurrentState | TinyRecursiveModelState | HierarchicalReasoningModelState",
)


class RecurrentStep(Generic[RecurrentStateT], ABC):
    VALIDATOR: ClassVar[
        type[RecurrentLayerValidator]
        | type[TinyRecursiveModelRecurrentValidator]
        | type[HierarchicalReasoningModelRecurrentValidator]
    ]

    def __init__(self, utilities: RecurrentUtilities, output_dim: int) -> None:
        self.utilities = utilities
        self.output_dim = output_dim
        self.diagnostic_observer: Callable[[Tensor, Tensor], None] | None = None
        self.__diagnostic_suppression_depth = 0

    @abstractmethod
    def forward(
        self,
        recurrent_state: RecurrentStateT,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> RecurrentStateT:
        pass

    @abstractmethod
    def initialize(
        self, state: LayerState, include_caller_loss: bool = True
    ) -> RecurrentStateT:
        pass

    @abstractmethod
    def compute_output(
        self,
        recurrent_state: RecurrentStateT,
        target_state: RecurrentStateT | None = None,
    ) -> LayerState:
        pass

    @abstractmethod
    def update_state(
        self,
        recurrent_state: RecurrentStateT,
        output_state: LayerState,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> RecurrentStateT:
        pass

    def fork(self, recurrent_state: RecurrentStateT) -> RecurrentStateT:
        return replace(recurrent_state)

    def finalize(self, recurrent_state: RecurrentStateT) -> LayerState:
        finalized_hidden, finalized_loss = self.utilities.finalize_halting(
            recurrent_state.output_hidden,
            recurrent_state.branch_loss,
            recurrent_state.halting_state,
        )
        return replace(
            recurrent_state.context_state, hidden=finalized_hidden, loss=finalized_loss
        )

    def _compute_block_output(
        self,
        state: LayerState,
        previous_hidden: Tensor,
        previous_context: LayerState,
        block: Callable[[LayerState], LayerState],
        transition_index: int,
        residual_state: ResidualState | None = None,
        target_residual_state: ResidualState | None = None,
    ) -> LayerState:
        state = self.utilities.normalize_before_model(state)
        block_input = self.utilities.apply_memory(
            state.hidden, MemoryPositionOptions.BEFORE_AFFINE
        )
        state.hidden = block_input
        output_state = block(state)
        self.VALIDATOR.validate_transition_output(
            output_state, block_input, self.output_dim
        )
        candidate_hidden = self.utilities.apply_memory(
            output_state.hidden, MemoryPositionOptions.AFTER_AFFINE
        )
        candidate_state = replace(output_state, hidden=candidate_hidden)
        candidate_state = self.utilities.normalize_after_model(candidate_state)
        candidate_state.hidden = self.utilities.apply_gate(candidate_state.hidden)
        residual_input = candidate_state.hidden
        aligned_previous_hidden = output_state.align_recurrent_hidden(
            previous_hidden, source_state=previous_context
        )
        candidate_state.hidden = self.utilities.apply_residual(
            candidate_state.hidden,
            aligned_previous_hidden,
            residual_state,
            transition_index,
        )
        candidate_state = self.utilities.normalize_after_residual(candidate_state)
        self.utilities.advance_detached_target_residual_state(
            residual_input, previous_hidden, target_residual_state, transition_index
        )
        return candidate_state

    def _apply_halting(
        self,
        recurrent_state: RecurrentStateT,
        output_state: LayerState,
        previous_hidden: Tensor,
        update_enabled: bool,
        observe_transition: bool,
    ) -> tuple[HaltingStateBase | None, Tensor]:
        halting_state, output_hidden = self.utilities.apply_halting(
            recurrent_state.halting_state,
            output_state.hidden,
            update_enabled=update_enabled,
        )
        if observe_transition:
            aligned_previous_hidden = output_state.align_recurrent_hidden(
                previous_hidden, source_state=recurrent_state.context_state
            )
            if self.diagnostic_observer is not None:
                self.diagnostic_observer(aligned_previous_hidden, output_hidden)
        return halting_state, output_hidden

    def set_diagnostic_observer(
        self, observer: Callable[[Tensor, Tensor], None] | None
    ) -> None:
        self.diagnostic_observer = observer

    @property
    def diagnostic_observation_enabled(self) -> bool:
        return self.__diagnostic_suppression_depth == 0

    @contextmanager
    def _observe_model(self, enabled: bool) -> Generator[None, None, None]:
        if enabled:
            yield
            return
        self.__diagnostic_suppression_depth += 1
        try:
            yield
        finally:
            self.__diagnostic_suppression_depth -= 1
