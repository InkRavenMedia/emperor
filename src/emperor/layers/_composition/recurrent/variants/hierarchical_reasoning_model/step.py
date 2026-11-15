from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from emperor.layers._composition.recurrent.runtime.step import RecurrentStep
from emperor.layers._composition.recurrent.validation import (
    HierarchicalReasoningModelRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model.state import (
    HierarchicalReasoningModelState,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor, nn

    from emperor.layers._composition.recurrent.utilities import RecurrentUtilities
    from emperor.layers._state import LayerState


class HierarchicalReasoningModelStep(RecurrentStep[HierarchicalReasoningModelState]):
    VALIDATOR = HierarchicalReasoningModelRecurrentValidator

    def __init__(
        self,
        utilities: RecurrentUtilities,
        output_dim: int,
        get_module: Callable[[str], nn.Module],
        get_buffer: Callable[[str], Tensor],
        low_cycles: int,
    ) -> None:
        super().__init__(utilities, output_dim)
        self.low_cycles = low_cycles
        self.__get_module = get_module
        self.__get_buffer = get_buffer

    def initialize(
        self, state: LayerState, include_caller_loss: bool = True
    ) -> HierarchicalReasoningModelState:
        initial_shape = (1,) * (state.hidden.ndim - 1) + (state.hidden.shape[-1],)
        return HierarchicalReasoningModelState(
            fixed_input=state.hidden,
            high=self.__get_buffer("high_initial")
            .view(initial_shape)
            .expand_as(state.hidden),
            low=self.__get_buffer("low_initial")
            .view(initial_shape)
            .expand_as(state.hidden),
            initial_loss=state.loss if include_caller_loss else None,
            auxiliary_losses=(),
            context_state=state,
            transition_index=0,
        )

    def forward(
        self,
        recurrent_state: HierarchicalReasoningModelState,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> HierarchicalReasoningModelState:
        with self._observe_model(observe_transition):
            output_state = self.compute_output(recurrent_state)
        return self.update_state(
            recurrent_state, output_state, tracks_gradients, observe_transition
        )

    def compute_output(
        self,
        recurrent_state: HierarchicalReasoningModelState,
        target_state: HierarchicalReasoningModelState | None = None,
    ) -> LayerState:
        updates_primary = self.__updates_high(recurrent_state.transition_index)
        if updates_primary:
            transition_input = recurrent_state.high + recurrent_state.low
            previous_hidden = recurrent_state.high
        else:
            transition_input = (
                recurrent_state.low + recurrent_state.high + recurrent_state.fixed_input
            )
            previous_hidden = recurrent_state.low
        transition_state = replace(
            recurrent_state.context_state,
            hidden=transition_input,
            loss=None,
            halting_state=None,
        )
        return self._compute_block_output(
            transition_state,
            previous_hidden,
            recurrent_state.context_state,
            self.__get_module("high_model" if updates_primary else "low_model"),
            recurrent_state.transition_index,
        )

    def update_state(
        self,
        recurrent_state: HierarchicalReasoningModelState,
        output_state: LayerState,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> HierarchicalReasoningModelState:
        updates_primary = self.__updates_high(recurrent_state.transition_index)
        previous_hidden = (
            recurrent_state.high if updates_primary else recurrent_state.low
        )
        halting_state, output_hidden = self._apply_halting(
            recurrent_state,
            output_state,
            previous_hidden,
            tracks_gradients and updates_primary,
            observe_transition,
        )
        auxiliary_losses = recurrent_state.auxiliary_losses
        if output_state.loss is not None:
            reduced_loss = (
                output_state.loss
                if output_state.loss.dim() == 0
                else output_state.loss.mean()
            )
            auxiliary_losses = (*auxiliary_losses, reduced_loss)
        context_state = replace(output_state, hidden=output_hidden)
        high = context_state.align_recurrent_hidden(
            recurrent_state.high,
            initial=self.__get_buffer("high_initial"),
            source_state=recurrent_state.context_state,
        )
        low = context_state.align_recurrent_hidden(
            recurrent_state.low,
            initial=self.__get_buffer("low_initial"),
            source_state=recurrent_state.context_state,
        )
        if updates_primary:
            high = output_hidden
        else:
            low = output_hidden
        return replace(
            recurrent_state,
            high=high,
            low=low,
            context_state=context_state,
            fixed_input=context_state.align_recurrent_hidden(
                recurrent_state.fixed_input, source_state=recurrent_state.context_state
            ),
            auxiliary_losses=auxiliary_losses,
            transition_index=recurrent_state.transition_index + 1,
            halting_state=halting_state,
            all_items_halted=self.utilities.all_items_halted(halting_state),
        )

    def __updates_high(self, transition_index: int) -> bool:
        phase_index = transition_index % (self.low_cycles + 1)
        return phase_index >= self.low_cycles
