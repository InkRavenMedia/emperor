from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from emperor.layers._composition.recurrent.runtime.step import RecurrentStep
from emperor.layers._composition.recurrent.validation import RecurrentLayerValidator
from emperor.layers._composition.recurrent.variants.standard.state import (
    StandardRecurrentState,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from emperor.layers._composition.recurrent.utilities import RecurrentUtilities
    from emperor.layers._state import LayerState


class StandardRecurrentStep(RecurrentStep[StandardRecurrentState]):
    VALIDATOR = RecurrentLayerValidator

    def __init__(
        self,
        utilities: RecurrentUtilities,
        output_dim: int,
        get_module: Callable[[str], nn.Module],
        reinject_original_hidden: bool,
    ) -> None:
        super().__init__(utilities, output_dim)
        self.reinject_original_hidden = reinject_original_hidden
        self._get_module = get_module

    def initialize(
        self, state: LayerState, include_caller_loss: bool = True
    ) -> StandardRecurrentState:
        residual_state = self.utilities.initialize_residual_state(state.hidden)
        return StandardRecurrentState(
            hidden=state.hidden,
            fixed_input=state.hidden,
            loss=state.loss if include_caller_loss else None,
            context_state=state,
            transition_index=0,
            residual_state=residual_state,
        )

    def forward(
        self,
        recurrent_state: StandardRecurrentState,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> StandardRecurrentState:
        with self._observe_model(observe_transition):
            output_state = self.compute_output(recurrent_state)
        return self.update_state(
            recurrent_state, output_state, tracks_gradients, observe_transition
        )

    def compute_output(
        self,
        recurrent_state: StandardRecurrentState,
        target_state: StandardRecurrentState | None = None,
    ) -> LayerState:
        transition_input = recurrent_state.hidden
        if self.reinject_original_hidden:
            transition_input = transition_input + recurrent_state.fixed_input
        transition_state = replace(
            recurrent_state.context_state,
            hidden=transition_input,
            loss=recurrent_state.loss,
            halting_state=None,
        )
        return self._compute_block_output(
            transition_state,
            recurrent_state.hidden,
            recurrent_state.context_state,
            partial(self._run_block, recurrent_state=recurrent_state),
            recurrent_state.transition_index,
            recurrent_state.residual_state,
            target_state.residual_state if target_state is not None else None,
        )

    def _run_block(
        self, state: LayerState, recurrent_state: StandardRecurrentState
    ) -> LayerState:
        return self._get_module("block_model")(state)

    def update_state(
        self,
        recurrent_state: StandardRecurrentState,
        output_state: LayerState,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> StandardRecurrentState:
        halting_state, output_hidden = self._apply_halting(
            recurrent_state,
            output_state,
            recurrent_state.hidden,
            tracks_gradients,
            observe_transition,
        )
        context_state = replace(output_state, hidden=output_hidden)
        return replace(
            recurrent_state,
            hidden=output_hidden,
            loss=output_state.loss,
            context_state=context_state,
            fixed_input=context_state.align_recurrent_hidden(
                recurrent_state.fixed_input, source_state=recurrent_state.context_state
            ),
            transition_index=recurrent_state.transition_index + 1,
            halting_state=halting_state,
            all_items_halted=self.utilities.all_items_halted(halting_state),
        )

    def fork(self, recurrent_state: StandardRecurrentState) -> StandardRecurrentState:
        return replace(
            recurrent_state,
            residual_state=self.utilities.fork_residual_state(
                recurrent_state.residual_state
            ),
        )
