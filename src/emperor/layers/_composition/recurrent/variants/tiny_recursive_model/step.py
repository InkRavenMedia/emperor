from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from emperor.layers._composition.recurrent.runtime.step import RecurrentStep
from emperor.layers._composition.recurrent.validation import (
    TinyRecursiveModelRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.tiny_recursive_model.state import (
    TinyRecursiveModelState,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor, nn

    from emperor.layers._composition.recurrent.utilities import RecurrentUtilities
    from emperor.layers._state import LayerState


class TinyRecursiveModelStep(RecurrentStep[TinyRecursiveModelState]):
    VALIDATOR = TinyRecursiveModelRecurrentValidator

    def __init__(
        self,
        utilities: RecurrentUtilities,
        output_dim: int,
        get_module: Callable[[str], nn.Module],
        get_buffer: Callable[[str], Tensor],
        latent_updates_per_answer_update: int,
    ) -> None:
        super().__init__(utilities, output_dim)
        self.latent_updates_per_answer_update = latent_updates_per_answer_update
        self.__get_module = get_module
        self.__get_buffer = get_buffer

    def initialize(
        self, state: LayerState, include_caller_loss: bool = True
    ) -> TinyRecursiveModelState:
        initial_shape = (1,) * (state.hidden.ndim - 1) + (state.hidden.shape[-1],)
        return TinyRecursiveModelState(
            fixed_input=state.hidden,
            answer=self.__get_buffer("answer_initial")
            .view(initial_shape)
            .expand_as(state.hidden),
            latent=self.__get_buffer("latent_initial")
            .view(initial_shape)
            .expand_as(state.hidden),
            initial_loss=state.loss if include_caller_loss else None,
            auxiliary_losses=(),
            context_state=state,
            transition_index=0,
        )

    def forward(
        self,
        recurrent_state: TinyRecursiveModelState,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> TinyRecursiveModelState:
        with self._observe_model(observe_transition):
            output_state = self.compute_output(recurrent_state)
        return self.update_state(
            recurrent_state, output_state, tracks_gradients, observe_transition
        )

    def compute_output(
        self,
        recurrent_state: TinyRecursiveModelState,
        target_state: TinyRecursiveModelState | None = None,
    ) -> LayerState:
        updates_primary = self.__updates_answer(recurrent_state.transition_index)
        if updates_primary:
            transition_input = recurrent_state.answer + recurrent_state.latent
            previous_hidden = recurrent_state.answer
        else:
            transition_input = (
                recurrent_state.latent
                + recurrent_state.answer
                + recurrent_state.fixed_input
            )
            previous_hidden = recurrent_state.latent
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
            self.__get_module("block_model"),
            recurrent_state.transition_index,
        )

    def update_state(
        self,
        recurrent_state: TinyRecursiveModelState,
        output_state: LayerState,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> TinyRecursiveModelState:
        updates_primary = self.__updates_answer(recurrent_state.transition_index)
        previous_hidden = (
            recurrent_state.answer if updates_primary else recurrent_state.latent
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
        answer = context_state.align_recurrent_hidden(
            recurrent_state.answer,
            initial=self.__get_buffer("answer_initial"),
            source_state=recurrent_state.context_state,
        )
        latent = context_state.align_recurrent_hidden(
            recurrent_state.latent,
            initial=self.__get_buffer("latent_initial"),
            source_state=recurrent_state.context_state,
        )
        if updates_primary:
            answer = output_hidden
        else:
            latent = output_hidden
        return replace(
            recurrent_state,
            answer=answer,
            latent=latent,
            context_state=context_state,
            fixed_input=context_state.align_recurrent_hidden(
                recurrent_state.fixed_input, source_state=recurrent_state.context_state
            ),
            auxiliary_losses=auxiliary_losses,
            transition_index=recurrent_state.transition_index + 1,
            halting_state=halting_state,
            all_items_halted=self.utilities.all_items_halted(halting_state),
        )

    def __updates_answer(self, transition_index: int) -> bool:
        phase_index = transition_index % (self.latent_updates_per_answer_update + 1)
        return phase_index >= self.latent_updates_per_answer_update
