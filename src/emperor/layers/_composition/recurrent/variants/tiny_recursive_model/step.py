from __future__ import annotations

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

    from torch import Tensor

    from emperor.layers._composition.recurrent.config import (
        TinyRecursiveModelRecurrentConfig,
    )
    from emperor.layers._state import LayerState


class TinyRecursiveModelStep(RecurrentStep[TinyRecursiveModelState]):
    VALIDATOR = TinyRecursiveModelRecurrentValidator

    def __init__(
        self,
        cfg: TinyRecursiveModelRecurrentConfig,
        register_initial_buffer: Callable[[str], None],
        get_buffer: Callable[[str], Tensor],
    ) -> None:
        super().__init__(cfg)
        self.latent_updates_per_answer_update: int = (
            cfg.latent_updates_per_answer_update
        )
        self.__get_buffer = get_buffer
        register_initial_buffer("answer_initial")
        register_initial_buffer("latent_initial")
        self.block_model = self._build_transition_model(cfg.block_config)

    def initialize_recurrent_state(
        self, state: LayerState, include_caller_loss: bool = True
    ) -> TinyRecursiveModelState:
        initial_shape = (1,) * (state.hidden.ndim - 1) + (state.hidden.shape[-1],)
        answer_initial = self.__get_buffer("answer_initial")
        answer = answer_initial.view(initial_shape).expand_as(state.hidden)
        latent_initial = self.__get_buffer("latent_initial")
        latent = latent_initial.view(initial_shape).expand_as(state.hidden)
        return TinyRecursiveModelState(
            fixed_input=state.hidden,
            answer=answer,
            latent=latent,
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
        updates_answer = self.__updates_answer(recurrent_state.transition_index)
        if updates_answer:
            transition_input = recurrent_state.answer + recurrent_state.latent
            previous_hidden = recurrent_state.answer
        else:
            latent_input = recurrent_state.latent + recurrent_state.answer
            transition_input = latent_input + recurrent_state.fixed_input
            previous_hidden = recurrent_state.latent
        transition_state = recurrent_state.layer_state_for_transition(transition_input)
        return self.pipeline.process_recurrent_transition(
            transition_state,
            run_model=self.__run_block,
            recurrent_state=recurrent_state,
            previous_hidden=previous_hidden,
        )

    def __updates_answer(self, transition_index: int) -> bool:
        phase_index = transition_index % (self.latent_updates_per_answer_update + 1)
        return phase_index >= self.latent_updates_per_answer_update

    def __run_block(self, state: LayerState) -> LayerState:
        block_input = state.hidden
        output_state = self._call_transition_block(self.block_model, state)
        self.VALIDATOR.validate_transition_output(
            output_state, block_input, self.cfg.output_dim
        )
        return output_state

    def update_state(
        self,
        recurrent_state: TinyRecursiveModelState,
        output_state: LayerState,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> TinyRecursiveModelState:
        updates_answer = self.__updates_answer(recurrent_state.transition_index)
        previous_hidden = (
            recurrent_state.answer if updates_answer else recurrent_state.latent
        )
        update_halting = tracks_gradients and updates_answer
        halting_update = self._apply_halting(
            recurrent_state,
            output_state,
            previous_hidden,
            update_halting,
            observe_transition,
        )
        answer_initial = self.__get_buffer("answer_initial")
        latent_initial = self.__get_buffer("latent_initial")
        return recurrent_state.next_state_from_output(
            output_state,
            halting_update,
            updates_answer=updates_answer,
            answer_initial=answer_initial,
            latent_initial=latent_initial,
        )
