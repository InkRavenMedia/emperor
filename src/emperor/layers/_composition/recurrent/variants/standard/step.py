from __future__ import annotations

from typing import TYPE_CHECKING

from emperor.layers._composition.recurrent.runtime.step import RecurrentStep
from emperor.layers._composition.recurrent.validation import RecurrentLayerValidator
from emperor.layers._composition.recurrent.variants.standard.state import (
    StandardRecurrentState,
)

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.layers._composition.recurrent.config import RecurrentLayerConfig
    from emperor.layers._state import LayerState


class StandardRecurrentStep(RecurrentStep[StandardRecurrentState]):
    VALIDATOR = RecurrentLayerValidator

    def __init__(self, cfg: RecurrentLayerConfig) -> None:
        super().__init__(cfg)
        self.reinject_original_hidden = cfg.reinject_original_hidden_flag is True
        self.block_model = self._build_transition_model(cfg.block_config)

    def initialize_recurrent_state(
        self, state: LayerState, include_caller_loss: bool = True
    ) -> StandardRecurrentState:
        residual_state = self.pipeline.new_recurrent_residual_state(state.hidden)
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
        transition_input = self._maybe_reinject_original_input(recurrent_state)
        transition_state = recurrent_state.layer_state_for_transition(transition_input)
        return self.pipeline.process_recurrent_transition(
            transition_state,
            run_model=self._run_block,
            recurrent_state=recurrent_state,
            target_state=target_state,
        )

    def _maybe_reinject_original_input(
        self, recurrent_state: StandardRecurrentState
    ) -> Tensor:
        if self.reinject_original_hidden:
            return recurrent_state.hidden + recurrent_state.fixed_input
        return recurrent_state.hidden

    def _run_block(
        self, state: LayerState, *, complete_prefix: bool = False
    ) -> LayerState:
        block_input = state.hidden
        output_state = self._call_transition_block(
            self.block_model, state, complete_prefix=complete_prefix
        )
        self.VALIDATOR.validate_transition_output(
            output_state, block_input, self.cfg.output_dim
        )
        return output_state

    def update_state(
        self,
        recurrent_state: StandardRecurrentState,
        output_state: LayerState,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> StandardRecurrentState:
        halting_update = self._apply_halting(
            recurrent_state,
            output_state,
            recurrent_state.hidden,
            tracks_gradients,
            observe_transition,
        )
        return recurrent_state.next_state_from_output(output_state, halting_update)

    def fork(self, recurrent_state: StandardRecurrentState) -> StandardRecurrentState:
        forked_residual_state = self.pipeline.fork_recurrent_residual_state(
            recurrent_state.residual_state
        )
        return recurrent_state.with_forked_residual_state(forked_residual_state)
