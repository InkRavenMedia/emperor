from __future__ import annotations

from typing import TYPE_CHECKING

from emperor.layers._composition.recurrent.runtime.step import RecurrentStep
from emperor.layers._composition.recurrent.state import new_recurrent_initial_buffer
from emperor.layers._composition.recurrent.validation import (
    HierarchicalReasoningModelRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model.state import (
    HierarchicalReasoningModelState,
)

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.layers._composition.recurrent.config import (
        HierarchicalReasoningModelRecurrentConfig,
    )
    from emperor.layers._state import LayerState
    from emperor.nn import Module


class HierarchicalReasoningModelStep(RecurrentStep[HierarchicalReasoningModelState]):
    VALIDATOR = HierarchicalReasoningModelRecurrentValidator

    if TYPE_CHECKING:
        high_initial: Tensor
        low_initial: Tensor

    def __init__(self, cfg: HierarchicalReasoningModelRecurrentConfig) -> None:
        super().__init__(cfg)
        self.low_cycles: int = cfg.low_cycles
        self.initialization_standard_deviation: float = (
            cfg.initialization_standard_deviation
        )
        self.__register_initial_buffers()
        self.high_model = self._build_transition_model(cfg.high_block_config)
        self.low_model = self._build_transition_model(cfg.low_block_config)

    def __register_initial_buffers(self) -> None:
        for buffer_name in ("high_initial", "low_initial"):
            initial_buffer = new_recurrent_initial_buffer(
                self.cfg.output_dim, self.initialization_standard_deviation
            )
            self.register_buffer(buffer_name, initial_buffer, persistent=True)

    def initialize_recurrent_state(
        self, state: LayerState, include_caller_loss: bool = True
    ) -> HierarchicalReasoningModelState:
        self.VALIDATOR.validate_initial_buffers(
            state.hidden,
            high_initial=self.high_initial,
            low_initial=self.low_initial,
            expected_feature_dim=self.cfg.output_dim,
        )
        initial_shape = (1,) * (state.hidden.ndim - 1) + (state.hidden.shape[-1],)
        high = self.high_initial.view(initial_shape).expand_as(state.hidden)
        low = self.low_initial.view(initial_shape).expand_as(state.hidden)
        return HierarchicalReasoningModelState(
            fixed_input=state.hidden,
            high=high,
            low=low,
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
        updates_high = self.__updates_high(recurrent_state.transition_index)
        if updates_high:
            transition_input = recurrent_state.high + recurrent_state.low
            previous_hidden = recurrent_state.high
        else:
            low_input = recurrent_state.low + recurrent_state.high
            transition_input = low_input + recurrent_state.fixed_input
            previous_hidden = recurrent_state.low
        transition_state = recurrent_state.layer_state_for_transition(transition_input)
        block = self.high_model if updates_high else self.low_model
        return self.pipeline.process_recurrent_transition(
            transition_state,
            run_model=lambda state: self.__run_block(state, block),
            recurrent_state=recurrent_state,
            previous_hidden=previous_hidden,
        )

    def __updates_high(self, transition_index: int) -> bool:
        phase_index = transition_index % (self.low_cycles + 1)
        return phase_index >= self.low_cycles

    def __run_block(self, state: LayerState, block: Module) -> LayerState:
        block_input = state.hidden
        output_state = self._call_transition_block(block, state)
        self.VALIDATOR.validate_transition_output(
            output_state, block_input, self.cfg.output_dim
        )
        return output_state

    def update_state(
        self,
        recurrent_state: HierarchicalReasoningModelState,
        output_state: LayerState,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> HierarchicalReasoningModelState:
        updates_high = self.__updates_high(recurrent_state.transition_index)
        previous_hidden = recurrent_state.high if updates_high else recurrent_state.low
        update_halting = tracks_gradients and updates_high
        halting_update = self._apply_halting(
            recurrent_state,
            output_state,
            previous_hidden,
            update_halting,
            observe_transition,
        )
        return recurrent_state.next_state_from_output(
            output_state,
            halting_update,
            updates_high=updates_high,
            high_initial=self.high_initial,
            low_initial=self.low_initial,
        )
