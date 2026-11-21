from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

import torch

from emperor.layers._composition.recurrent.base import RecurrentCompositionAbstract
from emperor.layers._composition.recurrent.runtime.handoff import RecurrentDepthHandoff
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentIterationSchedule,
)
from emperor.layers._composition.recurrent.validation import (
    HierarchicalReasoningModelRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model.step import (
    HierarchicalReasoningModelStep,
)
from emperor.nn import once_per_recurrent_run

if TYPE_CHECKING:
    from emperor.config import ConfigBase
    from emperor.layers._composition.recurrent.config import (
        HierarchicalReasoningModelRecurrentConfig,
    )
    from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model import (
        HierarchicalReasoningModelState,
    )
    from emperor.layers._state import LayerState


class HierarchicalReasoningModelRecurrent(RecurrentCompositionAbstract):
    VALIDATOR = HierarchicalReasoningModelRecurrentValidator
    _STEP_OWNED_NAMES = (
        "pipeline",
        "high_model",
        "low_model",
        "high_initial",
        "low_initial",
        "initialization_standard_deviation",
    )
    supports_recurrent_diagnostics = True

    def __init__(
        self,
        cfg: HierarchicalReasoningModelRecurrentConfig,
        overrides: HierarchicalReasoningModelRecurrentConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.cfg: HierarchicalReasoningModelRecurrentConfig
        self.recurrent_iteration_schedule = RecurrentIterationSchedule(self.cfg)
        self.high_block_config: ConfigBase = self.cfg.high_block_config
        self.low_block_config: ConfigBase = self.cfg.low_block_config
        self.high_cycles: int = self.cfg.high_cycles

        self.step = self.__build_step()
        self.low_cycles: int = self.step.low_cycles
        self.handoff = self.__build_handoff()

    def __build_step(self) -> HierarchicalReasoningModelStep:
        return HierarchicalReasoningModelStep(self.cfg)

    def __build_handoff(self) -> RecurrentDepthHandoff[HierarchicalReasoningModelState]:
        return RecurrentDepthHandoff(self.step, self.recurrent_iteration_schedule)

    def forward(self, state: LayerState) -> LayerState:
        self.VALIDATOR.validate_state(state, self.input_dim)

        with once_per_recurrent_run():
            if self.recurrent_iteration_schedule.transitioning:
                return self.handoff.run_and_blend_depth_results(
                    state, runtime_modules=self.modules
                )
            updated_state = self.__run_recurrence_with_step_limit(state)
            self.recurrent_iteration_schedule.record_successful_forward()
            return updated_state

    def __run_recurrence_with_step_limit(self, state: LayerState) -> LayerState:
        recurrent_state = self.step.initialize_recurrent_state(state)
        while self.__has_remaining_steps_and_active_items(recurrent_state):
            recurrent_state = self.__maybe_detach_recurrent_state(recurrent_state)
            tracks_gradients = self.__should_track_gradients(recurrent_state)
            with self.__gradient_context_for_step(tracks_gradients):
                recurrent_state = self.step(recurrent_state, tracks_gradients)
        output_state = self.step.finalize(recurrent_state)
        return self.__update_state_from_output(state, output_state)

    def __has_remaining_steps_and_active_items(
        self, recurrent_state: HierarchicalReasoningModelState
    ) -> bool:
        step_limit = self.recurrent_iteration_schedule.active_transition_count
        has_remaining_steps = recurrent_state.transition_index < step_limit
        has_active_items = not recurrent_state.all_items_halted
        return has_remaining_steps and has_active_items

    def __maybe_detach_recurrent_state(
        self, recurrent_state: HierarchicalReasoningModelState
    ) -> HierarchicalReasoningModelState:
        if self.__is_first_step_after_no_gradient_prefix(recurrent_state):
            recurrent_state = recurrent_state.detach()
        return recurrent_state

    def __is_first_step_after_no_gradient_prefix(
        self, recurrent_state: HierarchicalReasoningModelState
    ) -> bool:
        no_gradient_steps = (
            self.recurrent_iteration_schedule.no_gradient_transition_count
        )
        has_steps_without_gradients = no_gradient_steps > 0
        is_first_gradient_step = recurrent_state.transition_index == no_gradient_steps
        return has_steps_without_gradients and is_first_gradient_step

    def __should_track_gradients(
        self, recurrent_state: HierarchicalReasoningModelState
    ) -> bool:
        return (
            recurrent_state.transition_index
            >= self.recurrent_iteration_schedule.no_gradient_transition_count
        )

    def __gradient_context_for_step(
        self, tracks_gradients: bool
    ) -> AbstractContextManager[None]:
        if tracks_gradients:
            return nullcontext()
        return torch.no_grad()

    def __update_state_from_output(
        self, state: LayerState, output_state: LayerState
    ) -> LayerState:
        state.hidden = output_state.hidden
        state.loss = output_state.loss
        state.commit_recurrent_context(output_state)
        return state
