from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.layers._composition.recurrent.base import RecurrentCompositionAbstract
from emperor.layers._composition.recurrent.runtime.handoff import RecurrentDepthHandoff
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentIterationSchedule,
)
from emperor.layers._composition.recurrent.utilities import RecurrentUtilities
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
    from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model.state import (
        HierarchicalReasoningModelState,
    )
    from emperor.layers._state import LayerState
    from emperor.nn import Module


class HierarchicalReasoningModelRecurrent(RecurrentCompositionAbstract):
    VALIDATOR = HierarchicalReasoningModelRecurrentValidator
    supports_recurrent_diagnostics = True

    if TYPE_CHECKING:
        high_initial: Tensor
        low_initial: Tensor

    def __init__(
        self,
        cfg: HierarchicalReasoningModelRecurrentConfig,
        overrides: HierarchicalReasoningModelRecurrentConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.cfg: HierarchicalReasoningModelRecurrentConfig
        self.recurrent_iteration_schedule = RecurrentIterationSchedule(self.cfg)
        self.utilities = RecurrentUtilities(self.cfg)
        self.high_block_config: ConfigBase = self.cfg.high_block_config
        self.low_block_config: ConfigBase = self.cfg.low_block_config
        self.high_cycles: int = self.cfg.high_cycles
        self.low_cycles: int = self.cfg.low_cycles
        self.initialization_standard_deviation: float = (
            self.cfg.initialization_standard_deviation
        )

        self.__register_initial_buffer("high_initial")
        self.__register_initial_buffer("low_initial")
        self.high_model: Module = self._build_transition_model(self.high_block_config)
        self.low_model: Module = self._build_transition_model(self.low_block_config)
        self.step = HierarchicalReasoningModelStep(
            utilities=self.utilities,
            output_dim=self.output_dim,
            get_module=self.get_submodule,
            get_buffer=self.get_buffer,
            low_cycles=self.low_cycles,
        )
        self.handoff = RecurrentDepthHandoff(
            self.step, self.recurrent_iteration_schedule
        )

    def __register_initial_buffer(self, buffer_name: str) -> None:
        initial_buffer = self._new_recurrent_initial_buffer(
            self.initialization_standard_deviation
        )
        self.register_buffer(buffer_name, initial_buffer, persistent=True)

    def forward(self, state: LayerState) -> LayerState:
        self.VALIDATOR.validate_state(state, self.input_dim)
        self.VALIDATOR.validate_initial_buffers(
            state.hidden,
            high_initial=self.high_initial,
            low_initial=self.low_initial,
            expected_feature_dim=self.output_dim,
        )

        with once_per_recurrent_run():
            if self.recurrent_iteration_schedule.transitioning:
                return self.handoff.run_and_blend_depth_results(
                    state, runtime_modules=self.modules
                )
            step_limit = self.recurrent_iteration_schedule.active_transition_count
            no_gradient_steps = (
                self.recurrent_iteration_schedule.no_gradient_transition_count
            )
            return self.__run_recurrence_with_step_limit(
                state, step_limit, no_gradient_steps
            )

    def __run_recurrence_with_step_limit(
        self, state: LayerState, step_limit: int, no_gradient_steps: int
    ) -> LayerState:
        recurrent_state = self.step.initialize(state)
        while self.__has_remaining_steps_and_active_items(recurrent_state, step_limit):
            if self.__is_first_step_after_no_gradient_prefix(
                recurrent_state, no_gradient_steps
            ):
                recurrent_state = recurrent_state.detach()
            tracks_gradients = recurrent_state.transition_index >= no_gradient_steps
            with self.__gradient_context_for_step(tracks_gradients):
                recurrent_state = self.step.forward(recurrent_state, tracks_gradients)
        output_state = self.step.finalize(recurrent_state)
        state.hidden = output_state.hidden
        state.loss = output_state.loss
        state.commit_recurrent_context(output_state)
        self.recurrent_iteration_schedule.record_successful_forward()
        return state

    def __has_remaining_steps_and_active_items(
        self, recurrent_state: HierarchicalReasoningModelState, step_limit: int
    ) -> bool:
        return (
            recurrent_state.transition_index < step_limit
            and not recurrent_state.all_items_halted
        )

    def __is_first_step_after_no_gradient_prefix(
        self, recurrent_state: HierarchicalReasoningModelState, no_gradient_steps: int
    ) -> bool:
        return (
            no_gradient_steps > 0
            and recurrent_state.transition_index == no_gradient_steps
        )

    def __gradient_context_for_step(
        self, tracks_gradients: bool
    ) -> AbstractContextManager[None]:
        if tracks_gradients:
            return nullcontext()
        return torch.no_grad()
