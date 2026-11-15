from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.layers._composition.recurrent.base import RecurrentCompositionAbstract
from emperor.layers._composition.recurrent.runtime.execution import (
    PreparedRecurrentTransition,
    RecurrentExecution,
)
from emperor.layers._composition.recurrent.validation import (
    TinyRecursiveModelRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.tiny_recursive_model.state import (
    TinyRecursiveModelState,
)

if TYPE_CHECKING:
    from emperor.config import ConfigBase
    from emperor.layers._composition.recurrent.base import RecurrentTransitionResult
    from emperor.layers._composition.recurrent.config import (
        TinyRecursiveModelRecurrentConfig,
    )
    from emperor.layers._state import LayerState
    from emperor.nn import Module


class TinyRecursiveModelRecurrent(RecurrentCompositionAbstract):
    VALIDATOR = TinyRecursiveModelRecurrentValidator
    supports_recurrent_diagnostics = True

    if TYPE_CHECKING:
        answer_initial: Tensor
        latent_initial: Tensor

    def __init__(
        self,
        cfg: TinyRecursiveModelRecurrentConfig,
        overrides: TinyRecursiveModelRecurrentConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.cfg: TinyRecursiveModelRecurrentConfig
        self.block_config: ConfigBase = self.cfg.block_config
        self.latent_updates_per_answer_update: int = (
            self.cfg.latent_updates_per_answer_update
        )
        self.answer_update_count: int = self.cfg.answer_update_count
        self.initialization_standard_deviation: float = (
            self.cfg.initialization_standard_deviation
        )

        self.__register_initial_buffer("answer_initial")
        self.__register_initial_buffer("latent_initial")
        self.block_model: Module = self._build_transition_model(self.block_config)
        self.__recurrent_execution: RecurrentExecution[TinyRecursiveModelState] = (
            RecurrentExecution()
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
            answer_initial=self.answer_initial,
            latent_initial=self.latent_initial,
            expected_feature_dim=self.output_dim,
        )

        return self.__recurrent_execution.execute(
            self,
            state,
            self.recurrent_iteration_schedule,
        )

    def _initialize_recurrent_execution_state(
        self,
        layer_state: LayerState,
        branch_base_loss: Tensor | None,
    ) -> TinyRecursiveModelState:
        fixed_input = layer_state.hidden
        return TinyRecursiveModelState(
            fixed_input=fixed_input,
            answer=self._expand_recurrent_initial(
                self.answer_initial,
                fixed_input,
            ),
            latent=self._expand_recurrent_initial(
                self.latent_initial,
                fixed_input,
            ),
            initial_loss=branch_base_loss,
            auxiliary_losses=(),
            context_state=layer_state,
            transition_index=0,
        )

    @staticmethod
    def _detach_recurrent_execution_state(
        recurrent_state: TinyRecursiveModelState,
    ) -> TinyRecursiveModelState:
        return replace(
            recurrent_state,
            answer=recurrent_state.answer.detach(),
            latent=recurrent_state.latent.detach(),
            context_state=recurrent_state.context_state.detach_recurrent_context(),
        )

    def _prepare_recurrent_transition(
        self,
        recurrent_state: TinyRecursiveModelState,
        tracks_gradients: bool,
    ) -> PreparedRecurrentTransition:
        if self.__updates_answer(recurrent_state):
            return PreparedRecurrentTransition(
                run_transition=self.block_model,
                transition_input=recurrent_state.answer + recurrent_state.latent,
                previous_evolving_hidden=recurrent_state.answer,
                halting_update_enabled=tracks_gradients,
            )
        return PreparedRecurrentTransition(
            run_transition=self.block_model,
            transition_input=(
                recurrent_state.latent
                + recurrent_state.answer
                + recurrent_state.fixed_input
            ),
            previous_evolving_hidden=recurrent_state.latent,
            halting_update_enabled=False,
        )

    def _apply_recurrent_transition_result(
        self,
        recurrent_state: TinyRecursiveModelState,
        transition_result: RecurrentTransitionResult,
    ) -> TinyRecursiveModelState:
        auxiliary_losses = self.__with_reduced_transition_loss(
            recurrent_state.auxiliary_losses,
            transition_result.loss,
        )
        context_state = transition_result.context_state or recurrent_state.context_state
        answer = context_state.align_recurrent_hidden(
            recurrent_state.answer,
            initial=self.answer_initial,
            source_state=recurrent_state.context_state,
        )
        latent = context_state.align_recurrent_hidden(
            recurrent_state.latent,
            initial=self.latent_initial,
            source_state=recurrent_state.context_state,
        )
        if self.__updates_answer(recurrent_state):
            answer = transition_result.hidden
        else:
            latent = transition_result.hidden
        return replace(
            recurrent_state,
            answer=answer,
            latent=latent,
            context_state=context_state,
            fixed_input=context_state.align_recurrent_hidden(
                recurrent_state.fixed_input,
                source_state=recurrent_state.context_state,
            ),
            auxiliary_losses=auxiliary_losses,
            transition_index=recurrent_state.transition_index + 1,
            halting_state=transition_result.halting_state,
            all_items_halted=transition_result.all_items_halted,
        )

    def __with_reduced_transition_loss(
        self,
        auxiliary_losses: tuple[Tensor, ...],
        transition_loss: Tensor | None,
    ) -> tuple[Tensor, ...]:
        if transition_loss is None:
            return auxiliary_losses
        return (*auxiliary_losses, self._reduce_auxiliary_loss(transition_loss))

    @staticmethod
    def _fork_recurrent_handoff_state(
        recurrent_state: TinyRecursiveModelState,
    ) -> TinyRecursiveModelState:
        return replace(recurrent_state)

    def __updates_answer(
        self,
        recurrent_state: TinyRecursiveModelState,
    ) -> bool:
        transitions_per_iteration = (
            self.recurrent_iteration_schedule.transitions_per_iteration
        )
        phase_index = recurrent_state.transition_index % transitions_per_iteration
        return phase_index >= self.latent_updates_per_answer_update
