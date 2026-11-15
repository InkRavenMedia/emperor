from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.layers._composition.recurrent.base import RecurrentCompositionAbstract
from emperor.layers._composition.recurrent.runtime.execution import (
    PreparedRecurrentTransition,
    RecurrentExecutionResult,
    RecurrentTransitionResult,
)
from emperor.layers._composition.recurrent.runtime.execution.handoff import (
    RecurrentBranchedHandoff,
    RecurrentNestedHandoff,
)
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentBranchExecutionPlan,
    RecurrentNestedSmoothHandoffExecutionPlan,
    RecurrentSmoothHandoffExecutionPlan,
)
from emperor.layers._composition.recurrent.validation import (
    TinyRecursiveModelRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.tiny_recursive_model.state import (
    TinyRecursiveModelState,
)
from emperor.nn import once_per_recurrent_run

if TYPE_CHECKING:
    from emperor.config import ConfigBase
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
        self.__branched_handoff: RecurrentBranchedHandoff[TinyRecursiveModelState] = (
            RecurrentBranchedHandoff(self.execution)
        )
        self.__nested_handoff: RecurrentNestedHandoff[TinyRecursiveModelState] = (
            RecurrentNestedHandoff(self.execution)
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

        with once_per_recurrent_run():
            execution_plan = self.recurrent_iteration_schedule.execution_plan()
            if isinstance(
                execution_plan,
                (
                    RecurrentSmoothHandoffExecutionPlan,
                    RecurrentNestedSmoothHandoffExecutionPlan,
                ),
            ):
                return self.__run_smooth_handoff(state, execution_plan)

            recurrent_state = self._initialize_recurrent_execution_state(
                state, branch_base_loss=state.loss
            )
            recurrent_state = self.run_branch(
                recurrent_state, execution_plan.target_branch
            )
            execution_result = self.execution.finalize(recurrent_state)
            return self.execution.commit(
                state, self.recurrent_iteration_schedule, execution_result
            )

    def run_branch(
        self,
        recurrent_state: TinyRecursiveModelState,
        branch_plan: RecurrentBranchExecutionPlan,
        observe_transitions: bool = True,
    ) -> TinyRecursiveModelState:
        while (
            recurrent_state.transition_index < branch_plan.transition_count
            and not recurrent_state.all_items_halted
        ):
            if branch_plan.starts_gradient_suffix(recurrent_state.transition_index):
                recurrent_state = self._detach_recurrent_execution_state(
                    recurrent_state
                )
            transition_index = recurrent_state.transition_index
            tracks_gradients = branch_plan.tracks_gradients(transition_index)
            with branch_plan.gradient_context(transition_index):
                prepared_transition = self._prepare_recurrent_transition(
                    recurrent_state,
                    tracks_gradients=tracks_gradients,
                )
                transition_result = self.execution.run_transition(
                    recurrent_state,
                    prepared_transition,
                    validate_output=self.VALIDATOR.validate_transition_output,
                    transition_index=transition_index,
                    observe_transition=observe_transitions,
                )
                recurrent_state = self._apply_recurrent_transition_result(
                    recurrent_state, transition_result
                )
        return recurrent_state

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

    def __run_smooth_handoff(
        self,
        state: LayerState,
        execution_plan: RecurrentSmoothHandoffExecutionPlan
        | RecurrentNestedSmoothHandoffExecutionPlan,
    ) -> LayerState:
        with self.execution.handoff(state, self.modules()):
            recurrent_state = self._initialize_recurrent_execution_state(
                state, branch_base_loss=None
            )
            if isinstance(execution_plan, RecurrentNestedSmoothHandoffExecutionPlan):
                execution_result = self.__nested_handoff.execute(
                    recurrent_state,
                    state,
                    execution_plan,
                    run_branch=self.run_branch,
                )
            else:
                execution_result = self.__branched_handoff.execute(
                    recurrent_state,
                    state,
                    execution_plan,
                    run_branch=self.run_branch,
                    run_boundary=self.run_handoff_boundary,
                )
            return self.execution.commit(
                state, self.recurrent_iteration_schedule, execution_result
            )

    def run_handoff_boundary(
        self,
        source_boundary_state: TinyRecursiveModelState,
        source_plan: RecurrentBranchExecutionPlan,
    ) -> tuple[RecurrentExecutionResult, TinyRecursiveModelState]:
        if source_plan.starts_gradient_suffix(source_boundary_state.transition_index):
            source_boundary_state = self._detach_recurrent_execution_state(
                source_boundary_state
            )
        transition_index = source_boundary_state.transition_index
        source_tracks_gradients = source_plan.tracks_gradients(transition_index)
        with source_plan.gradient_context(transition_index):
            prepared_transition = self._prepare_recurrent_transition(
                source_boundary_state,
                tracks_gradients=source_tracks_gradients,
            )
            source_result, target_transition = (
                self.__branched_handoff.run_shared_transition(
                    source_boundary_state,
                    prepared_transition,
                    run_source_branch=partial(
                        self.__continue_source_branch,
                        source_boundary_state,
                        source_plan,
                    ),
                    runtime_modules=self.modules(),
                    validate_output=self.VALIDATOR.validate_transition_output,
                    transition_index=transition_index,
                )
            )
        target_state = self._fork_recurrent_handoff_state(source_boundary_state)
        target_state = self._apply_recurrent_transition_result(
            target_state, target_transition
        )
        return source_result, target_state

    def __continue_source_branch(
        self,
        source_boundary_state: TinyRecursiveModelState,
        source_plan: RecurrentBranchExecutionPlan,
        transition_result: RecurrentTransitionResult,
    ) -> RecurrentExecutionResult:
        source_state = self._fork_recurrent_handoff_state(source_boundary_state)
        source_state = self._apply_recurrent_transition_result(
            source_state, transition_result
        )
        source_state = self.run_branch(
            source_state, source_plan, observe_transitions=False
        )
        return self.execution.finalize(source_state)
