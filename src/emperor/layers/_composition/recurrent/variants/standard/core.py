from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.config import ConfigBase
from emperor.layers._composition.recurrent.base import RecurrentCompositionAbstract
from emperor.layers._composition.recurrent.config import RecurrentLayerConfig
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
from emperor.layers._composition.recurrent.validation import RecurrentLayerValidator
from emperor.layers._composition.recurrent.variants.standard.state import (
    StandardRecurrentState,
)
from emperor.layers._state import LayerState
from emperor.nn import once_per_recurrent_run

if TYPE_CHECKING:
    from emperor.layers._composition.residual.base import ResidualState


class RecurrentLayer(RecurrentCompositionAbstract):
    VALIDATOR = RecurrentLayerValidator
    supports_recurrent_diagnostics = True

    def __init__(
        self,
        cfg: RecurrentLayerConfig,
        overrides: RecurrentLayerConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.cfg: RecurrentLayerConfig
        self.max_steps: int = self.cfg.max_steps
        self.reinject_original_hidden_flag: bool = (
            self.cfg.reinject_original_hidden_flag is True
        )
        self.block_config: ConfigBase = self.cfg.block_config

        self.block_model = self._build_transition_model(self.block_config)
        self.__branched_handoff: RecurrentBranchedHandoff[StandardRecurrentState] = (
            RecurrentBranchedHandoff(self.execution)
        )
        self.__nested_handoff: RecurrentNestedHandoff[StandardRecurrentState] = (
            RecurrentNestedHandoff(self.execution)
        )

    def forward(self, state: LayerState) -> LayerState:
        self.VALIDATOR.validate_state(state, self.input_dim)
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
        recurrent_state: StandardRecurrentState,
        branch_plan: RecurrentBranchExecutionPlan,
        observe_transitions: bool = True,
    ) -> StandardRecurrentState:
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
    ) -> StandardRecurrentState:
        residual_state = self.__initialize_recurrent_residual_state(layer_state)
        return StandardRecurrentState(
            hidden=layer_state.hidden,
            fixed_input=layer_state.hidden,
            loss=branch_base_loss,
            context_state=layer_state,
            transition_index=0,
            residual_state=residual_state,
        )

    def __initialize_recurrent_residual_state(
        self,
        layer_state: LayerState,
    ) -> ResidualState | None:
        return self.utilities.initialize_residual_state(
            layer_state.hidden,
        )

    @staticmethod
    def _detach_recurrent_execution_state(
        recurrent_state: StandardRecurrentState,
    ) -> StandardRecurrentState:
        return replace(
            recurrent_state,
            hidden=recurrent_state.hidden.detach(),
            context_state=recurrent_state.context_state.detach_recurrent_context(),
        )

    def _prepare_recurrent_transition(
        self,
        recurrent_state: StandardRecurrentState,
        tracks_gradients: bool,
    ) -> PreparedRecurrentTransition:
        previous_hidden = recurrent_state.hidden
        transition_input = self.__maybe_reinject_original_hidden(
            previous_hidden,
            recurrent_state.fixed_input,
        )
        return PreparedRecurrentTransition(
            run_transition=self.block_model,
            transition_input=transition_input,
            previous_evolving_hidden=previous_hidden,
            halting_update_enabled=tracks_gradients,
            loss=recurrent_state.loss,
            residual_state=recurrent_state.residual_state,
        )

    def __maybe_reinject_original_hidden(
        self,
        hidden: Tensor,
        fixed_input: Tensor,
    ) -> Tensor:
        if not self.reinject_original_hidden_flag:
            return hidden
        return hidden + fixed_input

    @staticmethod
    def _apply_recurrent_transition_result(
        recurrent_state: StandardRecurrentState,
        transition_result: RecurrentTransitionResult,
    ) -> StandardRecurrentState:
        context_state = transition_result.context_state or recurrent_state.context_state
        return replace(
            recurrent_state,
            hidden=transition_result.hidden,
            loss=transition_result.loss,
            context_state=context_state,
            fixed_input=context_state.align_recurrent_hidden(
                recurrent_state.fixed_input,
                source_state=recurrent_state.context_state,
            ),
            transition_index=recurrent_state.transition_index + 1,
            halting_state=transition_result.halting_state,
            all_items_halted=transition_result.all_items_halted,
        )

    def _fork_recurrent_handoff_state(
        self,
        recurrent_state: StandardRecurrentState,
    ) -> StandardRecurrentState:
        return replace(
            recurrent_state,
            residual_state=self.utilities.fork_residual_state(
                recurrent_state.residual_state
            ),
        )

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
        source_boundary_state: StandardRecurrentState,
        source_plan: RecurrentBranchExecutionPlan,
    ) -> tuple[RecurrentExecutionResult, StandardRecurrentState]:
        if source_plan.starts_gradient_suffix(source_boundary_state.transition_index):
            source_boundary_state = self._detach_recurrent_execution_state(
                source_boundary_state
            )
        target_state = None
        target_residual_state = None
        transition_index = source_boundary_state.transition_index
        source_tracks_gradients = source_plan.tracks_gradients(transition_index)
        with source_plan.gradient_context(transition_index):
            prepared_transition = self._prepare_recurrent_transition(
                source_boundary_state,
                tracks_gradients=source_tracks_gradients,
            )
            if prepared_transition.residual_state is not None:
                target_state = self._fork_recurrent_handoff_state(source_boundary_state)
                with torch.no_grad():
                    target_prepared_transition = self._prepare_recurrent_transition(
                        target_state, tracks_gradients=False
                    )
                target_residual_state = target_prepared_transition.residual_state
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
                    target_residual_state=target_residual_state,
                    transition_index=transition_index,
                )
            )
        if target_state is None:
            target_state = self._fork_recurrent_handoff_state(source_boundary_state)
        target_state = self._apply_recurrent_transition_result(
            target_state, target_transition
        )
        return source_result, target_state

    def __continue_source_branch(
        self,
        source_boundary_state: StandardRecurrentState,
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
