from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar

import torch
from torch import Tensor, nn

from emperor.layers._composition.recurrent.runtime.execution import (
    RecurrentTransitionExecution,
)
from emperor.layers._composition.recurrent.runtime.execution.interface import (
    RecurrentTransitionResult as RecurrentTransitionResult,
)
from emperor.layers._composition.recurrent.runtime.execution.runtime_state import (
    RecurrentRuntimeStateGuard,
)
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentIterationSchedule,
)
from emperor.layers._composition.recurrent.utilities import RecurrentUtilities
from emperor.layers._support import LayerModuleBase

if TYPE_CHECKING:
    from collections.abc import Callable

    from emperor.config import ConfigBase
    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.recurrent.config import (
        RecurrentCompositionConfig,
    )
    from emperor.layers._composition.recurrent.runtime.execution.interface import (
        PreparedRecurrentTransition,
        RecurrentExecutionState,
        RecurrentTransitionCandidate,
    )
    from emperor.layers._composition.residual.base import (
        ResidualState,
    )
    from emperor.layers._state import LayerState
    from emperor.nn import Module


_ProvisionalSourceBranchOutput = TypeVar("_ProvisionalSourceBranchOutput")


class RecurrentCompositionAbstract(LayerModuleBase, ABC):
    supports_recurrent_diagnostics = False
    __runtime_state_guard = RecurrentRuntimeStateGuard()

    def __init__(
        self,
        cfg: RecurrentCompositionConfig,
        overrides: RecurrentCompositionConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: RecurrentCompositionConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.recurrent_iteration_schedule = RecurrentIterationSchedule(self.cfg)
        self.utilities = RecurrentUtilities(self.cfg)
        self.execution = RecurrentTransitionExecution(self.utilities, self.output_dim)

    def _build_transition_model(self, transition_config: ConfigBase) -> Module:
        return self._build_from_config(
            transition_config,
            input_dim=self.output_dim,
            output_dim=self.output_dim,
        )

    def _new_recurrent_initial_buffer(self, standard_deviation: float) -> Tensor:
        initial_buffer = torch.empty(self.output_dim)
        if standard_deviation == 0:
            return initial_buffer.zero_()
        return nn.init.trunc_normal_(
            initial_buffer,
            mean=0.0,
            std=standard_deviation,
            a=-2.0 * standard_deviation,
            b=2.0 * standard_deviation,
        )

    @staticmethod
    def _expand_recurrent_initial(initial: Tensor, hidden: Tensor) -> Tensor:
        initial_broadcast_shape = (1,) * (hidden.ndim - 1) + (hidden.shape[-1],)
        return initial.view(initial_broadcast_shape).expand_as(hidden)

    def _run_recurrent_transition(
        self,
        recurrent_state: RecurrentExecutionState,
        prepared_transition: PreparedRecurrentTransition,
        transition_index: int = 0,
        observe_transition: bool = True,
    ) -> RecurrentTransitionResult:
        return self.execution.run_transition(
            recurrent_state,
            prepared_transition,
            self.VALIDATOR.validate_transition_output,
            transition_index,
            observe_transition,
        )

    def _run_shared_handoff_boundary_transition(
        self,
        recurrent_state: RecurrentExecutionState,
        prepared_transition: PreparedRecurrentTransition,
        run_provisional_source_branch: Callable[
            [RecurrentTransitionResult],
            _ProvisionalSourceBranchOutput,
        ],
        target_residual_state: ResidualState | None = None,
        transition_index: int = 0,
    ) -> tuple[_ProvisionalSourceBranchOutput, RecurrentTransitionResult]:
        """Share one candidate, roll back the full source branch, then commit target."""
        transition_candidate = self.execution.compute_transition(
            recurrent_state,
            prepared_transition,
            self.VALIDATOR.validate_transition_output,
            transition_index=transition_index,
        )
        self.utilities.advance_detached_target_residual_state(
            transition_candidate.residual_input,
            prepared_transition.previous_evolving_hidden,
            target_residual_state,
            transition_index,
        )
        with self.__runtime_state_guard.isolate_provisional_branch(self):
            source_result = self.execution.complete_transition(
                recurrent_state,
                transition_candidate,
                previous_evolving_hidden=prepared_transition.previous_evolving_hidden,
                halting_update_enabled=prepared_transition.halting_update_enabled,
                observe_transition=False,
            )
            provisional_source_branch_output = run_provisional_source_branch(
                source_result
            )
        target_candidate = self.__detach_recurrent_transition_candidate(
            transition_candidate
        )
        target_result = self.execution.complete_transition(
            recurrent_state,
            target_candidate,
            previous_evolving_hidden=prepared_transition.previous_evolving_hidden,
            halting_update_enabled=False,
            observe_transition=True,
        )
        return provisional_source_branch_output, target_result

    @staticmethod
    def __detach_recurrent_transition_candidate(
        transition_candidate: RecurrentTransitionCandidate,
    ) -> RecurrentTransitionCandidate:
        """Detach the target continuation after the provisional source has finished."""
        return replace(
            transition_candidate,
            hidden=transition_candidate.hidden.detach(),
            loss=(
                None
                if transition_candidate.loss is None
                else transition_candidate.loss.detach()
            ),
            residual_input=transition_candidate.residual_input.detach(),
            context_state=transition_candidate.context_state.detach_recurrent_context(),
        )

    def _finalize_recurrent_halting(
        self,
        hidden: Tensor,
        loss: Tensor | None,
        halting_state: HaltingStateBase | None,
    ) -> tuple[Tensor, Tensor | None]:
        return self.utilities.finalize_halting(hidden, loss, halting_state)

    @abstractmethod
    def forward(self, state: LayerState) -> LayerState:
        pass
