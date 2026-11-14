from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, TypeVar

import torch
from torch import Tensor, nn

from emperor.layers._composition.recurrent.runtime.execution.runtime_state import (
    RecurrentRuntimeStateGuard,
)
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentIterationSchedule,
)
from emperor.layers._composition.recurrent.utilities import RecurrentUtilities
from emperor.layers._support import LayerModuleBase
from emperor.memory import MemoryPositionOptions

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from emperor.config import ConfigBase
    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.recurrent.config import (
        RecurrentCompositionConfig,
    )
    from emperor.layers._composition.recurrent.runtime.execution.interface import (
        PreparedRecurrentTransition,
    )
    from emperor.layers._composition.residual.base import (
        ResidualState,
    )
    from emperor.layers._state import LayerState
    from emperor.nn import Module


class _RecurrentTransitionContext(Protocol):
    """Forward-local state required by the shared transition pipeline."""

    @property
    def context_state(self) -> LayerState: ...

    @property
    def halting_state(self) -> HaltingStateBase | None: ...


@dataclass(frozen=True)
class RecurrentTransitionResult:
    hidden: Tensor
    loss: Tensor | None
    halting_state: HaltingStateBase | None
    all_items_halted: bool
    context_state: LayerState | None = None


@dataclass(frozen=True)
class _RecurrentTransitionCandidate:
    hidden: Tensor
    loss: Tensor | None
    residual_input: Tensor
    context_state: LayerState


_ProvisionalSourceBranchOutput = TypeVar("_ProvisionalSourceBranchOutput")


class RecurrentCompositionAbstract(LayerModuleBase, ABC):
    """Build shared controllers and execute individual recurrent transitions."""

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
        self._recurrent_diagnostic_observer: Callable[[Tensor, Tensor], None] | None = (
            None
        )
        self.__recurrent_diagnostic_observation_suppression_depth = 0

    def _build_transition_model(self, transition_config: ConfigBase) -> Module:
        return self._build_from_config(
            transition_config,
            input_dim=self.output_dim,
            output_dim=self.output_dim,
        )

    def _set_recurrent_diagnostic_observer(
        self,
        observer: Callable[[Tensor, Tensor], None] | None,
    ) -> None:
        self._recurrent_diagnostic_observer = observer

    def _observe_recurrent_step(
        self,
        previous_hidden: Tensor,
        output_hidden: Tensor,
    ) -> None:
        observer = self._recurrent_diagnostic_observer
        if observer is not None:
            observer(previous_hidden, output_hidden)

    @property
    def _recurrent_diagnostic_observation_enabled(self) -> bool:
        return self.__recurrent_diagnostic_observation_suppression_depth == 0

    @contextmanager
    def __recurrent_diagnostic_observation_context(
        self,
        enabled: bool,
    ) -> Generator[None, None, None]:
        if enabled:
            yield
            return
        self.__recurrent_diagnostic_observation_suppression_depth += 1
        try:
            yield
        finally:
            self.__recurrent_diagnostic_observation_suppression_depth -= 1

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
        recurrent_state: _RecurrentTransitionContext,
        prepared_transition: PreparedRecurrentTransition,
        transition_index: int = 0,
        observe_transition: bool = True,
    ) -> RecurrentTransitionResult:
        with self.__recurrent_diagnostic_observation_context(
            enabled=observe_transition,
        ):
            transition_candidate = self.__compute_recurrent_transition_candidate(
                recurrent_state,
                prepared_transition,
                transition_index=transition_index,
            )
        return self.__apply_halting_and_observe_recurrent_transition(
            recurrent_state,
            transition_candidate,
            previous_evolving_hidden=prepared_transition.previous_evolving_hidden,
            halting_update_enabled=prepared_transition.halting_update_enabled,
            observe_transition=observe_transition,
        )

    def _run_shared_handoff_boundary_transition(
        self,
        recurrent_state: _RecurrentTransitionContext,
        prepared_transition: PreparedRecurrentTransition,
        run_provisional_source_branch: Callable[
            [RecurrentTransitionResult],
            _ProvisionalSourceBranchOutput,
        ],
        target_residual_state: ResidualState | None = None,
        transition_index: int = 0,
    ) -> tuple[_ProvisionalSourceBranchOutput, RecurrentTransitionResult]:
        """Share one candidate, roll back the full source branch, then commit target."""
        transition_candidate = self.__compute_recurrent_transition_candidate(
            recurrent_state,
            prepared_transition,
            transition_index=transition_index,
        )
        self.utilities.advance_detached_target_residual_state(
            transition_candidate.residual_input,
            prepared_transition.previous_evolving_hidden,
            target_residual_state,
            transition_index,
        )
        with self.__runtime_state_guard.isolate_provisional_branch(self):
            source_result = self.__apply_halting_and_observe_recurrent_transition(
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
        target_result = self.__apply_halting_and_observe_recurrent_transition(
            recurrent_state,
            target_candidate,
            previous_evolving_hidden=prepared_transition.previous_evolving_hidden,
            halting_update_enabled=False,
            observe_transition=True,
        )
        return provisional_source_branch_output, target_result

    @staticmethod
    def __detach_recurrent_transition_candidate(
        transition_candidate: _RecurrentTransitionCandidate,
    ) -> _RecurrentTransitionCandidate:
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

    def __compute_recurrent_transition_candidate(
        self,
        recurrent_state: _RecurrentTransitionContext,
        prepared_transition: PreparedRecurrentTransition,
        transition_index: int,
    ) -> _RecurrentTransitionCandidate:
        transition_state = replace(
            recurrent_state.context_state,
            hidden=prepared_transition.transition_input,
            loss=prepared_transition.loss,
            halting_state=None,
        )
        transition_state = self.utilities.normalize_before_model(transition_state)
        transition_model_input = self.utilities.apply_memory(
            transition_state.hidden,
            MemoryPositionOptions.BEFORE_AFFINE,
        )
        transition_state.hidden = transition_model_input
        output_state = prepared_transition.run_transition(transition_state)
        self.VALIDATOR.validate_transition_output(
            output_state,
            transition_model_input,
            expected_feature_dim=self.output_dim,
        )
        candidate_hidden = self.utilities.apply_memory(
            output_state.hidden,
            MemoryPositionOptions.AFTER_AFFINE,
        )
        candidate_state = replace(output_state, hidden=candidate_hidden)
        candidate_state = self.utilities.normalize_after_model(candidate_state)
        candidate_state.hidden = self.utilities.apply_gate(candidate_state.hidden)
        residual_input = candidate_state.hidden
        aligned_previous_hidden = output_state.align_recurrent_hidden(
            prepared_transition.previous_evolving_hidden,
            source_state=recurrent_state.context_state,
        )
        candidate_state.hidden = self.utilities.apply_residual(
            candidate_state.hidden,
            aligned_previous_hidden,
            prepared_transition.residual_state,
            transition_index,
        )
        candidate_state = self.utilities.normalize_after_residual(candidate_state)
        return _RecurrentTransitionCandidate(
            hidden=candidate_state.hidden,
            loss=output_state.loss,
            residual_input=residual_input,
            context_state=candidate_state,
        )

    def __apply_halting_and_observe_recurrent_transition(
        self,
        recurrent_state: _RecurrentTransitionContext,
        transition_candidate: _RecurrentTransitionCandidate,
        previous_evolving_hidden: Tensor,
        halting_update_enabled: bool,
        observe_transition: bool,
    ) -> RecurrentTransitionResult:
        halting_state, output_hidden = self.utilities.apply_halting(
            recurrent_state.halting_state,
            transition_candidate.hidden,
            update_enabled=halting_update_enabled,
        )
        if observe_transition:
            aligned_previous_hidden = (
                transition_candidate.context_state.align_recurrent_hidden(
                    previous_evolving_hidden,
                    source_state=recurrent_state.context_state,
                )
            )
            self._observe_recurrent_step(
                aligned_previous_hidden,
                output_hidden,
            )
        return RecurrentTransitionResult(
            hidden=output_hidden,
            loss=transition_candidate.loss,
            halting_state=halting_state,
            all_items_halted=self.utilities.all_items_halted(halting_state),
            context_state=replace(
                transition_candidate.context_state, hidden=output_hidden
            ),
        )

    def _finalize_recurrent_halting(
        self,
        hidden: Tensor,
        loss: Tensor | None,
        halting_state: HaltingStateBase | None,
    ) -> tuple[Tensor, Tensor | None]:
        return self.utilities.finalize_halting(hidden, loss, halting_state)

    def _accumulate_recurrent_losses(
        self,
        initial_loss: Tensor | None,
        auxiliary_losses: Iterable[Tensor],
    ) -> Tensor | None:
        accumulated_loss = initial_loss
        for auxiliary_loss in auxiliary_losses:
            accumulated_loss = self._accumulate_auxiliary_loss(
                accumulated_loss,
                auxiliary_loss,
            )
        return accumulated_loss

    @staticmethod
    def _blend_recurrent_branch_losses(
        common_loss: Tensor | None,
        source_loss: Tensor | None,
        target_loss: Tensor | None,
        transition_weight: float,
    ) -> Tensor | None:
        if transition_weight == 0.0:
            return source_loss
        if transition_weight == 1.0:
            return target_loss

        blended_loss = common_loss
        for branch_loss, branch_weight in (
            (source_loss, 1.0 - transition_weight),
            (target_loss, transition_weight),
        ):
            if branch_loss is None:
                continue
            branch_loss_delta = (
                branch_loss if common_loss is None else branch_loss - common_loss
            )
            weighted_branch_loss_delta = branch_weight * branch_loss_delta
            blended_loss = (
                weighted_branch_loss_delta
                if blended_loss is None
                else blended_loss + weighted_branch_loss_delta
            )
        return blended_loss

    @abstractmethod
    def forward(self, state: LayerState) -> LayerState:
        """Apply one configured recurrent composition to a LayerState."""
