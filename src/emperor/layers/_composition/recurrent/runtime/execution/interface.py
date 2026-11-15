from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

from torch import Tensor

from emperor.layers._state import LayerState

if TYPE_CHECKING:
    from collections.abc import Callable

    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.residual.base import ResidualState


@dataclass(frozen=True)
class RecurrentExecutionResult:
    hidden: Tensor
    loss: Tensor | None
    context_state: LayerState | None = None


@dataclass(frozen=True)
class RecurrentTransitionResult:
    hidden: Tensor
    loss: Tensor | None
    halting_state: HaltingStateBase | None
    all_items_halted: bool
    context_state: LayerState | None = None


@dataclass(frozen=True)
class RecurrentTransitionCandidate:
    hidden: Tensor
    loss: Tensor | None
    residual_input: Tensor
    context_state: LayerState


class RecurrentExecutionState(Protocol):
    @property
    def context_state(self) -> LayerState: ...

    @property
    def transition_index(self) -> int: ...

    @property
    def halting_state(self) -> HaltingStateBase | None: ...

    @property
    def all_items_halted(self) -> bool: ...

    @property
    def output_hidden(self) -> Tensor: ...

    @property
    def branch_loss(self) -> Tensor | None: ...


@dataclass(frozen=True)
class PreparedRecurrentTransition:
    run_transition: Callable[[LayerState], LayerState]
    transition_input: Tensor
    previous_evolving_hidden: Tensor
    halting_update_enabled: bool
    loss: Tensor | None = None
    residual_state: ResidualState | None = None


_StateT = TypeVar("_StateT", bound=RecurrentExecutionState)
_ProvisionalSourceBranchOutput = TypeVar("_ProvisionalSourceBranchOutput")


class RecurrentExecutionAdapter(Protocol[_StateT]):
    """Recurrent variant ``nn.Module`` Interface consumed by Recurrent Execution."""

    def _run_recurrent_transition(
        self,
        recurrent_state: _StateT,
        prepared_transition: PreparedRecurrentTransition,
        transition_index: int = 0,
        observe_transition: bool = True,
    ) -> RecurrentTransitionResult: ...

    def _run_shared_handoff_boundary_transition(
        self,
        recurrent_state: _StateT,
        prepared_transition: PreparedRecurrentTransition,
        run_provisional_source_branch: Callable[
            [RecurrentTransitionResult],
            _ProvisionalSourceBranchOutput,
        ],
        target_residual_state: ResidualState | None = None,
        transition_index: int = 0,
    ) -> tuple[_ProvisionalSourceBranchOutput, RecurrentTransitionResult]: ...

    def _finalize_recurrent_halting(
        self,
        hidden: Tensor,
        loss: Tensor | None,
        halting_state: HaltingStateBase | None,
    ) -> tuple[Tensor, Tensor | None]: ...

    def _initialize_recurrent_execution_state(
        self,
        layer_state: LayerState,
        branch_base_loss: Tensor | None,
    ) -> _StateT: ...

    def _detach_recurrent_execution_state(
        self,
        recurrent_state: _StateT,
    ) -> _StateT: ...

    def _prepare_recurrent_transition(
        self,
        recurrent_state: _StateT,
        tracks_gradients: bool,
    ) -> PreparedRecurrentTransition: ...

    def _apply_recurrent_transition_result(
        self,
        recurrent_state: _StateT,
        transition_result: RecurrentTransitionResult,
    ) -> _StateT: ...

    def _fork_recurrent_handoff_state(
        self,
        recurrent_state: _StateT,
    ) -> _StateT: ...
