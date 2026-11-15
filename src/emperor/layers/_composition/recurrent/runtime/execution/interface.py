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


RecurrentStateT = TypeVar("RecurrentStateT", bound=RecurrentExecutionState)
