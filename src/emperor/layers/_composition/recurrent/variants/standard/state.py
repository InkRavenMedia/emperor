from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

from torch import Tensor

from emperor.layers._state import LayerState

if TYPE_CHECKING:
    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.residual.base import ResidualState


@dataclass(frozen=True)
class StandardRecurrentState:
    hidden: Tensor
    fixed_input: Tensor
    loss: Tensor | None
    context_state: LayerState
    transition_index: int
    residual_state: ResidualState | None = None
    halting_state: HaltingStateBase | None = None
    all_items_halted: bool = False

    @property
    def output_hidden(self) -> Tensor:
        return self.hidden

    @property
    def branch_loss(self) -> Tensor | None:
        return self.loss

    def detach(self) -> Self:
        return replace(
            self,
            hidden=self.hidden.detach(),
            context_state=self.context_state.detach_recurrent_context(),
        )
