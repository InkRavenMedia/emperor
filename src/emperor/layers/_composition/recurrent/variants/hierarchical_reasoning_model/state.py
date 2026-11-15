from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

from torch import Tensor

if TYPE_CHECKING:
    from emperor.halting import HaltingStateBase
    from emperor.layers._state import LayerState


@dataclass(frozen=True)
class HierarchicalReasoningModelState:
    fixed_input: Tensor
    high: Tensor
    low: Tensor
    initial_loss: Tensor | None
    auxiliary_losses: tuple[Tensor, ...]
    context_state: LayerState
    transition_index: int
    halting_state: HaltingStateBase | None = None
    all_items_halted: bool = False

    @property
    def output_hidden(self) -> Tensor:
        return self.high

    @property
    def branch_loss(self) -> Tensor | None:
        accumulated_loss = self.initial_loss
        for auxiliary_loss in self.auxiliary_losses:
            accumulated_loss = (
                auxiliary_loss
                if accumulated_loss is None
                else accumulated_loss + auxiliary_loss
            )
        return accumulated_loss

    def detach(self) -> Self:
        return replace(
            self,
            high=self.high.detach(),
            low=self.low.detach(),
            context_state=self.context_state.detach_recurrent_context(),
        )
