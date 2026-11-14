"""Forward-local state for Hierarchical Reasoning Model recurrence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
