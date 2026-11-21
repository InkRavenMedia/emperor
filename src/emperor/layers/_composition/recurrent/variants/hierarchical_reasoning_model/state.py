from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

from torch import Tensor

from emperor.layers._composition.recurrent.state import (
    RecurrentState,
    append_auxiliary_loss,
)
from emperor.layers._state import are_all_items_halted

if TYPE_CHECKING:
    from emperor.halting import HaltingStateBase
    from emperor.layers._state import LayerState


@dataclass(frozen=True, kw_only=True)
class HierarchicalReasoningModelState(RecurrentState):
    high: Tensor
    low: Tensor
    initial_loss: Tensor | None
    auxiliary_losses: tuple[Tensor, ...]

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

    def next_state_from_output(
        self,
        output_state: LayerState,
        halting_update: tuple[HaltingStateBase | None, Tensor],
        *,
        updates_high: bool,
        high_initial: Tensor,
        low_initial: Tensor,
    ) -> Self:
        halting_state, output_hidden = halting_update
        auxiliary_losses = append_auxiliary_loss(
            self.auxiliary_losses, output_state.loss
        )
        context_state = replace(output_state, hidden=output_hidden)
        high = context_state.align_recurrent_hidden(
            self.high, initial=high_initial, source_state=self.context_state
        )
        low = context_state.align_recurrent_hidden(
            self.low, initial=low_initial, source_state=self.context_state
        )
        if updates_high:
            high = output_hidden
        else:
            low = output_hidden
        fixed_input = self.align_fixed_input(context_state)
        all_items_halted = are_all_items_halted(halting_state)
        return replace(
            self,
            high=high,
            low=low,
            context_state=context_state,
            fixed_input=fixed_input,
            auxiliary_losses=auxiliary_losses,
            transition_index=self.transition_index + 1,
            halting_state=halting_state,
            all_items_halted=all_items_halted,
        )

    def detach(self) -> Self:
        return replace(
            self,
            high=self.high.detach(),
            low=self.low.detach(),
            context_state=self.context_state.detach_recurrent_context(),
        )
