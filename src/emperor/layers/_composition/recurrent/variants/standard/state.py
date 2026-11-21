from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

from torch import Tensor

from emperor.layers._composition.recurrent.state import RecurrentState
from emperor.layers._state import LayerState, are_all_items_halted

if TYPE_CHECKING:
    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.residual.base import ResidualState


@dataclass(frozen=True, kw_only=True)
class StandardRecurrentState(RecurrentState):
    hidden: Tensor
    loss: Tensor | None
    residual_state: ResidualState | None = None

    @property
    def output_hidden(self) -> Tensor:
        return self.hidden

    @property
    def branch_loss(self) -> Tensor | None:
        return self.loss

    def layer_state_for_transition(self, transition_input: Tensor) -> LayerState:
        return replace(
            self.context_state,
            hidden=transition_input,
            loss=self.loss,
            halting_state=None,
        )

    def next_state_from_output(
        self,
        output_state: LayerState,
        halting_update: tuple[HaltingStateBase | None, Tensor],
    ) -> Self:
        halting_state, output_hidden = halting_update
        next_context = replace(output_state, hidden=output_hidden)
        output_loss = output_state.loss
        fixed_input = self.align_fixed_input(next_context)
        all_items_halted = are_all_items_halted(halting_state)
        return replace(
            self,
            hidden=output_hidden,
            loss=output_loss,
            context_state=next_context,
            fixed_input=fixed_input,
            transition_index=self.transition_index + 1,
            halting_state=halting_state,
            all_items_halted=all_items_halted,
        )

    def with_forked_residual_state(self, residual_state: ResidualState | None) -> Self:
        return replace(self, residual_state=residual_state)

    def detach(self) -> Self:
        return replace(
            self,
            hidden=self.hidden.detach(),
            context_state=self.context_state.detach_recurrent_context(),
        )
