from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from emperor.halting import HaltingStateBase
    from emperor.layers._state import LayerState


@dataclass(frozen=True, kw_only=True)
class RecurrentState(ABC):
    fixed_input: Tensor
    context_state: LayerState
    transition_index: int
    halting_state: HaltingStateBase | None = None
    all_items_halted: bool = False

    @property
    @abstractmethod
    def output_hidden(self) -> Tensor:
        pass

    @property
    @abstractmethod
    def branch_loss(self) -> Tensor | None:
        pass

    def layer_state_for_transition(self, transition_input: Tensor) -> LayerState:
        return replace(
            self.context_state, hidden=transition_input, loss=None, halting_state=None
        )

    def align_fixed_input(self, output_state: LayerState) -> Tensor:
        return output_state.align_recurrent_hidden(
            self.fixed_input, source_state=self.context_state
        )

    @abstractmethod
    def detach(self) -> Self:
        pass


def new_recurrent_initial_buffer(output_dim: int, standard_deviation: float) -> Tensor:
    initial_buffer = torch.empty(output_dim)
    if standard_deviation == 0:
        return initial_buffer.zero_()
    return nn.init.trunc_normal_(
        initial_buffer,
        mean=0.0,
        std=standard_deviation,
        a=-2.0 * standard_deviation,
        b=2.0 * standard_deviation,
    )


def append_auxiliary_loss(
    losses: tuple[Tensor, ...], output_loss: Tensor | None
) -> tuple[Tensor, ...]:
    if output_loss is None:
        return losses
    reduced_loss = output_loss if output_loss.dim() == 0 else output_loss.mean()
    return (*losses, reduced_loss)
