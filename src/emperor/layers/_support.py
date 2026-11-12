from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._state import LayerState


class LayerModuleBase(Module):
    def forward_sequence(self, hidden: Tensor) -> LayerState:
        """Transform sequence rows with fresh execution state and return hidden/loss."""
        from emperor.layers._layer import Layer

        return Layer.run_model_on_sequence_rows(self, hidden)

    def _reduce_auxiliary_loss(self, loss: Tensor) -> Tensor:
        return loss if loss.dim() == 0 else loss.mean()

    def _accumulate_auxiliary_loss(
        self,
        loss: Tensor | None,
        auxiliary_loss: Tensor,
    ) -> Tensor:
        return auxiliary_loss if loss is None else loss + auxiliary_loss
