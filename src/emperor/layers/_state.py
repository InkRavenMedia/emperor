from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.residual.base import ResidualState


@dataclass
class LayerState:
    hidden: Tensor
    loss: Tensor | None = None
    halting_state: HaltingStateBase | None = None
    residual_state: ResidualState | None = field(
        default=None,
        kw_only=True,
        repr=False,
        compare=False,
    )

    def validate_recurrent_shape(self, previous_hidden: Tensor) -> None:
        """Owners with sequence-layout metadata may explicitly allow resizing."""
        if self.hidden.shape != previous_hidden.shape:
            raise ValueError(
                "recurrent block must preserve hidden shape, got candidate "
                f"shape {tuple(self.hidden.shape)} and previous shape {tuple(previous_hidden.shape)}"
            )

    def align_recurrent_hidden(
        self,
        hidden: Tensor,
        *,
        initial: Tensor | None = None,
        source_state: LayerState | None = None,
    ) -> Tensor:
        self.validate_recurrent_shape(hidden)
        return hidden

    def commit_recurrent_context(self, context: LayerState) -> None:
        """Commit owner-specific context after a successful recurrent execution."""

    def detach_recurrent_context(self) -> LayerState:
        return self

    def select_recurrent_tokens(self, selected: LayerState) -> LayerState:
        return selected

    def merge_recurrent_tokens(
        self, selected: LayerState, hidden: Tensor
    ) -> LayerState:
        return replace(self, hidden=hidden, loss=selected.loss)

    @contextmanager
    def scoped_residual_state(
        self, replacement: ResidualState | None
    ) -> Iterator[None]:
        enclosing_residual_state = self.residual_state
        self.residual_state = replacement
        try:
            yield
        finally:
            self.residual_state = enclosing_residual_state


@dataclass(kw_only=True)
class _MaskedInputLayerState(LayerState):
    """Zero inactive inputs before each model, including after normalization."""

    input_mask: Tensor
