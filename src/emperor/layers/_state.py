from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, cast, overload

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

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

    __loss_recorder: ClassVar[
        ContextVar[tuple[Module, Callable[[Tensor | None], None]] | None]
    ] = ContextVar("tensor_module_loss_recorder", default=None)

    @staticmethod
    @overload
    def sum_losses(first: Tensor, /, *others: Tensor | None) -> Tensor: ...

    @staticmethod
    @overload
    def sum_losses(*losses: Tensor | None) -> Tensor | None: ...

    @staticmethod
    def sum_losses(*losses: Tensor | None) -> Tensor | None:
        present = [loss for loss in losses if loss is not None]
        if present:
            return sum(present[1:], present[0])
        return None

    @staticmethod
    def module_loss(module: Module) -> Tensor | None:
        get_loss = getattr(module, "get_auxiliary_loss", None)
        if not callable(get_loss):
            return None
        loss = cast("Tensor | None", get_loss())
        if loss is None or loss.ndim == 0:
            return loss
        return loss.mean()

    @staticmethod
    def run_tensor_module(
        module: Module,
        inputs: Tensor,
        *,
        forward: Callable[[Tensor], Tensor] | None = None,
    ) -> LayerState:
        """Capture a nested loss without changing the tensor contract for overrides or hooks."""
        if forward is None:
            forward = module
        losses: list[Tensor | None] = []
        token = LayerState.__loss_recorder.set((module, losses.append))
        try:
            hidden = forward(inputs)
        finally:
            LayerState.__loss_recorder.reset(token)
        losses.append(LayerState.module_loss(module))
        return LayerState(hidden=hidden, loss=LayerState.sum_losses(*losses))

    @staticmethod
    def loss_recorder(module: Module) -> Callable[[Tensor | None], None] | None:
        active = LayerState.__loss_recorder.get()
        if active is not None and active[0] is module:
            return active[1]
        return None

    def tensor_output(self, module: Module) -> Tensor:
        record_loss = self.loss_recorder(module)
        if record_loss is not None:
            record_loss(self.loss)
        return self.hidden

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


def are_all_items_halted(halting_state: HaltingStateBase | None) -> bool:
    if halting_state is None or halting_state.halt_mask is None:
        return False
    return bool(halting_state.halt_mask.all().item())
