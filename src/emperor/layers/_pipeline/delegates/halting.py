from __future__ import annotations

from typing import TYPE_CHECKING

from emperor.layers._pipeline.delegates.validation import LayerHaltingDelegateValidator
from emperor.layers._state import LayerState, are_all_items_halted
from emperor.nn import Module

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting import HaltingInterface, HaltingStateBase
    from emperor.layers._config import LayerConfig


class LayerHaltingDelegate(Module):
    """Own the configured halting model, completion check and loss finalization."""

    VALIDATOR = LayerHaltingDelegateValidator

    def __init__(
        self,
        cfg: LayerConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.VALIDATOR.validate(self)
        self.config = self.cfg.halting_config
        self.output_dim: int = self.cfg.output_dim

        self.model: HaltingInterface[HaltingStateBase] | None = self.__build_model()
        self.is_terminal = False

    def __build_model(self) -> HaltingInterface[HaltingStateBase] | None:
        model = self._build_from_config(
            self.config,
            input_dim=self.output_dim,
        )
        return self.VALIDATOR.validate_built_halting_model_interface(model)

    def bind_shared(self, model: HaltingInterface[HaltingStateBase]) -> None:
        self.model = model

    def mark_as_terminal_layer(self) -> None:
        self.is_terminal = True

    def should_skip(self, state: LayerState) -> bool:
        if self.model is None or state.halting_state is None:
            return False
        return self.all_items_halted(state.halting_state)

    def all_items_halted(self, halting_state: HaltingStateBase | None) -> bool:
        return are_all_items_halted(halting_state)

    def apply_halting(self, state: LayerState) -> LayerState:
        model = self.model
        if model is None:
            return state

        halting_state, halting_output = model.update_halting_state(
            state.halting_state, state.hidden
        )
        state.halting_state = halting_state
        if self.is_terminal or self.all_items_halted(halting_state):
            state.hidden, state.loss = self.finalize_accumulation(
                halting_state, state.hidden, state.loss
            )
        else:
            state.hidden = halting_output
        return state

    def finalize_accumulation(
        self,
        halting_state: HaltingStateBase | None,
        hidden: Tensor,
        loss: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        model = self.model
        if model is None or halting_state is None:
            return hidden, loss
        hidden, halting_loss = model.finalize_weighted_accumulation(
            halting_state, hidden
        )
        auxiliary_loss = self.__maybe_reduce_halting_loss(halting_loss)
        return hidden, self.__maybe_accumulate_auxiliary_loss(loss, auxiliary_loss)

    @staticmethod
    def __maybe_reduce_halting_loss(halting_loss: Tensor) -> Tensor:
        if halting_loss.dim() == 0:
            return halting_loss
        return halting_loss.mean()

    @staticmethod
    def __maybe_accumulate_auxiliary_loss(
        accumulated_loss: Tensor | None,
        auxiliary_loss: Tensor,
    ) -> Tensor:
        if accumulated_loss is None:
            return auxiliary_loss
        return accumulated_loss + auxiliary_loss
