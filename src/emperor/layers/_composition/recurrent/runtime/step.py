from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from emperor.layers._composition.recurrent.config import RecurrentCompressionConfig
from emperor.layers._composition.recurrent.runtime.compression import (
    RecurrentTokenCompressor,
)
from emperor.layers._composition.recurrent.state import RecurrentState
from emperor.layers._composition.recurrent.validation import (
    HierarchicalReasoningModelRecurrentValidator,
    RecurrentLayerValidator,
    TinyRecursiveModelRecurrentValidator,
)
from emperor.layers._pipeline import LayerProcessingPipeline
from emperor.nn import Module

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from torch import Tensor

    from emperor.config import ConfigBase
    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.recurrent.config import RecurrentCompositionConfig
    from emperor.layers._state import LayerState


RecurrentStateT = TypeVar("RecurrentStateT", bound=RecurrentState)


class RecurrentStep(Module, Generic[RecurrentStateT], ABC):
    VALIDATOR: ClassVar[
        type[RecurrentLayerValidator]
        | type[TinyRecursiveModelRecurrentValidator]
        | type[HierarchicalReasoningModelRecurrentValidator]
    ]

    def __init__(self, cfg: RecurrentCompositionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pipeline = LayerProcessingPipeline(self.cfg)
        compression = cfg.resolve_compression_config()
        self.compressor = (
            RecurrentTokenCompressor(
                compression,
                RecurrentCompressionConfig(input_dim=cfg.output_dim),
            )
            if compression is not None
            else None
        )
        self.diagnostic_observer: Callable[[Tensor, Tensor], None] | None = None
        self.__diagnostic_suppression_depth = 0

    def _build_transition_model(self, block_config: ConfigBase) -> Module:
        if self.compressor is not None:
            block_config = self.compressor.configure_block(block_config)
        return self._build_from_config(
            block_config, input_dim=self.cfg.output_dim, output_dim=self.cfg.output_dim
        )

    def _call_transition_block(
        self, block: Module, state: LayerState, *, complete_prefix: bool = False
    ) -> LayerState:
        if self.compressor is None:
            return block(state)
        return self.compressor(block, state, complete_prefix=complete_prefix)

    @abstractmethod
    def forward(
        self,
        recurrent_state: RecurrentStateT,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> RecurrentStateT:
        pass

    @abstractmethod
    def initialize_recurrent_state(
        self, state: LayerState, include_caller_loss: bool = True
    ) -> RecurrentStateT:
        pass

    @abstractmethod
    def compute_output(
        self,
        recurrent_state: RecurrentStateT,
        target_state: RecurrentStateT | None = None,
    ) -> LayerState:
        pass

    @abstractmethod
    def update_state(
        self,
        recurrent_state: RecurrentStateT,
        output_state: LayerState,
        tracks_gradients: bool,
        observe_transition: bool = True,
    ) -> RecurrentStateT:
        pass

    def fork(self, recurrent_state: RecurrentStateT) -> RecurrentStateT:
        forked_residual_state = self.pipeline.fork_recurrent_residual_state(
            recurrent_state.residual_state
        )
        return recurrent_state.with_forked_residual_state(forked_residual_state)

    def finalize(self, recurrent_state: RecurrentStateT) -> LayerState:
        hidden, loss = self.pipeline.halting.finalize_accumulation(
            recurrent_state.halting_state,
            recurrent_state.output_hidden,
            recurrent_state.branch_loss,
        )
        return replace(recurrent_state.context_state, hidden=hidden, loss=loss)

    def _apply_halting(
        self,
        recurrent_state: RecurrentStateT,
        output_state: LayerState,
        previous_hidden: Tensor,
        update_enabled: bool,
        observe_transition: bool,
    ) -> tuple[HaltingStateBase | None, Tensor]:
        halting_state, output_hidden = self.__maybe_update_halting(
            recurrent_state.halting_state, output_state.hidden, update_enabled
        )
        if observe_transition:
            aligned_previous_hidden = output_state.align_recurrent_hidden(
                previous_hidden, source_state=recurrent_state.context_state
            )
            if self.diagnostic_observer is not None:
                self.diagnostic_observer(aligned_previous_hidden, output_hidden)
        return halting_state, output_hidden

    def __maybe_update_halting(
        self,
        halting_state: HaltingStateBase | None,
        hidden: Tensor,
        update_enabled: bool,
    ) -> tuple[HaltingStateBase | None, Tensor]:
        model = self.pipeline.halting.model
        if model is None or not update_enabled:
            return halting_state, hidden
        halting_state, output_hidden = model.update_halting_state(halting_state, hidden)
        self.VALIDATOR.validate_halting_output(output_hidden, hidden)
        return halting_state, output_hidden

    def set_diagnostic_observer(
        self, observer: Callable[[Tensor, Tensor], None] | None
    ) -> None:
        self.diagnostic_observer = observer

    @property
    def diagnostic_observation_enabled(self) -> bool:
        return self.__diagnostic_suppression_depth == 0

    @contextmanager
    def _observe_model(self, enabled: bool) -> Generator[None, None, None]:
        if enabled:
            yield
            return
        self.__diagnostic_suppression_depth += 1
        try:
            yield
        finally:
            self.__diagnostic_suppression_depth -= 1
