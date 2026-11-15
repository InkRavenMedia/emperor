from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from emperor.layers._composition.recurrent.runtime.execution.interface import (
    RecurrentTransitionResult as RecurrentTransitionResult,
)
from emperor.layers._support import LayerModuleBase

if TYPE_CHECKING:
    from emperor.config import ConfigBase
    from emperor.layers._composition.recurrent.config import RecurrentCompositionConfig
    from emperor.layers._state import LayerState
    from emperor.nn import Module


class RecurrentCompositionAbstract(LayerModuleBase, ABC):
    supports_recurrent_diagnostics = False

    def __init__(
        self,
        cfg: RecurrentCompositionConfig,
        overrides: RecurrentCompositionConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: RecurrentCompositionConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim

    def _build_transition_model(self, transition_config: ConfigBase) -> Module:
        return self._build_from_config(
            transition_config,
            input_dim=self.output_dim,
            output_dim=self.output_dim,
        )

    def _new_recurrent_initial_buffer(self, standard_deviation: float) -> Tensor:
        initial_buffer = torch.empty(self.output_dim)
        if standard_deviation == 0:
            return initial_buffer.zero_()
        return nn.init.trunc_normal_(
            initial_buffer,
            mean=0.0,
            std=standard_deviation,
            a=-2.0 * standard_deviation,
            b=2.0 * standard_deviation,
        )

    @staticmethod
    def _expand_recurrent_initial(initial: Tensor, hidden: Tensor) -> Tensor:
        initial_broadcast_shape = (1,) * (hidden.ndim - 1) + (hidden.shape[-1],)
        return initial.view(initial_broadcast_shape).expand_as(hidden)

    @abstractmethod
    def forward(self, state: LayerState) -> LayerState:
        pass
