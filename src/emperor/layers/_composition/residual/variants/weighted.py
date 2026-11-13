"""Weighted residual connection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from emperor.layers._composition.residual.base import (
    ResidualConnectionAbstract,
    ResidualState,
)
from emperor.layers._composition.residual.config import WeightedResidualConfig

if TYPE_CHECKING:
    from emperor.layers import LayerStack, LayerStackConfig
    from emperor.linears import LinearAbstract, LinearLayerConfig


class WeightedResidual(ResidualConnectionAbstract):
    def __init__(
        self,
        cfg: WeightedResidualConfig,
        overrides: WeightedResidualConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.model_config: LayerStackConfig | LinearLayerConfig | None = (
            self.cfg.model_config
        )
        self.raw_weight: nn.Parameter | None = None
        self.model: LayerStack | LinearAbstract | None = None
        self.__initialize_scalar_weight_or_coefficient_model()

    def __initialize_scalar_weight_or_coefficient_model(self) -> None:
        if self.model_config is None:
            initial_raw_coefficient = self._initial_raw_mix_coefficient()
            self.raw_weight = nn.Parameter(initial_raw_coefficient)
            return

        self.model = self._build_from_config(
            self.model_config,
            input_dim=self.residual_dim * 2,
            output_dim=self.residual_dim,
        )

    @staticmethod
    def _initial_raw_mix_coefficient() -> Tensor:
        return torch.tensor(0.0)

    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        residual_state: ResidualState | None = None,
    ) -> Tensor:
        raw_mix_coefficient = self._resolve_raw_mix_coefficient(current, previous)
        residual_weight = torch.tanh(raw_mix_coefficient)
        return previous + residual_weight * current

    def _resolve_raw_mix_coefficient(
        self,
        current: Tensor,
        previous: Tensor,
    ) -> Tensor:
        if self.model is None:
            self.VALIDATOR.validate_raw_mix_coefficient(self.raw_weight)
            raw_mix_coefficient = self.raw_weight
        else:
            raw_mix_coefficient = self.__compute_raw_mix_coefficient_with_model(
                current, previous
            )

        return raw_mix_coefficient

    def __compute_raw_mix_coefficient_with_model(
        self,
        current: Tensor,
        previous: Tensor,
    ) -> Tensor:
        concatenated_representations = torch.cat((current, previous), dim=-1)
        return self.model.forward_hidden(concatenated_representations)
