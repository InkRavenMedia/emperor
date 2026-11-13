"""Weighted-blend residual connection."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from emperor.layers._composition.residual.base import (
    ResidualConnectionAbstract,
    ResidualState,
)
from emperor.layers._composition.residual.config import WeightedBlendResidualConfig
from emperor.layers._options import ActivationOptions

if TYPE_CHECKING:
    from emperor.layers import LayerStack, LayerStackConfig
    from emperor.linears import LinearAbstract, LinearLayerConfig


class WeightedBlendResidual(ResidualConnectionAbstract):
    def __init__(
        self,
        cfg: WeightedBlendResidualConfig,
        overrides: WeightedBlendResidualConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.coefficient_activation: ActivationOptions = self.cfg.coefficient_activation
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
        initial_alpha = 0.9
        initial_logit = math.log(initial_alpha / (1.0 - initial_alpha))
        return torch.tensor(initial_logit)

    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        residual_state: ResidualState | None = None,
    ) -> Tensor:
        current_coefficient = self._resolve_mix_coefficient(current, previous)
        previous_coefficient = 1.0 - current_coefficient
        current_contribution = current_coefficient * current
        previous_contribution = previous_coefficient * previous
        return current_contribution + previous_contribution

    def _resolve_mix_coefficient(
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

        return self.__activate_mix_coefficient(raw_mix_coefficient)

    def __activate_mix_coefficient(self, raw_mix_coefficient: Tensor) -> Tensor:
        if self.coefficient_activation == ActivationOptions.DISABLED:
            return raw_mix_coefficient
        return self.coefficient_activation(raw_mix_coefficient)

    def __compute_raw_mix_coefficient_with_model(
        self,
        current: Tensor,
        previous: Tensor,
    ) -> Tensor:
        concatenated_representations = torch.cat((current, previous), dim=-1)
        return self.model.forward_hidden(concatenated_representations)
