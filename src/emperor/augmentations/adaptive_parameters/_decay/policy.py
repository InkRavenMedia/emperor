from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from emperor.augmentations.adaptive_parameters._decay.base import DecayAbstract
from emperor.augmentations.adaptive_parameters._decay.options import (
    WeightDecayScheduleOptions,
)
from emperor.augmentations.adaptive_parameters._decay.validation import DecayValidator
from emperor.augmentations.adaptive_parameters._decay.variants.exponential import (
    ExponentialDecay,
)
from emperor.augmentations.adaptive_parameters._decay.variants.linear import (
    LinearDecay,
)
from emperor.augmentations.adaptive_parameters._decay.variants.multiplicative import (
    MultiplicativeDecay,
)

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._biases.config import (
        DynamicBiasConfig,
    )
    from emperor.augmentations.adaptive_parameters._weights.config import (
        DynamicWeightConfig,
    )


class DecayPolicy(nn.Module):
    """Apply the selected decay formula with shared warm-up and forward counters."""

    VALIDATOR = DecayValidator

    decay_variant: DecayAbstract | None

    def __init__(self, cfg: DynamicWeightConfig | DynamicBiasConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.decay_schedule_option = self.cfg.decay_schedule
        self.decay_rate = self.cfg.decay_rate
        self.decay_warmup_batches = self.cfg.decay_warmup_batches or 0
        initial_decay_step = torch.zeros(1)
        self.register_buffer("decay_step", initial_decay_step)
        initial_warmup_step = torch.zeros(1)
        self.register_buffer("warmup_step", initial_warmup_step)

    @property
    def decay_schedule_option(self) -> WeightDecayScheduleOptions | None:
        return self._decay_schedule_option

    @decay_schedule_option.setter
    def decay_schedule_option(self, option: WeightDecayScheduleOptions | None) -> None:
        self._decay_schedule_option = option
        match option:
            case WeightDecayScheduleOptions.EXPONENTIAL:
                self.decay_variant = ExponentialDecay()
            case WeightDecayScheduleOptions.LINEAR:
                self.decay_variant = LinearDecay()
            case WeightDecayScheduleOptions.MULTIPLICATIVE:
                self.decay_variant = MultiplicativeDecay()
            case _:
                self.decay_variant = None

    def forward(self, parameters: Tensor) -> Tensor:
        if self.__is_decay_schedule_disabled():
            return parameters

        is_warming_up = self.__advance_warmup_if_active()
        if is_warming_up:
            return parameters

        decay_factor = self.__compute_decay_factor_by_schedule()
        self.__advance_decay_step()
        return parameters * decay_factor

    def __is_decay_schedule_disabled(self) -> bool:
        return (
            self.decay_schedule_option is None
            or self.decay_schedule_option == WeightDecayScheduleOptions.DISABLED
        )

    def __advance_warmup_if_active(self) -> bool:
        if self.warmup_step < self.decay_warmup_batches:
            if self.training:
                self.warmup_step += 1
            return True
        return False

    def __compute_decay_factor_by_schedule(self) -> Tensor:
        active_decay_variant = self.VALIDATOR.validate_decay_variant(
            self.decay_variant, self.decay_schedule_option
        )
        active_decay_rate = cast(float, self.decay_rate)
        current_decay_step = self.decay_step
        decay_factor = active_decay_variant.compute_factor(
            active_decay_rate, current_decay_step
        )
        return decay_factor

    def __advance_decay_step(self) -> None:
        if self.training:
            self.decay_step += 1
