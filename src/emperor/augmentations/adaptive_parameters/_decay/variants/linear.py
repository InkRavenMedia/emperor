import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._decay.base import DecayAbstract


class LinearDecay(DecayAbstract):
    def compute_factor(self, decay_rate: float, current_decay_step: Tensor) -> Tensor:
        unbounded_linear_decay_factor = 1.0 - decay_rate * current_decay_step
        nonnegative_linear_decay_factor = torch.clamp(
            unbounded_linear_decay_factor, min=0.0
        )
        return nonnegative_linear_decay_factor
