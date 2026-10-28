import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._decay.base import DecayAbstract


class MultiplicativeDecay(DecayAbstract):
    def compute_factor(self, decay_rate: float, current_decay_step: Tensor) -> Tensor:
        per_step_retention_factor = 1.0 - decay_rate
        multiplicative_decay_base = current_decay_step.new_tensor(
            per_step_retention_factor
        )
        multiplicative_decay_factor = torch.pow(
            multiplicative_decay_base,
            current_decay_step,
        )
        return multiplicative_decay_factor
