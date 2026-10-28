import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._decay.base import DecayAbstract


class ExponentialDecay(DecayAbstract):
    def compute_factor(self, decay_rate: float, current_decay_step: Tensor) -> Tensor:
        maximum_finite_decay_rate = torch.finfo(current_decay_step.dtype).max
        dtype_aligned_decay_rate = current_decay_step.new_tensor(decay_rate)
        bounded_decay_rate = dtype_aligned_decay_rate.clamp(
            max=maximum_finite_decay_rate
        )
        negative_decay_rate = -bounded_decay_rate
        exponential_decay_exponent = negative_decay_rate * current_decay_step
        exponential_decay_factor = torch.exp(exponential_decay_exponent)
        return exponential_decay_factor
