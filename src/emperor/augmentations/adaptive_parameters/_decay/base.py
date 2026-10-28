from torch import Tensor


class DecayAbstract:
    """Calculate a decay factor without owning counters or parameter state."""

    def compute_factor(self, decay_rate: float, current_decay_step: Tensor) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_factor()."
        )
