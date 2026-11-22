"""Lift one image-level stick-breaking decision onto its entire token sequence."""

from dataclasses import replace

import torch.nn.functional as F
from torch import Tensor

from emperor.halting import (
    HaltingHiddenStateModeOptions,
    StickBreaking,
    StickBreakingState,
)
from emperor.neuron._cluster.routing.halting import _NeuronHaltingLifecycle
from emperor.neuron._cluster.routing.sequence_tokens import SequenceTokens


class SequenceHalting:
    def __init__(self, model: StickBreaking | None):
        self.model = model
        self.state: StickBreakingState | None = None
        self.accumulated: Tensor | None = None

    def update(self, tokens: SequenceTokens, active: Tensor) -> SequenceTokens:
        if self.model is None or not active.any():
            return tokens
        summary = tokens.summary()
        prior_mass = (
            self.state.accumulated_halt_probabilities if self.state is not None else 0
        )
        self.state, _ = _NeuronHaltingLifecycle.update(
            self.model, self.state, summary, summary, active
        )
        mass = (self.state.accumulated_halt_probabilities - prior_mass).masked_fill(
            ~active, 0
        )
        weighted = tokens.hidden * mass[:, None, None]
        if self.accumulated is not None:
            weighted = weighted + F.pad(
                self.accumulated, (0, 0, 0, weighted.size(1) - self.accumulated.size(1))
            )
        self.accumulated = weighted
        if self.model.hidden_state_mode == HaltingHiddenStateModeOptions.ACCUMULATED:
            hidden = weighted.where(active[:, None, None], tokens.hidden)
            return replace(tokens, hidden=hidden)
        return tokens

    def halt_mask(self, tokens: SequenceTokens) -> Tensor:
        return _NeuronHaltingLifecycle.halt_mask_tensor(
            self.state, tokens.hidden.size(0), tokens.hidden.device
        )

    def finish(
        self, tokens: SequenceTokens, raw: SequenceTokens, escaped: Tensor
    ) -> tuple[SequenceTokens, Tensor | None]:
        if self.state is None:
            return tokens, None
        _, ponder = _NeuronHaltingLifecycle.finalize(
            self.model, self.state, raw.summary(), None
        )
        remaining = (1 - self.state.accumulated_halt_probabilities).masked_fill(
            self.state.halt_mask, 0
        )
        hidden = self.accumulated + remaining[:, None, None] * raw.hidden
        hidden = tokens.hidden.where(escaped[:, None, None], hidden)
        return replace(tokens, hidden=hidden), ponder
