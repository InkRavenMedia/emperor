"""Token layouts used at whole-image routing boundaries."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from emperor.transformer import TransformerEncoderLayerState

if TYPE_CHECKING:
    from emperor.neuron._neuron.core import Neuron


@dataclass(frozen=True)
class SequenceTokens:
    hidden: Tensor
    padding: Tensor
    attention_mask: Tensor | None = None

    @classmethod
    def from_state(cls, state: TransformerEncoderLayerState) -> "SequenceTokens":
        hidden = state.hidden
        padding = state.key_padding_mask
        if padding is None:
            padding = torch.zeros(
                hidden.shape[:2], dtype=torch.bool, device=hidden.device
            )
        else:
            padding = padding.bool()
            if state.attention_mask is None:
                # Use the expansion layout: valid patches first, including [CLS].
                # An explicit attention mask instead addresses fixed physical positions.
                order = padding.long().argsort(dim=1, stable=True)
                hidden = hidden.gather(1, order.unsqueeze(-1).expand_as(hidden))
                padding = padding.gather(1, order)
            hidden = hidden.masked_fill(padding.unsqueeze(-1), 0)
        return cls(hidden, padding, state.attention_mask)

    def state(self) -> TransformerEncoderLayerState:
        mask = self.padding if self.padding.any() else None
        return TransformerEncoderLayerState(
            hidden=self.hidden,
            key_padding_mask=mask,
            attention_mask=self.attention_mask,
        )

    def rows(self, indices: Tensor) -> "SequenceTokens":
        mask = self.attention_mask
        if mask is not None and mask.ndim == 3 and mask.size(0) > 1:
            # Attention uses a flattened batch × heads leading dimension.
            mask = (
                mask.unflatten(0, (self.hidden.size(0), -1))
                .index_select(0, indices)
                .flatten(0, 1)
            )
        return SequenceTokens(
            self.hidden.index_select(0, indices),
            self.padding.index_select(0, indices),
            mask,
        )

    def pad_to(self, length: int) -> "SequenceTokens":
        extra = length - self.hidden.shape[1]
        if extra == 0:
            return self
        return SequenceTokens(
            F.pad(self.hidden, (0, 0, 0, extra)),
            F.pad(self.padding, (0, extra), value=True),
            self.attention_mask,
        )

    def summary(self) -> Tensor:
        count = (~self.padding).sum(dim=1, keepdim=True).clamp_min(1)
        return self.hidden.sum(dim=1) / count.to(self.hidden.dtype)


def read_sequence_history(
    neuron: "Neuron",
    result: TransformerEncoderLayerState,
    history: list[SequenceTokens],
    rows: Tensor | None = None,
) -> TransformerEncoderLayerState:
    residual = neuron.sequence_residual
    if residual is None:
        return result
    current = SequenceTokens.from_state(result)
    sources = [
        (source if rows is None else source.rows(rows)).pad_to(current.hidden.shape[1])
        for source in history
    ]
    residual_state = residual.new_state(sources[0].hidden)
    for source in sources[1:]:
        residual_state.append(source.hidden)
    padding = [source.padding for source in sources]
    padding.append(current.padding)
    residual_state.next_source_padding = torch.stack(padding)
    mixed = residual(
        current.hidden,
        sources[-1].hidden,
        residual_state=residual_state,
    )
    return replace(
        result,
        hidden=mixed,
        key_padding_mask=current.padding if current.padding.any() else None,
    )
