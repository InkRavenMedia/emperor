from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.attention import AttentionLayerState
from emperor.transformer.sequence_expansion._state import _SequenceExpansionState

if TYPE_CHECKING:
    from emperor.halting import HaltingStateBase
    from emperor.transformer.sequence_expansion import SequenceExpansionRecord


@dataclass
class TransformerEncoderLayerState(_SequenceExpansionState, AttentionLayerState):
    """Encoder state with forward-local records of optional expansion events."""

    sequence_expansions: tuple["SequenceExpansionRecord", ...] = ()

    @property
    def expansion_padding_mask(self) -> Tensor | None:
        return self.key_padding_mask

    @expansion_padding_mask.setter
    def expansion_padding_mask(self, value: Tensor | None) -> None:
        self.key_padding_mask = value

    def commit_recurrent_context(self, context: "TransformerEncoderLayerState") -> None:
        super().commit_recurrent_context(context)
        self.attention_mask = context.attention_mask


@dataclass
class TransformerDecoderLayerState(_SequenceExpansionState):
    """Context preserved while decoder blocks pass through Emperor controllers."""

    hidden: Tensor
    loss: Tensor | None = None
    halting_state: "HaltingStateBase | None" = None
    target_key_padding_mask: Tensor | None = None
    target_attention_mask: Tensor | None = None
    encoder_output: Tensor | None = None
    encoder_padding_mask: Tensor | None = None
    cross_attention_mask: Tensor | None = None
    controller_state: object | None = None

    @property
    def expansion_padding_mask(self) -> Tensor | None:
        return self.target_key_padding_mask

    @expansion_padding_mask.setter
    def expansion_padding_mask(self, value: Tensor | None) -> None:
        self.target_key_padding_mask = value

    def commit_recurrent_context(self, context: "TransformerDecoderLayerState") -> None:
        super().commit_recurrent_context(context)
        self.target_attention_mask = context.target_attention_mask
        self.cross_attention_mask = context.cross_attention_mask

    def select_recurrent_tokens(
        self, selected: "TransformerDecoderLayerState"
    ) -> "TransformerDecoderLayerState":
        selected = super().select_recurrent_tokens(selected)
        if self.sequence_expansions:
            # Sampling pads shorter selections with duplicate indices. Regenerate
            # the validated causal mask; sampled padding remains excluded by keys.
            selected.resize_expansion_masks(selected.hidden.size(1))
        return selected

    def resize_expansion_masks(self, sequence_length: int) -> None:
        # Explicit target masks are validated as standard causal masks; cross
        # masks must have the same source restrictions for every target query.
        target_mask = self.target_attention_mask
        if target_mask is not None:
            blocked = torch.ones(
                sequence_length,
                sequence_length,
                device=target_mask.device,
                dtype=torch.bool,
            ).triu(1)
            expanded = target_mask.new_zeros(
                *target_mask.shape[:-2], sequence_length, sequence_length
            )
            expanded = expanded.masked_fill(
                blocked, True if target_mask.dtype == torch.bool else -torch.inf
            )
            previous_length = target_mask.size(-1)
            common_length = min(sequence_length, previous_length)
            prefix = target_mask[..., :common_length, :common_length]
            # Selected-token masks may contain duplicate padded indices. Keep
            # their gradient connections only where they match causal order.
            expanded_prefix = expanded[..., :common_length, :common_length]
            expanded[..., :common_length, :common_length] = torch.where(
                prefix == expanded_prefix, prefix, expanded_prefix
            )
            self.target_attention_mask = expanded
        cross_mask = self.cross_attention_mask
        if cross_mask is not None and sequence_length > cross_mask.size(-2):
            appended_rows = cross_mask[..., :1, :].expand(
                *cross_mask.shape[:-2],
                sequence_length - cross_mask.size(-2),
                cross_mask.size(-1),
            )
            self.cross_attention_mask = torch.cat((cross_mask, appended_rows), dim=-2)
        elif cross_mask is not None:
            self.cross_attention_mask = cross_mask[..., :sequence_length, :]
