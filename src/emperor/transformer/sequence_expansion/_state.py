from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from emperor.layers import LayerState
from emperor.transformer.sequence_expansion._layout import (
    align_valid_tokens,
    append_valid_tokens,
)


@dataclass(frozen=True)
class SequenceExpansionRecord:
    """One event, with per-sample locations after reusing padding positions."""

    input_padding_mask: Tensor
    selected_counts: Tensor
    count_logits: Tensor
    selected_probabilities: Tensor
    position_ids: Tensor

    @property
    def generated_mask(self) -> Tensor:
        """Valid generated positions; ignore other positions in prediction losses."""
        return self.position_ids >= 0

    @property
    def token_indices(self) -> Tensor:
        """Physical positions of new tokens; -1 denotes an unused event slot."""
        input_counts = (~self.input_padding_mask).sum(1)
        offsets = torch.arange(self.position_ids.size(1), device=input_counts.device)
        indices = input_counts.unsqueeze(1) + offsets.unsqueeze(0)
        return indices.masked_fill(~self.generated_mask, -1)

    @property
    def output_length(self) -> int:
        output_counts = (~self.input_padding_mask).sum(1) + self.selected_counts
        return int(output_counts.max().item())

    def gather_generated(self, hidden: Tensor) -> Tensor:
        """Gather this event's tokens from its output or a later expanded state."""
        indices = self.token_indices.clamp_min(0).unsqueeze(-1)
        generated = hidden.gather(1, indices.expand(-1, -1, hidden.size(2)))
        return generated.masked_fill(~self.generated_mask.unsqueeze(-1), 0)


@dataclass(frozen=True)
class SequenceExpansionResult:
    """One model call's tokens, padding, event metadata, and predictor loss."""

    hidden: Tensor
    key_padding_mask: Tensor
    expansion: SequenceExpansionRecord
    loss: Tensor | None = None


@dataclass
class _SequenceExpansionState(LayerState):
    """Own expansion bookkeeping and its transitions through recurrent execution."""

    sequence_expansions: tuple["SequenceExpansionRecord", ...] = field(
        default=(), kw_only=True
    )
    _expansion_position_offset: Tensor | None = field(
        default=None, kw_only=True, repr=False
    )

    @property
    def expansion_position_offset(self) -> Tensor | None:
        """Full-sequence offsets for selected tokens; None uses valid input lengths."""
        return self._expansion_position_offset

    def apply_expansion(self, result: SequenceExpansionResult) -> None:
        """Adopt expanded tokens and context; the owning layer accumulates the loss."""
        if self._expansion_position_offset is not None:
            self._expansion_position_offset = (
                self._expansion_position_offset + result.expansion.selected_counts
            )
        self.resize_expansion_masks(result.hidden.size(1))
        self.hidden = result.hidden
        self.expansion_padding_mask = result.key_padding_mask
        self.sequence_expansions = (*self.sequence_expansions, result.expansion)

    def validate_recurrent_shape(self, previous_hidden: Tensor) -> None:
        if self.hidden.shape == previous_hidden.shape:
            return
        if (
            self.hidden.ndim != 3
            or previous_hidden.ndim != 3
            or self.hidden.size(0) != previous_hidden.size(0)
            or self.hidden.size(2) != previous_hidden.size(2)
            or self.hidden.size(1) == 0
            or not self.sequence_expansions
            or self.expansion_padding_mask is None
            or self.expansion_padding_mask.shape != self.hidden.shape[:2]
            or self.sequence_expansions[-1].output_length != self.hidden.size(1)
        ):
            super().validate_recurrent_shape(previous_hidden)

    def align_recurrent_hidden(
        self,
        hidden: Tensor,
        *,
        initial: Tensor | None = None,
        source_state: LayerState | None = None,
    ) -> Tensor:
        if not self.sequence_expansions and not getattr(
            source_state, "sequence_expansions", ()
        ):
            return super().align_recurrent_hidden(
                hidden, initial=initial, source_state=source_state
            )
        if source_state is None:
            source_state = replace(self, hidden=hidden)
            previous_padding_mask = self.sequence_expansions[-1].input_padding_mask
            source_state.expansion_padding_mask = previous_padding_mask
            if source_state.expansion_padding_mask.shape != hidden.shape[:2]:
                raise ValueError(
                    "SequenceExpansion alignment requires the source_state for an earlier layout."
                )
        return align_valid_tokens(
            hidden,
            self.__resolve_padding_mask(source_state),
            self.__resolve_padding_mask(self),
            initial=initial,
        )

    def __resolve_padding_mask(self, state: "_SequenceExpansionState") -> Tensor:
        padding = state.expansion_padding_mask
        if padding is None:
            return torch.zeros(
                state.hidden.shape[:2], dtype=torch.bool, device=state.hidden.device
            )
        return padding if padding.dtype == torch.bool else padding.isneginf()

    def commit_recurrent_context(self, context: "_SequenceExpansionState") -> None:
        self.expansion_padding_mask = context.expansion_padding_mask
        self.sequence_expansions = context.sequence_expansions
        self._expansion_position_offset = context._expansion_position_offset

    def detach_recurrent_context(self) -> "_SequenceExpansionState":
        return replace(
            self,
            sequence_expansions=tuple(
                replace(
                    record,
                    count_logits=record.count_logits.detach(),
                    selected_probabilities=record.selected_probabilities.detach(),
                )
                for record in self.sequence_expansions
            ),
        )

    def select_recurrent_tokens(
        self, selected: "_SequenceExpansionState"
    ) -> "_SequenceExpansionState":
        if self.hidden.ndim != 3:
            return selected
        valid_counts = (~self.__resolve_padding_mask(self)).sum(1)
        return replace(
            selected,
            sequence_expansions=(),
            _expansion_position_offset=valid_counts,
        )

    def merge_recurrent_tokens(
        self, selected: "_SequenceExpansionState", hidden: Tensor
    ) -> "_SequenceExpansionState":
        if not selected.sequence_expansions:
            return super().merge_recurrent_tokens(selected, hidden)
        base_padding_mask = self.__resolve_padding_mask(self)
        generated_hidden, generated_padding = self.__selected_generated_tokens(selected)
        hidden, merged_padding = append_valid_tokens(
            hidden, base_padding_mask, generated_hidden, generated_padding
        )
        merged_state = replace(
            self,
            hidden=hidden,
            loss=selected.loss,
            sequence_expansions=self.sequence_expansions
            + self.__rebase_selected_expansions(
                selected.sequence_expansions, base_padding_mask
            ),
        )
        merged_state.expansion_padding_mask = merged_padding
        merged_state.resize_expansion_masks(hidden.size(1))
        return merged_state

    def __selected_generated_tokens(
        self, selected: "_SequenceExpansionState"
    ) -> tuple[Tensor, Tensor]:
        selected_input_counts = (
            ~selected.sequence_expansions[0].input_padding_mask
        ).sum(1)
        selected_output_counts = (~selected.expansion_padding_mask).sum(1)
        generated_counts = selected_output_counts - selected_input_counts
        offsets = torch.arange(
            int(generated_counts.max().item()), device=selected.hidden.device
        )
        generated_padding = offsets.unsqueeze(0) >= generated_counts.unsqueeze(1)
        generated_indices = selected_input_counts.unsqueeze(1) + offsets.unsqueeze(0)
        generated_indices = generated_indices.masked_fill(generated_padding, 0)
        generated_hidden = selected.hidden.gather(
            1, generated_indices.unsqueeze(-1).expand(-1, -1, selected.hidden.size(2))
        )
        return generated_hidden, generated_padding

    def __rebase_selected_expansions(
        self, records: tuple[SequenceExpansionRecord, ...], padding_mask: Tensor
    ) -> tuple[SequenceExpansionRecord, ...]:
        rebased_records = []
        for record in records:
            rebased_records.append(replace(record, input_padding_mask=padding_mask))
            valid_counts = (~padding_mask).sum(1) + record.selected_counts
            positions = torch.arange(
                int(valid_counts.max().item()), device=padding_mask.device
            )
            padding_mask = positions.unsqueeze(0) >= valid_counts.unsqueeze(1)
        return tuple(rebased_records)

    def resize_expansion_masks(self, sequence_length: int) -> None:
        """Encoder states have no pairwise masks to resize during expansion."""
