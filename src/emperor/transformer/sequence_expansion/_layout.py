"""Shared token placement and alignment for sequence expansion.

These tensor operations consume boolean padding masks, where True marks padding.
They leave their inputs intact so earlier expansion passes remain usable in backward.
"""

import torch
from torch import Tensor


def append_valid_tokens(
    hidden: Tensor,
    padding_mask: Tensor,
    generated_hidden: Tensor,
    generated_padding_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Pack existing then generated tokens per sample, followed by batch padding."""
    input_valid_mask = ~padding_mask
    generated_valid_mask = ~generated_padding_mask
    input_token_counts = input_valid_mask.sum(1)
    output_token_counts = input_token_counts + generated_valid_mask.sum(1)
    output_sequence_length = int(output_token_counts.max().item())

    input_positions = input_valid_mask.cumsum(1) - 1
    generated_positions = (
        input_token_counts.unsqueeze(1) + generated_valid_mask.cumsum(1) - 1
    )
    input_batch_indices, input_token_indices = input_valid_mask.nonzero(as_tuple=True)
    generated_batch_indices, generated_token_indices = generated_valid_mask.nonzero(
        as_tuple=True
    )

    output_hidden = hidden.new_zeros(
        hidden.size(0), output_sequence_length, hidden.size(2)
    )
    output_hidden = output_hidden.index_put(
        (
            input_batch_indices,
            input_positions[input_batch_indices, input_token_indices],
        ),
        hidden[input_batch_indices, input_token_indices],
    )
    output_hidden = output_hidden.index_put(
        (
            generated_batch_indices,
            generated_positions[generated_batch_indices, generated_token_indices],
        ),
        generated_hidden[generated_batch_indices, generated_token_indices],
    )
    output_positions = torch.arange(output_sequence_length, device=hidden.device)
    output_padding_mask = output_positions.unsqueeze(
        0
    ) >= output_token_counts.unsqueeze(1)
    return output_hidden, output_padding_mask


def align_valid_tokens(
    hidden: Tensor,
    source_padding_mask: Tensor,
    target_padding_mask: Tensor,
    *,
    initial: Tensor | None = None,
) -> Tensor:
    """Match valid-token order across layouts and initialize missing target tokens."""
    source_position_ids = ((~source_padding_mask).cumsum(1) - 1).masked_fill(
        source_padding_mask, -1
    )
    target_position_ids = ((~target_padding_mask).cumsum(1) - 1).masked_fill(
        target_padding_mask, -1
    )
    source_index_by_position = torch.full(
        (hidden.size(0), max(hidden.size(1), target_padding_mask.size(1)) + 1),
        -1,
        device=hidden.device,
        dtype=torch.long,
    )
    source_token_indices = torch.arange(hidden.size(1), device=hidden.device).expand_as(
        source_position_ids
    )
    # Reserve lookup column zero for padding positions, whose logical ID is -1.
    source_index_by_position.scatter_(1, source_position_ids + 1, source_token_indices)
    matched_source_indices = source_index_by_position.gather(1, target_position_ids + 1)
    aligned_hidden = hidden.gather(
        1,
        matched_source_indices.clamp_min(0)
        .unsqueeze(-1)
        .expand(-1, -1, hidden.size(2)),
    )
    initial_hidden = 0 if initial is None else initial.to(hidden)
    aligned_hidden = torch.where(
        (matched_source_indices < 0).unsqueeze(-1), initial_hidden, aligned_hidden
    )
    return aligned_hidden.masked_fill((target_position_ids < 0).unsqueeze(-1), 0)
