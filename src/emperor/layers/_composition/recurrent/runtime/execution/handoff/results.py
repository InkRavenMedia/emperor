from __future__ import annotations

from dataclasses import replace

import torch

from emperor.layers._composition.recurrent.runtime.execution.interface import (
    RecurrentExecutionResult,
)


def accumulate_caller_loss_once(
    caller_loss: torch.Tensor | None,
    recurrent_result: RecurrentExecutionResult,
) -> RecurrentExecutionResult:
    if recurrent_result.loss is None:
        return replace(recurrent_result, loss=caller_loss)
    return replace(
        recurrent_result,
        loss=(
            recurrent_result.loss
            if caller_loss is None
            else caller_loss + recurrent_result.loss
        ),
    )


def blend_recurrent_branch_hidden(
    source: RecurrentExecutionResult,
    target: RecurrentExecutionResult,
    transition_weight: float,
) -> torch.Tensor:
    if transition_weight == 0.0:
        return source.hidden
    if transition_weight == 1.0:
        return target.hidden
    source_hidden = source.hidden
    if target.context_state is not None:
        source_hidden = target.context_state.align_recurrent_hidden(
            source_hidden, source_state=source.context_state
        )
    return (1.0 - transition_weight) * source_hidden + transition_weight * target.hidden


def blend_recurrent_branch_losses(
    common_loss: torch.Tensor | None,
    source_loss: torch.Tensor | None,
    target_loss: torch.Tensor | None,
    transition_weight: float,
) -> torch.Tensor | None:
    if transition_weight == 0.0:
        return source_loss
    if transition_weight == 1.0:
        return target_loss

    blended_loss = common_loss
    for branch_loss, branch_weight in (
        (source_loss, 1.0 - transition_weight),
        (target_loss, transition_weight),
    ):
        if branch_loss is None:
            continue
        branch_loss_delta = (
            branch_loss if common_loss is None else branch_loss - common_loss
        )
        weighted_branch_loss_delta = branch_weight * branch_loss_delta
        blended_loss = (
            weighted_branch_loss_delta
            if blended_loss is None
            else blended_loss + weighted_branch_loss_delta
        )
    return blended_loss
