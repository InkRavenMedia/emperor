"""Output alignment and loss accounting shared by both depth handoffs."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import replace

import torch

from emperor.layers._composition.recurrent.runtime.execution.interface import (
    RecurrentExecutionAdapter,
    RecurrentExecutionResult,
    _StateT,
)


def accumulate_caller_loss_once(
    adapter: RecurrentExecutionAdapter[_StateT],
    caller_loss: torch.Tensor | None,
    recurrent_result: RecurrentExecutionResult,
) -> RecurrentExecutionResult:
    if recurrent_result.loss is None:
        return replace(recurrent_result, loss=caller_loss)
    return replace(
        recurrent_result,
        loss=adapter._accumulate_auxiliary_loss(
            caller_loss,
            recurrent_result.loss,
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
