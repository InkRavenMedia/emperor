from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from emperor.layers._composition.recurrent.validation import (
    InnerThinkingRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.inner_thinking.state import (
    InnerThinkingTokenSelection,
)
from emperor.layers._composition.recurrent.variants.standard.step import (
    StandardRecurrentStep,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor, nn

    from emperor.layers._composition.recurrent.config import (
        InnerThinkingRecurrentConfig,
    )
    from emperor.layers._composition.recurrent.utilities import RecurrentUtilities
    from emperor.layers._composition.recurrent.variants.standard.state import (
        StandardRecurrentState,
    )
    from emperor.layers._state import LayerState


class InnerThinkingRecurrentStep(StandardRecurrentStep):
    VALIDATOR = InnerThinkingRecurrentValidator

    def __init__(
        self,
        utilities: RecurrentUtilities,
        output_dim: int,
        get_module: Callable[[str], nn.Module],
        get_parameter: Callable[[str], nn.Parameter],
        reinject_original_hidden: bool,
        cfg: InnerThinkingRecurrentConfig,
    ) -> None:
        super().__init__(utilities, output_dim, get_module, reinject_original_hidden)
        self.__get_parameter = get_parameter
        self.thinking_step_scale = (
            1.0 if cfg.thinking_step_scale is None else float(cfg.thinking_step_scale)
        )

    def _run_block(
        self, state: LayerState, recurrent_state: StandardRecurrentState
    ) -> LayerState:
        transition_index = recurrent_state.transition_index
        previous_hidden = recurrent_state.hidden
        transition_input = state.hidden
        InnerThinkingRecurrentValidator.validate_block_layout(
            self._get_module("block_model"), transition_input
        )
        if transition_index == 0:
            return self.__run_initial_thinking_transition(state, transition_input)
        return self.__run_selected_thinking_transition(
            state,
            transition_input,
            transition_index=transition_index,
            previous_hidden=previous_hidden,
        )

    def __run_initial_thinking_transition(
        self, state: LayerState, transition_input: Tensor
    ) -> LayerState:
        output_state = self._get_module("block_model")(state)
        self.VALIDATOR.validate_transition_output(
            output_state, transition_input, expected_feature_dim=self.output_dim
        )
        return replace(
            output_state,
            hidden=output_state.hidden
            * self.__get_parameter("thinking_step_weights")[0].to(output_state.hidden),
        )

    def __run_selected_thinking_transition(
        self,
        state: LayerState,
        transition_input: Tensor,
        transition_index: int,
        previous_hidden: Tensor,
    ) -> LayerState:
        padding_mask = getattr(state, "target_key_padding_mask", None)
        if padding_mask is None:
            padding_mask = getattr(state, "key_padding_mask", None)
        token_sample = self._get_module(f"samplers.{transition_index - 1}")(
            transition_input, padding_mask
        )
        selection = InnerThinkingTokenSelection(
            token_sample.indices, token_sample.valid, token_sample.sequence_length
        )
        selected_state = state.select_recurrent_tokens(
            selection.select_state(state, attention_heads=self.__attention_heads(state))
        )
        selected_input = selected_state.hidden
        selected_context = replace(selected_state)
        output_state = self._get_module("block_model")(selected_state)
        self.VALIDATOR.validate_transition_output(
            output_state, selected_input, expected_feature_dim=self.output_dim
        )
        selected_token_weights = token_sample.weights.unsqueeze(-1)
        step_weight = self.__get_parameter("thinking_step_weights")[
            transition_index
        ].to(output_state.hidden)
        selected_token_update = (
            selected_context.align_recurrent_hidden(
                output_state.hidden, source_state=output_state
            )
            * selected_token_weights
            * step_weight
            * self.thinking_step_scale
        )
        hidden = selection.scatter_update(previous_hidden, selected_token_update)
        # Newly generated tokens receive only the thinking-step weight; their
        # owner merges them into the full sequence, reusing its padding slots.
        generated_state = replace(
            output_state,
            hidden=output_state.hidden * step_weight * self.thinking_step_scale,
        )
        return state.merge_recurrent_tokens(generated_state, hidden)

    def __attention_heads(self, state: LayerState) -> int:
        if not any(
            getattr(state, name, None) is not None
            for name in (
                "attention_mask",
                "target_attention_mask",
                "cross_attention_mask",
            )
        ):
            return 1
        head_counts = {
            module.num_heads
            for module in self._get_module("block_model").modules()
            if hasattr(module, "num_heads")
        }
        InnerThinkingRecurrentValidator.validate_attention_heads(head_counts)
        return next(iter(head_counts), 1)
