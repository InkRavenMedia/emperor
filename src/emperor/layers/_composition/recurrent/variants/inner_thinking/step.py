from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar

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
    from emperor.layers._composition.recurrent.variants.standard.state import (
        StandardRecurrentState,
    )
    from emperor.layers._state import LayerState
    from emperor.sampler import TokenSamplingResult


class InnerThinkingRecurrentStep(StandardRecurrentStep):
    VALIDATOR: ClassVar[type[InnerThinkingRecurrentValidator]] = (
        InnerThinkingRecurrentValidator
    )

    def __init__(
        self,
        cfg: InnerThinkingRecurrentConfig,
        get_module: Callable[[str], nn.Module],
        get_parameter: Callable[[str], nn.Parameter],
    ) -> None:
        super().__init__(cfg)
        self.__get_module = get_module
        self.__get_parameter = get_parameter
        self.thinking_step_scale = (
            1.0 if cfg.thinking_step_scale is None else float(cfg.thinking_step_scale)
        )

    def compute_output(
        self,
        recurrent_state: StandardRecurrentState,
        target_state: StandardRecurrentState | None = None,
    ) -> LayerState:
        transition_input = self._maybe_reinject_original_input(recurrent_state)
        transition_state = recurrent_state.layer_state_for_transition(transition_input)
        return self.pipeline.process_recurrent_transition(
            transition_state,
            run_model=lambda state: self.__run_thinking_block(state, recurrent_state),
            recurrent_state=recurrent_state,
            target_state=target_state,
        )

    def __run_thinking_block(
        self, state: LayerState, recurrent_state: StandardRecurrentState
    ) -> LayerState:
        block_input = state.hidden
        self.VALIDATOR.validate_block_layout(self.block_model, block_input)
        if recurrent_state.transition_index == 0:
            output_state = self.__run_initial_thinking_transition(state)
        else:
            output_state = self.__run_selected_thinking_transition(
                state, recurrent_state
            )
        self.VALIDATOR.validate_transition_output(
            output_state, block_input, self.cfg.output_dim
        )
        return output_state

    def __run_initial_thinking_transition(self, state: LayerState) -> LayerState:
        output_state = self._run_block(state)
        step_weight = self.__thinking_step_weight(0, output_state)
        weighted_hidden = output_state.hidden * step_weight
        return replace(output_state, hidden=weighted_hidden)

    def __thinking_step_weight(
        self, transition_index: int, output_state: LayerState
    ) -> Tensor:
        return self.__get_parameter("thinking_step_weights")[transition_index].to(
            output_state.hidden
        )

    def __run_selected_thinking_transition(
        self,
        state: LayerState,
        recurrent_state: StandardRecurrentState,
    ) -> LayerState:
        transition_index = recurrent_state.transition_index
        token_sample = self.__sample_thinking_tokens(state, transition_index)
        selection = InnerThinkingTokenSelection(
            token_sample.indices, token_sample.valid, token_sample.sequence_length
        )
        selected_context, output_state = self.__run_block_on_selected_tokens(
            state, selection
        )
        selected_token_weights = token_sample.weights.unsqueeze(-1)
        step_weight = self.__thinking_step_weight(transition_index, output_state)
        selected_update = self.__compute_selected_token_update(
            selected_context, output_state, selected_token_weights, step_weight
        )
        updated_hidden = selection.scatter_update(
            recurrent_state.hidden, selected_update
        )
        generated_state = self.__weight_generated_tokens(output_state, step_weight)
        return state.merge_recurrent_tokens(generated_state, updated_hidden)

    def __sample_thinking_tokens(
        self, state: LayerState, transition_index: int
    ) -> TokenSamplingResult:
        padding_mask = getattr(state, "target_key_padding_mask", None)
        if padding_mask is None:
            padding_mask = getattr(state, "key_padding_mask", None)
        sampler = self.__get_module(f"samplers.{transition_index - 1}")
        return sampler(state.hidden, padding_mask)

    def __run_block_on_selected_tokens(
        self, state: LayerState, selection: InnerThinkingTokenSelection
    ) -> tuple[LayerState, LayerState]:
        attention_heads = self.__attention_heads(state)
        sampled_state = selection.select_state(state, attention_heads=attention_heads)
        selected_state = state.select_recurrent_tokens(sampled_state)
        selected_context = replace(selected_state)
        output_state = self._run_block(selected_state)
        return selected_context, output_state

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
            for module in self.block_model.modules()
            if hasattr(module, "num_heads")
        }
        self.VALIDATOR.validate_attention_heads(head_counts)
        return next(iter(head_counts), 1)

    def __compute_selected_token_update(
        self,
        selected_context: LayerState,
        output_state: LayerState,
        selected_token_weights: Tensor,
        step_weight: Tensor,
    ) -> Tensor:
        aligned_hidden = selected_context.align_recurrent_hidden(
            output_state.hidden, source_state=output_state
        )
        return (
            aligned_hidden
            * selected_token_weights
            * step_weight
            * self.thinking_step_scale
        )

    def __weight_generated_tokens(
        self, output_state: LayerState, step_weight: Tensor
    ) -> LayerState:
        # Newly generated tokens receive only the thinking-step weight; their
        # owner merges them into the full sequence, reusing its padding slots.
        generated_hidden = output_state.hidden * step_weight * self.thinking_step_scale
        return replace(output_state, hidden=generated_hidden)
