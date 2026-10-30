from torch import Tensor

from emperor.experts._state import MixtureOfExpertsLayerState
from emperor.experts._validation.layer import MixtureOfExpertsLayerValidator
from emperor.layers import Layer


class MixtureOfExpertsLayer(Layer):
    VALIDATOR = MixtureOfExpertsLayerValidator

    def _handle_layer_input(
        self, state: MixtureOfExpertsLayerState
    ) -> MixtureOfExpertsLayerState:
        if self.model.soft_moe_config is None:
            return state
        self.model.VALIDATOR.validate_forward_inputs(
            self.model,
            state.hidden,
            state.probabilities,
            state.indices,
            state.skip_mask,
        )
        state.hidden = self.__mask_inactive_token_features(
            state.hidden, state.skip_mask
        )
        return state

    def __mask_inactive_token_features(
        self, token_features: Tensor, skip_mask: Tensor | None
    ) -> Tensor:
        if skip_mask is None:
            return token_features
        return token_features.masked_fill(~skip_mask, 0)

    def _handle_model_processing(
        self,
        state: MixtureOfExpertsLayerState,
    ) -> MixtureOfExpertsLayerState:
        main_model_input = state.hidden
        routing_options = (
            {"causal_prefix_flag": True} if state.causal_prefix_flag else {}
        )
        output, skip_mask, loss = self.model(
            main_model_input,
            state.probabilities,
            state.indices,
            state.skip_mask,
            **routing_options,
        )
        state.skip_mask = skip_mask
        state.loss = loss if state.loss is None else state.loss + loss
        self.VALIDATOR.validate_output_rows(self, main_model_input, output)
        state.hidden = output
        return state
