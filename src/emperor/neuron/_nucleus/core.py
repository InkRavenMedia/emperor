from typing import TYPE_CHECKING

from torch import Tensor

from emperor.layers import LayerState
from emperor.neuron._nucleus.validation import NucleusValidator
from emperor.nn import Module
from emperor.transformer import TransformerEncoderLayerState

if TYPE_CHECKING:
    from emperor.neuron._config import NucleusConfig


class Nucleus(Module):
    VALIDATOR = NucleusValidator

    def __init__(
        self,
        cfg: "NucleusConfig",
        overrides: "NucleusConfig | None" = None,
    ):
        super().__init__()
        self.cfg: NucleusConfig = self._override_config(cfg, overrides)
        self.model_config = self.cfg.model_config
        self.VALIDATOR.validate(self)
        self.model = self.model_config.build()

    def forward(
        self,
        input: Tensor | TransformerEncoderLayerState,
    ) -> Tensor | TransformerEncoderLayerState:
        if isinstance(input, TransformerEncoderLayerState):
            if input.hidden.ndim != 3:
                raise ValueError(
                    "A sequence nucleus requires [batch, tokens, features]."
                )
        else:
            self.VALIDATOR.validate_forward_input(input)
            record_loss = LayerState.loss_recorder(self)
            if record_loss is not None:
                run_with_state = getattr(self.model, "forward_state", None)
                if callable(run_with_state):
                    state = run_with_state(input)
                    if not isinstance(state, LayerState):
                        raise TypeError(
                            "The nucleus model's forward_state must return LayerState"
                        )
                else:
                    state = LayerState.run_tensor_module(self.model, input)
                record_loss(state.loss)
                return state.hidden
        return self.model(input)
