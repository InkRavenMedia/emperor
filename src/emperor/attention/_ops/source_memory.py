"""Independent memory projection using the key/value model architecture."""

import torch.nn as nn
from torch import Tensor

from emperor.attention._config import MultiHeadAttentionConfig
from emperor.attention._validation import AttentionSourceMemoryValidator
from emperor.config import ConfigBase
from emperor.experts._layers.mixture import MixtureOfExperts
from emperor.layers import Layer
from emperor.nn import Module


class AttentionSourceMemory(Module):
    VALIDATOR = AttentionSourceMemoryValidator

    def __init__(
        self, cfg: MultiHeadAttentionConfig, *, model_config: ConfigBase | None = None
    ):
        super().__init__()
        self.cfg = cfg
        self.model_config = model_config
        self.VALIDATOR.validate_config(cfg)
        self.model = self._build_memory_model()

    def _build_memory_model(self) -> nn.Module:
        if self.model_config is not None:
            return self.model_config.build()
        input_dim = self.cfg.embedding_dim
        output_dim = self.cfg.query_key_projection_dim or input_dim
        model_config = self.cfg.projection_model_config
        overrides = type(model_config)(input_dim=input_dim, output_dim=output_dim)
        return model_config.build(overrides)

    def forward(
        self,
        source: Tensor,
        *,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        token_rows = source.reshape(-1, source.size(-1))
        if isinstance(self.model, MixtureOfExperts):
            return self.model(token_rows, probabilities, indices, skip_mask)
        state = Layer.run_model_from_hidden(self.model, token_rows)
        return state.hidden, skip_mask, state.loss
