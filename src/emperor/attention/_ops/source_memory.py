"""Generate one attention memory embedding from each reduced sequence chunk."""

import torch.nn as nn
from torch import Tensor

from emperor.attention._config import (
    AttentionSourceMemoryConfig,
    MultiHeadAttentionConfig,
)
from emperor.attention._ops.memory_context import MemoryContext
from emperor.attention._validation import AttentionSourceMemoryValidator
from emperor.config import ConfigBase
from emperor.layers import Layer, LayerState
from emperor.linears import LinearAbstract, LinearLayerConfig
from emperor.nn import Module


class AttentionSourceMemory(Module):
    VALIDATOR = AttentionSourceMemoryValidator

    def __init__(self, cfg: MultiHeadAttentionConfig) -> None:
        super().__init__()
        self.cfg: MultiHeadAttentionConfig = cfg
        self.VALIDATOR.validate_config(self.cfg)
        self.embedding_dim: int = self.cfg.embedding_dim
        self.source_memory_config: AttentionSourceMemoryConfig = (
            self.cfg.source_memory_config
        )
        self.context = MemoryContext(self.source_memory_config, self.embedding_dim)
        self.generator_model_config = self.__resolve_generator_config()
        self.model = self._build_memory_model()

    def __resolve_generator_config(self) -> ConfigBase:
        generator_config = self.source_memory_config.model_config
        if generator_config is None:
            generator_config = LinearLayerConfig(bias_flag=True)
        self.VALIDATOR.validate_generator_config(generator_config)
        return generator_config

    def _build_memory_model(self) -> nn.Module:
        return self._build_from_config(
            self.generator_model_config,
            input_dim=self.embedding_dim,
            output_dim=self.embedding_dim,
        )

    def forward(
        self,
        source: Tensor,
        *,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        context_embeddings, valid_memory_mask = self.context(source, skip_mask)
        memory_embeddings, auxiliary_loss = self.__generate_context_memory(
            context_embeddings, valid_memory_mask
        )
        # Attention consumes sequence-first flattened rows: [M * B, D].
        memory_rows = memory_embeddings.transpose(0, 1).reshape(-1, self.embedding_dim)
        valid_memory_rows = valid_memory_mask.transpose(0, 1).reshape(-1, 1)
        return memory_rows, valid_memory_rows, auxiliary_loss

    def __generate_context_memory(
        self, context_embeddings: Tensor, valid_memory_mask: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        memory_embeddings = context_embeddings.new_zeros(context_embeddings.shape)
        if not valid_memory_mask.any():
            return memory_embeddings, None
        generator_inputs = context_embeddings[valid_memory_mask]
        memory_state = self.__run_generator(generator_inputs)
        self.VALIDATOR.validate_generator_output(memory_state.hidden, generator_inputs)
        memory_embeddings[valid_memory_mask] = memory_state.hidden.to(
            dtype=memory_embeddings.dtype
        )
        return memory_embeddings, memory_state.loss

    def __run_generator(self, generator_inputs: Tensor) -> LayerState:
        if not isinstance(self.model, LinearAbstract):
            return Layer.run_model_from_hidden(self.model, generator_inputs)
        memory_state = LayerState(hidden=self.model(generator_inputs))
        auxiliary_loss_getter = getattr(self.model, "get_auxiliary_loss", None)
        memory_state.loss = (
            auxiliary_loss_getter() if auxiliary_loss_getter is not None else None
        )
        return memory_state
