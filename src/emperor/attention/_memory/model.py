"""Generate one attention memory embedding from each reduced sequence chunk."""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.attention._memory.config import AttentionSourceMemoryConfig
from emperor.attention._memory.context import MemoryContext
from emperor.attention._memory.validation import AttentionSourceMemoryValidator
from emperor.config import ConfigBase
from emperor.layers import Layer, LayerState
from emperor.linears import LinearAbstract, LinearLayerConfig
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.attention._runtime import MultiHeadAttentionInputs


class AttentionSourceMemory(Module):
    VALIDATOR = AttentionSourceMemoryValidator

    def __init__(
        self,
        cfg: AttentionSourceMemoryConfig,
        overrides: AttentionSourceMemoryConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: AttentionSourceMemoryConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate_config(self.cfg)
        self.embedding_dim: int = self.cfg.embedding_dim
        self.model_config: ConfigBase | None = self.cfg.model_config
        self.context = MemoryContext(self.cfg, self.embedding_dim)
        self.generator_model_config = self.__resolve_generator_config()
        self.model = self._build_memory_model()

    def __resolve_generator_config(self) -> ConfigBase:
        generator_config = self.model_config
        if generator_config is None:
            generator_config = LinearLayerConfig(bias_flag=True)
        self.VALIDATOR.validate_generator_config(generator_config)
        return generator_config

    def _build_memory_model(self) -> Module:
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
        memory_rows = memory_embeddings.transpose(0, 1).reshape(-1, self.embedding_dim)
        valid_memory_rows = valid_memory_mask.transpose(0, 1).reshape(-1, 1)
        return memory_rows, valid_memory_rows, auxiliary_loss

    def generate_source_memory(
        self,
        source: Tensor,
        *,
        excluded_token_mask: Tensor | None = None,
        batch_first_flag: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        """Generate chunk memory and return tokens in the source sequence layout."""
        memory_inputs = source
        valid_row_mask = None
        if excluded_token_mask is not None:
            memory_inputs = source.masked_fill(excluded_token_mask.unsqueeze(-1), 0)
            sequence_first_mask = excluded_token_mask
            if batch_first_flag:
                sequence_first_mask = excluded_token_mask.transpose(0, 1)
            valid_row_mask = ~sequence_first_mask.reshape(-1, 1)
        if batch_first_flag:
            memory_inputs = memory_inputs.transpose(0, 1)
        memory_rows, _, auxiliary_loss = self(memory_inputs, skip_mask=valid_row_mask)
        memory_tokens = memory_rows.reshape(
            -1, memory_inputs.size(1), self.embedding_dim
        )
        if batch_first_flag:
            memory_tokens = memory_tokens.transpose(0, 1)
        return memory_tokens, auxiliary_loss

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

    def append_to_projections(
        self,
        projections: "MultiHeadAttentionInputs",
        memory_projection: Tensor,
        source_length: int,
    ) -> "MultiHeadAttentionInputs":
        chunk_size = self.context.chunk_size
        expected_memory_length = (source_length + chunk_size - 1) // chunk_size
        self.VALIDATOR.validate_projected_output(
            memory_projection,
            projections.key,
            memory_length=expected_memory_length,
        )
        keys_with_memory = torch.cat((projections.key, memory_projection), dim=0)
        values_with_memory = torch.cat((projections.value, memory_projection), dim=0)
        memory_sequence_length = memory_projection.size(0)
        runtime_layout_with_memory = projections.runtime_layout.with_source_extension(
            memory_sequence_length
        )
        return replace(
            projections,
            key=keys_with_memory,
            value=values_with_memory,
            runtime_layout=runtime_layout_with_memory,
        )
