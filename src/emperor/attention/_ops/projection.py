"""Private attention projection operations."""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from emperor.attention._ops.source_memory import AttentionSourceMemory
from emperor.attention._validation import AttentionSourceMemoryValidator
from emperor.layers import Layer
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.attention._config import MultiHeadAttentionConfig
    from emperor.attention._runtime import (
        MultiHeadAttentionInputs,
    )
    from emperor.config import ConfigBase
    from emperor.layers import (
        LayerStackConfig,
        RecurrentCompositionConfig,
    )


class ProjectorBase(Module):
    VALIDATOR = AttentionSourceMemoryValidator

    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        *,
        source_memory_model_config: "ConfigBase | None" = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim: int = self.cfg.embedding_dim
        self.query_key_projection_dim: int = self.cfg.query_key_projection_dim
        self.value_projection_dim: int = self.cfg.value_projection_dim
        self.source_memory_flag: bool = bool(self.cfg.source_memory_flag)
        self.projection_model_config: LayerStackConfig | RecurrentCompositionConfig = (
            self.cfg.projection_model_config
        )
        self.__resolve_kv_dimensions()
        self.output_model = self._build_output_model()
        self.source_memory = self.__maybe_initialize_source_memory(
            source_memory_model_config
        )
        self.auxiliary_loss: Tensor | None = None

    def __maybe_initialize_source_memory(
        self,
        model_config: "ConfigBase | None",
    ) -> AttentionSourceMemory | None:
        if self.source_memory_flag:
            return AttentionSourceMemory(self.cfg, model_config=model_config)
        return None

    def __resolve_kv_dimensions(self) -> None:
        if not self.query_key_projection_dim:
            self.query_key_projection_dim = self.embedding_dim
        if not self.value_projection_dim:
            self.value_projection_dim = self.embedding_dim

    def _build_output_model(self) -> nn.Module:
        return self._create_model(self.value_projection_dim, self.embedding_dim)

    def _create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        overrides = type(self.projection_model_config)(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        return self.projection_model_config.build(overrides)

    def _compute_memory_projection(
        self, source: Tensor, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        active_rows = None
        if key_padding_mask is not None:
            active_rows = (~key_padding_mask.isneginf()).transpose(0, 1).reshape(-1, 1)
        memory, _, loss = self.source_memory(source, skip_mask=active_rows)
        if loss is not None:
            self._accumulate_auxiliary_loss(loss)
        return memory.reshape(source.size(0), source.size(1), -1)

    def _compute_projection(
        self,
        tensor: Tensor,
        model: nn.Module,
        input_mask: Tensor | None = None,
    ) -> Tensor:
        sequence_length, batch_size, embedding_dim = tensor.shape
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        tensor_reshaped = tensor.view(-1, embedding_dim)
        projection = self._forward_accumulating_loss(model, tensor_reshaped, input_mask)
        return projection.view(sequence_length, batch_size, -1)

    def _forward_accumulating_loss(
        self,
        model: nn.Module,
        tensor: Tensor,
        input_mask: Tensor | None = None,
    ) -> Tensor:
        projection_state = Layer.run_model_from_hidden(
            model,
            tensor,
            input_mask=input_mask,
        )
        if projection_state.loss is not None:
            self._accumulate_auxiliary_loss(projection_state.loss)
        return projection_state.hidden

    def compute_output_projection(
        self,
        weighted_values: Tensor,
    ) -> Tensor:
        uses_unflattened_sequence_batch_layout = weighted_values.dim() == 3
        if uses_unflattened_sequence_batch_layout:
            return self._compute_projection(weighted_values, self.output_model)
        return self._forward_accumulating_loss(self.output_model, weighted_values)

    def compute_qkv_projections(
        self,
        attention_inputs: "MultiHeadAttentionInputs",
    ) -> "MultiHeadAttentionInputs":
        if self.source_memory_flag:
            self.VALIDATOR.validate_inputs(attention_inputs)
        projections = self._compute_qkv_projections(attention_inputs)
        return self.__maybe_append_memory_sources(projections, attention_inputs)

    def __maybe_append_memory_sources(
        self,
        projections: "MultiHeadAttentionInputs",
        attention_inputs: "MultiHeadAttentionInputs",
    ) -> "MultiHeadAttentionInputs":
        if not self.source_memory_flag:
            return projections
        memory_projection = self._compute_memory_projection(
            attention_inputs.key, attention_inputs.key_padding_mask
        )
        self.VALIDATOR.validate_projected_output(memory_projection, projections.key)
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

    def _compute_qkv_projections(
        self,
        attention_inputs: "MultiHeadAttentionInputs",
    ) -> "MultiHeadAttentionInputs":
        raise NotImplementedError(
            "compute_qkv_projections must be implemented by subclass."
        )

    def _accumulate_auxiliary_loss(self, loss: Tensor) -> None:
        if self.auxiliary_loss is None:
            self.auxiliary_loss = loss
        else:
            self.auxiliary_loss = self.auxiliary_loss + loss

    def _clear_transient_state(self) -> None:
        self.auxiliary_loss = None

    def _get_auxiliary_loss(self) -> "Tensor | None":
        return self.auxiliary_loss
