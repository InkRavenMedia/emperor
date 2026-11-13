from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from emperor.layers._composition.residual.base import (
    ResidualConnectionAbstract,
    ResidualState,
)
from emperor.layers._composition.residual.config import AttentionResidualConfig
from emperor.layers._composition.residual.validation import AttentionResidualValidator
from emperor.layers._composition.residual.variants.attention.state import (
    AttentionResidualState,
)

if TYPE_CHECKING:
    from emperor.layers import LayerStack, LayerStackConfig
    from emperor.layers._state import LayerState
    from emperor.linears import LinearAbstract, LinearLayerConfig


class AttentionResidual(ResidualConnectionAbstract):
    VALIDATOR = AttentionResidualValidator

    def __init__(
        self,
        cfg: AttentionResidualConfig,
        overrides: AttentionResidualConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.model_config: LayerStackConfig | LinearLayerConfig | None = (
            self.cfg.model_config
        )
        self.block_size: int = self.cfg.block_size
        self.rms_norm_epsilon: float = self.cfg.rms_norm_epsilon

        self.query: nn.Parameter | None = None
        self.query_model: LayerStack | LinearAbstract | None = None
        self.__initialize_learned_query_or_query_model()
        self.key_norm = self.__build_key_norm()

    def __initialize_learned_query_or_query_model(self) -> None:
        if self.model_config is None:
            self.query = nn.Parameter(torch.zeros(self.residual_dim))
            return
        self.query_model = self._build_from_config(
            self.model_config,
            input_dim=self.residual_dim,
            output_dim=self.residual_dim,
        )

    def __build_key_norm(self) -> nn.RMSNorm:
        return nn.RMSNorm(
            self.residual_dim,
            eps=self.rms_norm_epsilon,
            elementwise_affine=True,
        )

    def new_state(self, initial_source: Tensor) -> AttentionResidualState:
        self.VALIDATOR.validate_source(initial_source, residual_dim=self.residual_dim)
        return AttentionResidualState(
            initial_source=initial_source,
            block_size=self.block_size,
        )

    def apply_to_layer_state(
        self,
        state: LayerState,
        previous: Tensor,
    ) -> LayerState:
        if state.residual_state is not None:
            return super().apply_to_layer_state(state, previous)

        created_residual_state = self.new_state(previous)
        self.VALIDATOR.validate_created_attention_state(
            self,
            created_residual_state,
        )
        state.residual_state = created_residual_state
        return super().apply_to_layer_state(state, previous)

    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        residual_state: ResidualState | None = None,
    ) -> Tensor:
        self.VALIDATOR.validate_attention_forward_inputs(self, current, residual_state)
        query = self.__resolve_query(current)
        residual_sources = self.__append_and_stack_residual_sources(
            residual_state, current
        )
        accumulator_sources = self.__convert_to_accumulator_precision(
            residual_sources,
        )
        normalized_source_keys = self.__normalize_residual_source_keys(
            accumulator_sources,
        )
        depth_weights = self.__calculate_residual_depth_weights(
            normalized_source_keys, query
        )
        mixed_residual_sources = self.__mix_depth_weighted_residual_sources(
            accumulator_sources, depth_weights
        )
        return mixed_residual_sources.to(dtype=residual_sources.dtype)

    def __resolve_query(self, current: Tensor) -> Tensor:
        if self.query is not None:
            return self.query

        is_single_feature_vector = current.ndim == 1
        query_model_input = current
        if is_single_feature_vector:
            query_model_input = current.unsqueeze(0)
        generated_query = self.query_model.forward_hidden(query_model_input)
        self.VALIDATOR.validate_query_model_output(generated_query, query_model_input)
        return generated_query.reshape(current.shape)

    @staticmethod
    def __append_and_stack_residual_sources(
        attention_state: AttentionResidualState,
        current: Tensor,
    ) -> Tensor:
        attention_state.append(current)
        return torch.stack(attention_state.sources, dim=0)

    def __convert_to_accumulator_precision(
        self,
        residual_sources: Tensor,
    ) -> Tensor:
        accumulator_dtype = self.__resolve_accumulator_dtype(
            residual_sources.dtype,
        )
        return residual_sources.to(dtype=accumulator_dtype)

    @staticmethod
    def __resolve_accumulator_dtype(values_dtype: torch.dtype) -> torch.dtype:
        if values_dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        else:
            return values_dtype

    def __normalize_residual_source_keys(
        self,
        accumulator_sources: Tensor,
    ) -> Tensor:
        return F.rms_norm(
            accumulator_sources,
            normalized_shape=(self.residual_dim,),
            weight=self.key_norm.weight.to(dtype=accumulator_sources.dtype),
            eps=self.rms_norm_epsilon,
        )

    def __calculate_residual_depth_weights(
        self,
        normalized_source_keys: Tensor,
        query: Tensor,
    ) -> Tensor:
        accumulator_query = query.to(dtype=normalized_source_keys.dtype)
        query_weighted_source_keys = normalized_source_keys * accumulator_query
        depth_attention_logits = torch.sum(query_weighted_source_keys, dim=-1)
        return torch.softmax(depth_attention_logits, dim=0)

    @staticmethod
    def __mix_depth_weighted_residual_sources(
        accumulator_sources: Tensor,
        depth_weights: Tensor,
    ) -> Tensor:
        feature_broadcast_depth_weights = depth_weights.unsqueeze(-1)
        depth_weighted_residual_sources = (
            feature_broadcast_depth_weights * accumulator_sources
        )
        return torch.sum(depth_weighted_residual_sources, dim=0)
