"""Soft query slots with shared token keys/values and delayed token reconstruction."""

from dataclasses import replace

import torch
from torch import Tensor

from emperor.attention._ops.projection import ProjectorBase
from emperor.attention._runtime import AttentionRuntimeLayout, MultiHeadAttentionInputs
from emperor.attention._variants.mixture.config import MixtureOfAttentionHeadsConfig
from emperor.attention._variants.mixture.processing import (
    MixtureOfAttentionHeadsProcessor,
)
from emperor.experts import SoftMoEConfig
from emperor.experts._layers.experts import Experts
from emperor.experts._routing.soft import SoftMoERouter


class SoftMixtureOfAttentionHeadsProjector(ProjectorBase):
    def __init__(self, cfg: MixtureOfAttentionHeadsConfig) -> None:
        super().__init__(cfg)
        # The shared attention pipeline calls its expert branch count top_k.
        self.top_k: int = cfg.experts_config.num_experts
        self.causal_attention_mask_flag: bool = cfg.causal_attention_mask_flag
        self.query_model: Experts = self.__build_experts(
            self.embedding_dim, self.query_key_projection_dim
        )
        self.key_model = self._create_model(
            self.embedding_dim, self.query_key_projection_dim
        )
        self.value_model = self._create_model(
            self.embedding_dim, self.value_projection_dim
        )
        self.soft_router: SoftMoERouter = self.__build_soft_router()
        self.combine_weights: Tensor | None = None

    def _build_output_model(self) -> Experts:
        return self.__build_experts(self.value_projection_dim, self.embedding_dim)

    def __build_experts(self, input_dim: int, output_dim: int) -> Experts:
        expert_config = replace(
            self.cfg.experts_config, input_dim=input_dim, output_dim=output_dim
        )
        return Experts(expert_config)

    def __build_soft_router(self) -> SoftMoERouter:
        routing_dimensions = SoftMoEConfig(
            input_dim=self.embedding_dim, num_experts=self.top_k
        )
        return self.cfg.experts_config.soft_moe_config.build(routing_dimensions)

    def _clear_transient_state(self) -> None:
        super()._clear_transient_state()
        self.combine_weights = None

    def compute_qkv_projections(
        self, attention_inputs: MultiHeadAttentionInputs
    ) -> MultiHeadAttentionInputs:
        token_features = attention_inputs.query.transpose(0, 1)
        active_token_mask = self.__active_token_mask(attention_inputs.key_padding_mask)
        masked_token_features = self.__mask_token_features(
            token_features, active_token_mask
        )
        query_slots = self.__compute_query_slots(
            masked_token_features, active_token_mask
        )
        keys, values = self.__compute_shared_key_value_projections(
            masked_token_features
        )
        slot_runtime_layout = replace(
            attention_inputs.runtime_layout,
            target_sequence_length=query_slots.shape[0],
        )
        slot_attention_mask = self.__expand_attention_mask_for_slots(
            attention_inputs.attention_mask
        )
        return replace(
            attention_inputs,
            query=query_slots,
            key=keys,
            value=values,
            attention_mask=slot_attention_mask,
            runtime_layout=slot_runtime_layout,
        )

    def __active_token_mask(self, padding_mask: Tensor | None) -> Tensor | None:
        if padding_mask is None:
            return None
        # The shared mask handler has canonicalized Boolean masks to 0/-inf.
        return (~padding_mask.isneginf()).unsqueeze(-1)

    def __mask_token_features(
        self, token_features: Tensor, active_token_mask: Tensor | None
    ) -> Tensor:
        if active_token_mask is None:
            return token_features
        return token_features.masked_fill(~active_token_mask, 0)

    def __compute_query_slots(
        self, token_features: Tensor, active_token_mask: Tensor | None
    ) -> Tensor:
        dispatch_weights, self.combine_weights = self.soft_router(
            token_features,
            active_token_mask,
            causal_prefix_flag=self.causal_attention_mask_flag,
        )
        query_slots, query_loss = self.__execute_query_experts(
            token_features, dispatch_weights, active_token_mask
        )
        self._accumulate_auxiliary_loss(query_loss)
        return query_slots.permute(2, 0, 1, 3).contiguous()

    def __execute_query_experts(
        self,
        token_features: Tensor,
        dispatch_weights: Tensor,
        active_token_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if self.causal_attention_mask_flag:
            return self.query_model.forward_causal_soft_tokens(
                token_features, dispatch_weights, active_token_mask
            )
        return self.query_model.forward_soft_tokens(
            token_features, dispatch_weights, active_token_mask
        )

    def __compute_shared_key_value_projections(
        self, token_features: Tensor
    ) -> tuple[Tensor, Tensor]:
        sequence_token_features = token_features.transpose(0, 1)
        keys = self._compute_projection(sequence_token_features, self.key_model)
        values = self._compute_projection(sequence_token_features, self.value_model)
        return keys, values

    def __expand_attention_mask_for_slots(
        self, attention_mask: Tensor | None
    ) -> Tensor | None:
        if not self.causal_attention_mask_flag or attention_mask is None:
            return attention_mask
        return attention_mask.repeat_interleave(
            self.soft_router.slots_per_expert, dim=-2
        )

    def compute_output_projection(self, weighted_values: Tensor) -> Tensor:
        attended_slots = weighted_values.permute(1, 2, 0, 3)
        output_slots, output_loss = self.output_model.forward_slots(attended_slots)
        self._accumulate_auxiliary_loss(output_loss)
        token_outputs = self.__combine_output_slots(output_slots)
        return token_outputs.transpose(0, 1)

    def __combine_output_slots(self, output_slots: Tensor) -> Tensor:
        combine_weights = self.combine_weights
        if combine_weights is None:
            raise RuntimeError(
                "Soft attention output requires the query routing weights."
            )
        with torch.autocast(device_type=output_slots.device.type, enabled=False):
            combination_features = output_slots.to(combine_weights.dtype)
            token_outputs = self.__reconstruct_tokens(
                combine_weights, combination_features
            )
        return token_outputs.to(output_slots.dtype)

    def __reconstruct_tokens(
        self, combine_weights: Tensor, output_slots: Tensor
    ) -> Tensor:
        if self.causal_attention_mask_flag:
            batch_size, sequence_length, expert_count, slots_per_expert = (
                combine_weights.shape
            )
            prefix_slot_shape = (
                batch_size,
                expert_count,
                sequence_length,
                slots_per_expert,
                self.embedding_dim,
            )
            prefix_output_slots = output_slots.reshape(prefix_slot_shape)
            return torch.einsum("btes,betsd->btd", combine_weights, prefix_output_slots)
        return torch.einsum("btes,besd->btd", combine_weights, output_slots)


class SoftMixtureOfAttentionHeadsProcessor(MixtureOfAttentionHeadsProcessor):
    def _compute_attention_output(
        self,
        weighted_values: Tensor,
        runtime_layout: AttentionRuntimeLayout | None = None,
    ) -> Tensor:
        # The projector restores the original token length after output experts;
        # runtime_layout describes slot queries, including per-token prefixes in GPT.
        return self.projector.compute_output_projection(weighted_values)
