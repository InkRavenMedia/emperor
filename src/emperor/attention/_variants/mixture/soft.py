"""Soft query slots with shared token keys/values and delayed token reconstruction."""

from dataclasses import replace

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
from emperor.nn import SoftSlotRouting


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
        self._routing: SoftSlotRouting | None = None

    def _build_output_model(self) -> Experts:
        return self.__build_experts(self.value_projection_dim, self.embedding_dim)

    def __build_experts(self, input_dim: int, output_dim: int) -> Experts:
        expert_config = replace(
            self.cfg.experts_config, input_dim=input_dim, output_dim=output_dim
        )
        return Experts(expert_config)

    def __build_soft_router(self) -> SoftMoERouter:
        soft_moe_dimensions = SoftMoEConfig(
            input_dim=self.embedding_dim, num_experts=self.top_k
        )
        soft_moe_config = self.cfg.experts_config.resolve_soft_moe_config()
        return soft_moe_config.build(soft_moe_dimensions)

    @property
    def combine_weights(self) -> Tensor | None:
        return self._routing.combine_weights if self._routing is not None else None

    def _clear_transient_state(self) -> None:
        super()._clear_transient_state()
        self._routing = None

    def _compute_qkv_projections(
        self, attention_inputs: MultiHeadAttentionInputs
    ) -> MultiHeadAttentionInputs:
        token_features = attention_inputs.query.transpose(0, 1)
        active_token_mask = self.__active_token_mask(attention_inputs.key_padding_mask)
        query_slots, routing = self.__compute_query_slots(
            token_features, active_token_mask
        )
        masked_token_features = routing.mask_inactive_tokens(token_features)
        self._routing = routing
        keys, values = self.__compute_shared_key_value_projections(
            masked_token_features, active_token_mask
        )
        slot_runtime_layout = replace(
            attention_inputs.runtime_layout,
            target_sequence_length=query_slots.shape[0],
            soft_slot_routing=routing,
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

    def __compute_query_slots(
        self, token_features: Tensor, active_token_mask: Tensor | None
    ) -> tuple[Tensor, SoftSlotRouting]:
        input_slots, routing = self.soft_router.compress_tokens(
            token_features,
            active_token_mask,
            causal_prefix_flag=self.causal_attention_mask_flag,
        )
        router_loss = self.soft_router.get_auxiliary_loss()
        if router_loss is not None:
            self._accumulate_auxiliary_loss(router_loss)
        query_slots, query_loss = self.query_model.forward_slots(
            input_slots,
            causal_prefix_flag=routing.causal_prefix_flag,
        )
        self._accumulate_auxiliary_loss(query_loss)
        return query_slots.permute(2, 0, 1, 3).contiguous(), routing

    def __compute_shared_key_value_projections(
        self, token_features: Tensor, active_token_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        sequence_token_features = token_features.transpose(0, 1)
        active_rows = None
        if active_token_mask is not None:
            active_rows = active_token_mask.transpose(0, 1).reshape(-1, 1)
        keys = self._compute_projection(
            sequence_token_features, self.key_model, active_rows
        )
        values = self._compute_projection(
            sequence_token_features, self.value_model, active_rows
        )
        return keys, values

    def __expand_attention_mask_for_slots(
        self, attention_mask: Tensor | None
    ) -> Tensor | None:
        if not self.causal_attention_mask_flag or attention_mask is None:
            return attention_mask
        return attention_mask.repeat_interleave(
            self.soft_router.slots_per_expert, dim=-2
        )

    def compute_output_projection(
        self, weighted_values: Tensor, routing: SoftSlotRouting | None = None
    ) -> Tensor:
        if routing is None:
            routing = self._routing
        if routing is None:
            raise RuntimeError(
                "Soft attention output requires the query routing weights."
            )
        attended_slots = weighted_values.permute(1, 2, 0, 3)
        output_slots, output_loss = self.output_model.forward_slots(attended_slots)
        self._accumulate_auxiliary_loss(output_loss)
        return routing.decompress_slots(output_slots).transpose(0, 1)


class SoftMixtureOfAttentionHeadsProcessor(MixtureOfAttentionHeadsProcessor):
    def _compute_attention_output(
        self,
        weighted_values: Tensor,
        runtime_layout: AttentionRuntimeLayout | None = None,
    ) -> Tensor:
        # The projector restores the original token length after output experts;
        # runtime_layout describes slot queries, including per-token prefixes in GPT.
        routing = (
            runtime_layout.soft_slot_routing if runtime_layout is not None else None
        )
        return self.projector.compute_output_projection(weighted_values, routing)
