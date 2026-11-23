"""Shared Soft MoE scoring with separate token and slot distributions."""

import torch
from torch import Tensor, nn
from torch.func import functional_call

from emperor.experts._config import SoftMoEConfig
from emperor.experts._validation.soft import SoftMoEValidator
from emperor.nn import Module, SoftSlotRouting
from emperor.sampler import RouterConfig, RouterModel


class SoftMoERouter(Module):
    VALIDATOR = SoftMoEValidator

    def __init__(
        self, cfg: SoftMoEConfig, overrides: SoftMoEConfig | None = None
    ) -> None:
        self.VALIDATOR.validate_config_type(cfg)
        self.VALIDATOR.validate_overrides_type(overrides)
        super().__init__()
        self.cfg: SoftMoEConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.input_dim: int = self.cfg.input_dim
        self.num_experts: int = self.cfg.num_experts
        self.slots_per_expert: int = self.cfg.slots_per_expert
        self.num_slots: int = self.num_experts * self.slots_per_expert
        self.router_config: RouterConfig = self.cfg.router_config
        self.grouping_flag: bool = self.VALIDATOR.has_grouping(
            self.router_config.model_config
        )
        self.router: RouterModel = self.__build_router()
        initial_logit_scale = torch.ones(())
        self.scale = nn.Parameter(initial_logit_scale)
        self.auxiliary_loss: Tensor | None = None

    def get_auxiliary_loss(self) -> Tensor | None:
        return self.auxiliary_loss

    def __build_router(self) -> RouterModel:
        router_overrides = RouterConfig(
            input_dim=self.input_dim,
            num_experts=self.num_experts,
            output_dim=self.num_slots,
        )
        return self.router_config.build(router_overrides)

    def compress_tokens(
        self,
        tokens: Tensor,
        active_token_mask: Tensor | None = None,
        *,
        causal_prefix_flag: bool = False,
    ) -> tuple[Tensor, SoftSlotRouting]:
        dispatch_weights, combine_weights = self(
            tokens, active_token_mask, causal_prefix_flag=causal_prefix_flag
        )
        routing = SoftSlotRouting(
            active_token_mask, causal_prefix_flag=causal_prefix_flag
        )
        slots = routing.compress_with_routing_weights(
            tokens, dispatch_weights, combine_weights
        )
        return slots, routing

    def forward(
        self,
        X: Tensor,
        skip_mask: Tensor | None = None,
        *,
        causal_prefix_flag: bool = False,
    ) -> tuple[Tensor, Tensor]:
        self.auxiliary_loss = None
        token_features = X
        self.VALIDATOR.validate_inputs(token_features, self.input_dim, skip_mask)
        self.VALIDATOR.validate_grouped_token_inputs(
            self, token_features, causal_prefix_flag
        )
        self.VALIDATOR.validate_parameter_device(token_features, self.scale)
        routing = SoftSlotRouting(skip_mask, causal_prefix_flag=causal_prefix_flag)
        scoring_features = routing.prepare_scoring_features(token_features)
        normalized_token_features = self.__normalize_active_token_features(
            scoring_features
        )
        token_slot_logits = self.__compute_token_slot_logits(
            token_features, normalized_token_features, scoring_features.dtype, skip_mask
        )
        return routing.compute_routing_weights(token_slot_logits)

    def __normalize_active_token_features(
        self, routing_token_features: Tensor
    ) -> Tensor:
        squared_token_features = routing_token_features.square()
        squared_token_norms = squared_token_features.sum(dim=-1, keepdim=True)
        stabilized_squared_token_norms = squared_token_norms + 1e-6
        inverse_token_norms = torch.rsqrt(stabilized_squared_token_norms)
        return routing_token_features * inverse_token_norms

    def __compute_token_slot_logits(
        self,
        token_features: Tensor,
        normalized_token_features: Tensor,
        routing_dtype: torch.dtype,
        skip_mask: Tensor | None,
    ) -> Tensor:
        slot_logit_parameters, slot_logit_buffers = (
            self.__cast_slot_logit_parameters_and_buffers(token_features, routing_dtype)
        )
        flattened_token_features = normalized_token_features.reshape(-1, self.input_dim)
        scoring_features = flattened_token_features
        compact_active_tokens = skip_mask is not None and not self.grouping_flag
        scoring_input_mask = None
        if skip_mask is not None and self.grouping_flag:
            scoring_input_mask = skip_mask.reshape(-1, 1)
        if compact_active_tokens:
            active_token_indices = skip_mask.reshape(-1).nonzero(as_tuple=True)[0]
            scoring_features = flattened_token_features.index_select(
                0, active_token_indices
            )
        # Keep routing scores in the promoted dtype during mixed-precision execution.
        with torch.autocast(device_type=token_features.device.type, enabled=False):
            slot_logit_state = functional_call(
                self.router,
                (slot_logit_parameters, slot_logit_buffers),
                (scoring_features,),
                {"input_mask": scoring_input_mask},
                strict=True,
            )
        self.__restore_slot_logit_buffers(slot_logit_buffers)
        self.auxiliary_loss = slot_logit_state.loss
        token_slot_logits = slot_logit_state.hidden
        if compact_active_tokens:
            all_token_slot_logits = token_slot_logits.new_zeros(
                flattened_token_features.size(0), self.num_slots
            )
            token_slot_logits = all_token_slot_logits.index_copy(
                0, active_token_indices, token_slot_logits
            )
        token_slot_logits_shape = (
            *token_features.shape[:2],
            self.num_experts,
            self.slots_per_expert,
        )
        token_slot_logits = token_slot_logits.reshape(token_slot_logits_shape)
        routing_logit_scale = self.scale.to(routing_dtype)
        return token_slot_logits * routing_logit_scale

    def __cast_slot_logit_parameters_and_buffers(
        self, token_features: Tensor, routing_dtype: torch.dtype
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        slot_logit_parameters = {}
        for parameter_name, parameter in self.router.named_parameters():
            self.VALIDATOR.validate_parameter_device(token_features, parameter)
            slot_logit_parameters[parameter_name] = parameter.to(routing_dtype)
        slot_logit_buffers = {
            buffer_name: self.__cast_floating_buffer_for_routing(buffer, routing_dtype)
            for buffer_name, buffer in self.router.named_buffers()
        }
        return slot_logit_parameters, slot_logit_buffers

    @torch.no_grad()
    def __restore_slot_logit_buffers(
        self, slot_logit_buffers: dict[str, Tensor]
    ) -> None:
        for buffer_name, buffer in self.router.named_buffers():
            updated_buffer = slot_logit_buffers[buffer_name]
            if updated_buffer is not buffer:
                buffer.copy_(updated_buffer)

    def __cast_floating_buffer_for_routing(
        self, buffer: Tensor, routing_dtype: torch.dtype
    ) -> Tensor:
        if buffer.is_floating_point():
            return buffer.to(routing_dtype)
        return buffer
