"""Depth-mapped Soft MoE scores with separate token and slot distributions."""

import torch
from torch import Tensor, nn
from torch.func import functional_call

from emperor.augmentations.adaptive_parameters import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)
from emperor.experts._config import SoftMoEConfig
from emperor.experts._validation.soft import SoftMoEValidator
from emperor.nn import Module


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
        self.slot_logit_model: DepthMappingLayerStack = self.__build_slot_logit_model()
        initial_logit_scale = torch.ones(())
        self.scale = nn.Parameter(initial_logit_scale)

    def __build_slot_logit_model(self) -> DepthMappingLayerStack:
        slot_logit_overrides = DepthMappingHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.slots_per_expert,
            generator_depth=self.num_experts,
        )
        return self.cfg.model_config.build(slot_logit_overrides)

    def forward(
        self,
        X: Tensor,
        skip_mask: Tensor | None = None,
        *,
        causal_prefix_flag: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return dispatch/combine weights, or prefix-update/combine weights.

        In causal mode, each update weight inserts the current token into the
        running slot average; its complement retains the previous prefix.
        """
        token_features = X
        self.VALIDATOR.validate_inputs(token_features, self.input_dim, skip_mask)
        self.VALIDATOR.validate_parameter_device(token_features, self.scale)
        routing_dtype = self.__resolve_routing_dtype(token_features)
        normalized_token_features = self.__normalize_active_token_features(
            token_features, skip_mask, routing_dtype
        )
        token_slot_logits = self.__compute_token_slot_logits(
            token_features, normalized_token_features, routing_dtype
        )
        return self.__compute_dispatch_and_combine_weights(
            token_slot_logits, skip_mask, causal_prefix_flag
        )

    def __resolve_routing_dtype(self, token_features: Tensor) -> torch.dtype:
        if token_features.dtype == torch.float64:
            return torch.float64
        return torch.float32

    def __normalize_active_token_features(
        self,
        token_features: Tensor,
        skip_mask: Tensor | None,
        routing_dtype: torch.dtype,
    ) -> Tensor:
        masked_token_features = self.__mask_inactive_token_features(
            token_features, skip_mask
        )
        routing_token_features = masked_token_features.to(routing_dtype)
        squared_token_features = routing_token_features.square()
        squared_token_norms = squared_token_features.sum(dim=-1, keepdim=True)
        stabilized_squared_token_norms = squared_token_norms + 1e-6
        inverse_token_norms = torch.rsqrt(stabilized_squared_token_norms)
        return routing_token_features * inverse_token_norms

    def __mask_inactive_token_features(
        self, token_features: Tensor, skip_mask: Tensor | None
    ) -> Tensor:
        if skip_mask is None:
            return token_features
        return token_features.masked_fill(~skip_mask, 0)

    def __compute_token_slot_logits(
        self,
        token_features: Tensor,
        normalized_token_features: Tensor,
        routing_dtype: torch.dtype,
    ) -> Tensor:
        slot_logit_parameters, slot_logit_buffers = (
            self.__cast_slot_logit_parameters_and_buffers(token_features, routing_dtype)
        )
        flattened_token_features = normalized_token_features.reshape(-1, self.input_dim)
        token_slot_logits = functional_call(
            self.slot_logit_model,
            (slot_logit_parameters, slot_logit_buffers),
            (flattened_token_features,),
            strict=True,
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
        for parameter_name, parameter in self.slot_logit_model.named_parameters():
            self.VALIDATOR.validate_parameter_device(token_features, parameter)
            slot_logit_parameters[parameter_name] = parameter.to(routing_dtype)
        slot_logit_buffers = {
            buffer_name: self.__cast_floating_buffer_for_routing(buffer, routing_dtype)
            for buffer_name, buffer in self.slot_logit_model.named_buffers()
        }
        return slot_logit_parameters, slot_logit_buffers

    def __cast_floating_buffer_for_routing(
        self, buffer: Tensor, routing_dtype: torch.dtype
    ) -> Tensor:
        if buffer.is_floating_point():
            return buffer.to(routing_dtype)
        return buffer

    def __compute_dispatch_and_combine_weights(
        self,
        token_slot_logits: Tensor,
        skip_mask: Tensor | None,
        causal_prefix_flag: bool,
    ) -> tuple[Tensor, Tensor]:
        active_token_mask = self.__expand_token_mask_for_slots(skip_mask)
        dispatch_weights = self.__compute_dispatch_weights(
            token_slot_logits, active_token_mask, causal_prefix_flag
        )
        flattened_slot_logits = token_slot_logits.flatten(2)
        flattened_combine_weights = flattened_slot_logits.softmax(dim=-1)
        combine_weights = flattened_combine_weights.reshape_as(token_slot_logits)
        if active_token_mask is not None:
            dispatch_weights = dispatch_weights.masked_fill(~active_token_mask, 0)
            combine_weights = combine_weights.masked_fill(~active_token_mask, 0)
        return dispatch_weights, combine_weights

    def __expand_token_mask_for_slots(self, skip_mask: Tensor | None) -> Tensor | None:
        if skip_mask is None:
            return None
        return skip_mask.unsqueeze(-1)

    def __compute_dispatch_weights(
        self,
        token_slot_logits: Tensor,
        active_token_mask: Tensor | None,
        causal_prefix_flag: bool,
    ) -> Tensor:
        if causal_prefix_flag:
            return self.__compute_prefix_update_weights(
                token_slot_logits, active_token_mask
            )
        dispatch_logits = token_slot_logits
        if active_token_mask is not None:
            dispatch_logits = token_slot_logits.masked_fill(
                ~active_token_mask, -torch.inf
            )
            sequence_has_active_tokens = active_token_mask.any(dim=1, keepdim=True)
            dispatch_logits = torch.where(
                sequence_has_active_tokens, dispatch_logits, 0
            )
        return dispatch_logits.softmax(dim=1)

    def __compute_prefix_update_weights(
        self, token_slot_logits: Tensor, active_token_mask: Tensor | None
    ) -> Tensor:
        prefix_logits = token_slot_logits
        if active_token_mask is not None:
            # Finite sentinels keep logcumsumexp gradients defined for empty prefixes.
            inactive_logit = torch.finfo(token_slot_logits.dtype).min
            prefix_logits = token_slot_logits.masked_fill(
                ~active_token_mask, inactive_logit
            )
        prefix_log_normalizers = torch.logcumsumexp(prefix_logits, dim=1)
        normalized_prefix_logits = prefix_logits - prefix_log_normalizers
        return normalized_prefix_logits.exp()
