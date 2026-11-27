import torch
from torch import Tensor

from emperor.nn._validation import SoftSlotRoutingValidator


class SoftSlotRouting:
    VALIDATOR = SoftSlotRoutingValidator

    def __init__(
        self,
        active_token_mask: Tensor | None = None,
        *,
        causal_prefix_flag: bool = False,
    ) -> None:
        self.active_token_mask = active_token_mask
        self.causal_prefix_flag = causal_prefix_flag
        self._routing_weights: tuple[Tensor, Tensor] | None = None

    @property
    def combine_weights(self) -> Tensor | None:
        return self._routing_weights[1] if self._routing_weights is not None else None

    def compress_tokens(self, tokens: Tensor, slot_logits: Tensor) -> Tensor:
        dispatch_weights, combine_weights = self.compute_routing_weights(slot_logits)
        return self.compress_with_routing_weights(
            tokens, dispatch_weights, combine_weights
        )

    def compress_with_routing_weights(
        self, tokens: Tensor, dispatch_weights: Tensor, combine_weights: Tensor
    ) -> Tensor:
        self.VALIDATOR.validate_matching_weights(dispatch_weights, combine_weights)
        slots = self.pool_tokens_with_dispatch_weights(tokens, dispatch_weights)
        self._routing_weights = (dispatch_weights, combine_weights)
        return slots

    def decompress_slots(self, slots: Tensor) -> Tensor:
        combine_weights = self.VALIDATOR.require_combine_weights(self.combine_weights)
        self.VALIDATOR.validate_processed_slots(
            slots, combine_weights, self.causal_prefix_flag
        )
        with torch.autocast(device_type=slots.device.type, enabled=False):
            slot_features = slots.to(combine_weights.dtype)
            if self.causal_prefix_flag:
                tokens = self.__combine_slots_for_each_prefix(
                    slot_features, combine_weights
                )
            else:
                tokens = torch.einsum(
                    "bt...,b...d->btd", combine_weights, slot_features
                )
        return tokens.to(slots.dtype)

    def compute_routing_weights(self, slot_logits: Tensor) -> tuple[Tensor, Tensor]:
        self.VALIDATOR.validate_token_slot_layout(slot_logits, self.active_token_mask)
        flat_scores = self.prepare_scoring_features(slot_logits.flatten(2))
        dispatch = self.__compute_dispatch_weights(flat_scores)
        combine = flat_scores.softmax(dim=-1)
        return (
            self.mask_inactive_tokens(dispatch).reshape_as(slot_logits),
            self.mask_inactive_tokens(combine).reshape_as(slot_logits),
        )

    def compute_dispatch_weights(self, slot_logits: Tensor) -> Tensor:
        """Normalize each slot across its sources when only pooling is needed."""
        self.VALIDATOR.validate_token_slot_layout(slot_logits, self.active_token_mask)
        scores = self.prepare_scoring_features(slot_logits.flatten(2))
        dispatch = self.__compute_dispatch_weights(scores)
        return self.mask_inactive_tokens(dispatch).reshape_as(slot_logits)

    def __compute_dispatch_weights(self, scores: Tensor) -> Tensor:
        if self.causal_prefix_flag:
            return self.__compute_prefix_update_weights(scores)
        if self.active_token_mask is not None:
            scores = scores.masked_fill(~self.active_token_mask, -torch.inf)
            has_active_tokens = self.active_token_mask.any(dim=1, keepdim=True)
            scores = torch.where(has_active_tokens, scores, 0)
        return scores.softmax(dim=1)

    def __compute_prefix_update_weights(self, scores: Tensor) -> Tensor:
        if self.active_token_mask is not None:
            lowest_finite_score = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~self.active_token_mask, lowest_finite_score)
        prefix_log_normalizers = torch.logcumsumexp(scores, dim=1)
        return (scores - prefix_log_normalizers).exp()

    def prepare_scoring_features(self, features: Tensor) -> Tensor:
        scoring_dtype = (
            torch.float64 if features.dtype == torch.float64 else torch.float32
        )
        return self.mask_inactive_tokens(features.to(scoring_dtype))

    def mask_inactive_tokens(self, features: Tensor) -> Tensor:
        if self.active_token_mask is None:
            return features
        return features.masked_fill(~self.active_token_mask, 0)

    def pool_tokens_with_dispatch_weights(
        self, tokens: Tensor, dispatch_weights: Tensor
    ) -> Tensor:
        """Pool batched tokens or a shared [sources, features] bank for each route."""
        self.VALIDATOR.validate_pooling_inputs(
            tokens, dispatch_weights, self.active_token_mask
        )
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            token_features = self.mask_inactive_tokens(tokens).to(
                dispatch_weights.dtype
            )
            if self.causal_prefix_flag:
                slots = self.__pool_causal_prefixes(token_features, dispatch_weights)
            else:
                slots = torch.einsum(
                    "bt...,btd->b...d", dispatch_weights, token_features
                )
        return slots.to(tokens.dtype)

    def __pool_causal_prefixes(
        self, tokens: Tensor, dispatch_weights: Tensor
    ) -> Tensor:
        batch_size, token_count, feature_dim = tokens.shape
        slot_axes = dispatch_weights.shape[2:]
        broadcast_token_shape = (batch_size, *(1 for _ in slot_axes), feature_dim)
        running_slots = tokens.new_zeros((batch_size, *slot_axes, feature_dim))
        prefix_slots: list[Tensor] = []
        for token_index in range(token_count):
            current_token = tokens[:, token_index].reshape(broadcast_token_shape)
            update_weights = dispatch_weights[:, token_index].unsqueeze(-1)
            retained_slots = (1 - update_weights) * running_slots
            running_slots = retained_slots + update_weights * current_token
            prefix_slots.append(running_slots)
        return torch.stack(prefix_slots, dim=-3).flatten(-3, -2)

    def __combine_slots_for_each_prefix(
        self, slots: Tensor, combine_weights: Tensor
    ) -> Tensor:
        token_count, slots_per_group = combine_weights.size(1), combine_weights.size(-1)
        prefix_slots = torch.unflatten(
            slots, -2, (token_count, slots_per_group)
        ).movedim(-3, 1)
        return torch.einsum("bt...,bt...d->btd", combine_weights, prefix_slots)
