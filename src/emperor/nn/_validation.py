import torch
from torch import Tensor


class SoftSlotRoutingValidator:
    @classmethod
    def validate_token_slot_layout(
        cls, token_slots: Tensor, active_token_mask: Tensor | None
    ) -> None:
        cls.validate_routing_tensor(token_slots)
        cls.validate_active_token_mask(active_token_mask, token_slots)

    @staticmethod
    def validate_routing_tensor(weights: object) -> None:
        if not isinstance(weights, Tensor) or not weights.is_floating_point():
            raise TypeError("Soft slot routing requires floating-point tensors.")
        if weights.ndim not in (3, 4) or 0 in weights.shape[1:]:
            raise ValueError("Soft slot routing requires nonempty token and slot axes.")

    @staticmethod
    def validate_active_token_mask(mask: object, weights: Tensor) -> None:
        if mask is None:
            return
        if (
            not isinstance(mask, Tensor)
            or mask.dtype != torch.bool
            or mask.shape != (*weights.shape[:2], 1)
        ):
            raise ValueError(
                "Active token mask must be Boolean with shape [batch, tokens, 1]."
            )
        if mask.device != weights.device:
            raise ValueError(
                "Active token mask and routing weights must be on the same device."
            )

    @staticmethod
    def validate_matching_weights(dispatch: object, combine: object) -> None:
        if not isinstance(dispatch, Tensor) or not isinstance(combine, Tensor):
            raise TypeError("Dispatch and combine weights must be tensors.")
        if (
            combine.shape != dispatch.shape
            or combine.dtype != dispatch.dtype
            or combine.device != dispatch.device
        ):
            raise ValueError(
                "Dispatch and combine weights must have the same shape, dtype and device."
            )

    @classmethod
    def validate_pooling_inputs(
        cls, tokens: object, weights: Tensor, active_token_mask: Tensor | None
    ) -> None:
        cls.validate_token_slot_layout(weights, active_token_mask)
        if (
            not isinstance(tokens, Tensor)
            or tokens.ndim != 3
            or tokens.shape[:2] != weights.shape[:2]
        ):
            raise ValueError(
                "Token features must match the batch and token axes of the route."
            )
        if tokens.device != weights.device:
            raise ValueError(
                "Token features and routing weights must be on the same device."
            )

    @staticmethod
    def require_combine_weights(weights: Tensor | None) -> Tensor:
        if weights is None:
            raise RuntimeError("Compress tokens before decompressing slots.")
        return weights

    @staticmethod
    def validate_processed_slots(
        slots: object, weights: Tensor, causal_prefix_flag: bool
    ) -> None:
        slots_per_group = weights.size(-1)
        if causal_prefix_flag:
            slots_per_group *= weights.size(1)
        expected_axes = (weights.size(0), *weights.shape[2:-1], slots_per_group)
        if not isinstance(slots, Tensor) or slots.shape[:-1] != expected_axes:
            raise ValueError(
                "Processed slots must match the batch and slot axes of the route."
            )
        if slots.device != weights.device:
            raise ValueError(
                "Processed slots and routing weights must be on the same device."
            )
