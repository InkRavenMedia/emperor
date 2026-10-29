"""Private self-attention validation implementation."""

import math
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention._validation import MultiHeadAttentionValidator
from emperor.attention._variants.self_attention.config import (
    SelfAttentionProjectionStrategy,
)
from emperor.layers import RecurrentCompositionConfig

if TYPE_CHECKING:
    from emperor.attention._base import MultiHeadAttentionAbstract
    from emperor.attention._runtime import MultiHeadAttentionInputs


class SelfAttentionValidator(MultiHeadAttentionValidator):
    OPTIONAL_FIELDS = MultiHeadAttentionValidator.OPTIONAL_FIELDS | {"rotary_base"}

    @classmethod
    def validate(cls, model: "MultiHeadAttentionAbstract") -> None:
        super().validate(model)
        cls.validate_self_attention_dimensions_equal(model)
        cls.validate_projection_strategy(model)
        cls.validate_rotary_config(model)

    @staticmethod
    def validate_rotary_config(model):
        base = model.cfg.rotary_base
        if base is None:
            return
        if (
            isinstance(base, bool)
            or not isinstance(base, (int, float))
            or not math.isfinite(base)
            or base <= 0
        ):
            raise ValueError("rotary_base must be finite and positive")
        if (model.embedding_dim // model.cfg.num_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if model.cfg.relative_positional_embedding_config is not None:
            raise ValueError("RoPE cannot be combined with relative positional logits")
        if model.cfg.zero_attention_flag or model.cfg.add_key_value_bias_flag:
            raise ValueError("RoPE does not support synthetic key/value positions")

    @staticmethod
    def validate_projection_strategy(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        projection_strategy = model.cfg.projection_strategy
        if not isinstance(
            model.cfg.projection_model_config,
            RecurrentCompositionConfig,
        ):
            return
        output_multipliers = {
            SelfAttentionProjectionStrategy.FUSED: 3,
            SelfAttentionProjectionStrategy.FUSED_KEY_VALUE: 2,
        }
        output_multiplier = output_multipliers.get(projection_strategy)
        if output_multiplier is not None:
            raise ValueError(
                "Self-attention with RecurrentCompositionConfig requires "
                "projection_strategy=SelfAttentionProjectionStrategy.SEPARATE; "
                f"the {projection_strategy.name} strategy changes embedding_dim "
                f"to {output_multiplier} * embedding_dim."
            )

    @staticmethod
    def validate_self_attention_dimensions_equal(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        query_key_projection_dim = model.query_key_projection_dim or model.embedding_dim
        value_projection_dim = model.value_projection_dim or model.embedding_dim
        if not (
            query_key_projection_dim == value_projection_dim == model.embedding_dim
        ):
            raise RuntimeError(
                "Self attention requires query_key_projection_dim, "
                "value_projection_dim, and embedding_dim to be equal, but got "
                f"query_key_projection_dim={model.query_key_projection_dim}, "
                f"value_projection_dim={model.value_projection_dim}, "
                f"embedding_dim={model.embedding_dim}."
            )

    @classmethod
    def validate_forward_inputs(
        cls,
        model: "MultiHeadAttentionAbstract",
        attention_inputs: "MultiHeadAttentionInputs",
    ) -> None:
        super().validate_forward_inputs(model, attention_inputs)
        if model.cfg.rotary_base is not None:
            if (
                attention_inputs.static_key is not None
                or attention_inputs.static_value is not None
            ):
                raise ValueError(
                    "RoPE requires full sequences; static KV positions are unsupported"
                )
            if attention_inputs.query is not attention_inputs.key:
                raise ValueError(
                    "RoPE requires self-attention over the same full sequence"
                )
        cls.validate_query_key_value_are_same_tensor(
            attention_inputs.query,
            attention_inputs.key,
            attention_inputs.value,
        )
        cls.validate_fused_projection_inputs(
            model,
            attention_inputs.query,
            attention_inputs.key,
        )

    @staticmethod
    def validate_fused_projection_inputs(
        model: "MultiHeadAttentionAbstract",
        query: Tensor,
        key: Tensor,
    ) -> None:
        if (
            model.cfg.projection_strategy == SelfAttentionProjectionStrategy.FUSED
            and query is not key
        ):
            raise RuntimeError(
                "SelfAttentionProjectionStrategy.FUSED requires query, key, and "
                "value to be the same tensor; use FUSED_KEY_VALUE when query and "
                "context differ."
            )

    @staticmethod
    def validate_query_key_value_are_same_tensor(
        query: Tensor, key: Tensor, value: Tensor
    ) -> None:
        del query
        if key is not value:
            raise RuntimeError(
                "Self attention requires the key and value to be the same tensor."
            )
