"""Supported self-attention and padding semantics for Soft MoE query slots."""

from typing import TYPE_CHECKING

import torch

from emperor.experts._validation.mixture import MixtureOfExpertsValidator
from emperor.experts._validation.soft import SoftMoEValidator

if TYPE_CHECKING:
    from emperor.attention._base import MultiHeadAttentionAbstract
    from emperor.attention._runtime import MultiHeadAttentionInputs


class SoftMixtureOfAttentionHeadsValidator:
    @staticmethod
    def validate_configuration(model: "MultiHeadAttentionAbstract") -> None:
        if model.cfg.use_kv_expert_models_flag:
            raise ValueError(
                "Soft MoE attention does not yet support expert key/value projections; "
                "set use_kv_expert_models_flag=False."
            )
        if not isinstance(model.batch_first_flag, bool):
            raise ValueError("Soft MoE attention requires explicit batch_first_flag.")
        if model.cfg.relative_positional_embedding_config is not None:
            raise ValueError(
                "Soft MoE attention does not support relative positions for slots."
            )
        if model.return_attention_weights_flag:
            raise ValueError(
                "Soft MoE attention does not support returning attention_weights."
            )
        MixtureOfExpertsValidator.validate_config(
            model.cfg.experts_config,
            input_dim=model.embedding_dim,
            output_dim=model.embedding_dim,
        )
        # Shared key/value projections must also preserve independent token rows.
        SoftMoEValidator.validate_expert_stack(model.cfg.projection_model_config)

    @staticmethod
    def validate_inputs(
        attention_inputs: "MultiHeadAttentionInputs",
    ) -> None:
        if (
            attention_inputs.query is not attention_inputs.key
            or attention_inputs.query is not attention_inputs.value
        ):
            raise ValueError(
                "Soft MoE attention requires self-attention with shared query/key/value inputs."
            )
        if (
            attention_inputs.static_key is not None
            or attention_inputs.static_value is not None
        ):
            raise ValueError(
                "Soft MoE attention does not support static key/value inputs."
            )
        if attention_inputs.attention_mask is not None:
            raise ValueError(
                "Soft MoE attention does not support token-pair attention masks; use a padding mask."
            )
        padding_mask = attention_inputs.key_padding_mask
        if padding_mask is None or padding_mask.dtype == torch.bool:
            return
        allowed_padding_values = (padding_mask == 0) | padding_mask.isneginf()
        if not allowed_padding_values.all():
            raise ValueError(
                "Soft MoE attention padding masks must be Boolean or additive 0/-inf."
            )
