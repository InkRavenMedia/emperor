"""Private mixer-attention validation implementation."""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor._validation import (
    ValidatorBase,
    _configuration_visits,
    _first_adaptive_grouping_path,
)
from emperor.layers import LayerState

if TYPE_CHECKING:
    from emperor.attention._variants.mixer.layer import MixerAttention


class MixerAttentionValidator(ValidatorBase):
    OPTIONAL_FIELDS = {
        "source_memory_config",
        "token_model_config",
    }

    @classmethod
    def validate(cls, model: "MixerAttention") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            embedding_dim=model.embedding_dim,
            sequence_length=model.sequence_length,
        )
        if model.cfg.source_memory_config is not None:
            cls.validate_memory_config(model.cfg.source_memory_config)
        cls._validate_mixing_model_config(model.mixing_model_config)
        if model.cfg.token_model_config is not None:
            cls.validate_token_model_config(model.cfg.token_model_config)

    @staticmethod
    def validate_token_model_config(config) -> None:
        from emperor.experts import MixtureOfExpertsModelConfig, SoftMoEConfig

        if not isinstance(config, MixtureOfExpertsModelConfig):
            raise TypeError("token_model_config must be a MixtureOfExpertsModelConfig.")
        if not any(
            isinstance(visit.config, SoftMoEConfig)
            for visit in _configuration_visits(
                config, "MixerAttention.token_model_config"
            )
        ):
            raise ValueError(
                "Mixer token_model_config requires sequence-aware Soft MoE."
            )

    @staticmethod
    def validate_memory_config(memory_config: object) -> None:
        from emperor.attention._memory.config import AttentionSourceMemoryConfig
        from emperor.memory import DynamicMemoryConfig

        if not isinstance(
            memory_config, (DynamicMemoryConfig, AttentionSourceMemoryConfig)
        ):
            raise TypeError(
                "source_memory_config must be a DynamicMemoryConfig or AttentionSourceMemoryConfig."
            )
        memory_validator = memory_config.registry_owner().VALIDATOR
        memory_validator.validate_source_memory_config(memory_config)

    @classmethod
    def _validate_mixing_model_config(cls, mixing_model_config: object) -> None:
        from emperor.experts import MixtureOfExpertsModelConfig, SoftMoEConfig
        from emperor.layers import LayerStackConfig, RecurrentCompositionConfig

        if not isinstance(
            mixing_model_config,
            (
                LayerStackConfig,
                MixtureOfExpertsModelConfig,
                RecurrentCompositionConfig,
            ),
        ):
            raise TypeError(
                "mixing_model_config must be a LayerStackConfig or "
                "RecurrentCompositionConfig, or a MixtureOfExpertsModelConfig for "
                "MixerAttention, got "
                f"{type(mixing_model_config).__name__}."
            )
        grouping_path = _first_adaptive_grouping_path(
            mixing_model_config,
            root="MixerAttentionConfig.mixing_model_config",
        )
        if grouping_path is not None:
            raise ValueError(
                "Adaptive parameter grouping is not supported by MixerAttention "
                "because its flattened rows represent embedding channels, not "
                f"sequence rows. Found grouping at {grouping_path}."
            )
        cls._validate_required_nested_mixing_configs(mixing_model_config)
        uses_soft_moe = any(
            isinstance(visit.config, SoftMoEConfig)
            for visit in _configuration_visits(mixing_model_config, "MixerAttention")
        )
        if uses_soft_moe and not isinstance(
            mixing_model_config, MixtureOfExpertsModelConfig
        ):
            raise ValueError(
                "Soft MoE token mixing requires a direct MixtureOfExpertsModelConfig."
            )

    @classmethod
    def _validate_required_nested_mixing_configs(cls, config: object) -> None:
        from emperor.experts import MixtureOfExpertsModelConfig
        from emperor.layers import RecurrentCompositionConfig

        if isinstance(config, MixtureOfExpertsModelConfig):
            if config.stack_config is None:
                raise ValueError(
                    "stack_config is required for a MixerAttention "
                    "MixtureOfExpertsModelConfig."
                )
            cls._validate_required_nested_mixing_configs(config.stack_config)
        elif isinstance(config, RecurrentCompositionConfig):
            missing_fields = config._missing_transition_config_fields()
            if missing_fields:
                missing_field_list = " and ".join(missing_fields)
                raise ValueError(
                    f"{missing_field_list} is required for a MixerAttention "
                    f"{type(config).__name__}."
                )
            for transition_config in config._transition_configs():
                cls._validate_required_nested_mixing_configs(transition_config)

    @classmethod
    def validate_forward_inputs(
        cls,
        model: "MixerAttention",
        q: "Tensor",
        k: "Tensor",
        v: "Tensor",
        key_padding_mask: "Tensor | None",
        attention_mask: "Tensor | None",
        static_k: "Tensor | None",
        static_v: "Tensor | None",
    ) -> None:
        cls._validate_self_processing(q, k, v)
        cls._validate_unsupported_inputs(
            attention_mask,
            static_k,
            static_v,
        )
        cls._validate_runtime_tensor(model, q)
        cls._validate_padding_mask(model, q, key_padding_mask)

    @staticmethod
    def _validate_self_processing(q: object, k: object, v: object) -> None:
        if q is not k or q is not v:
            raise RuntimeError(
                "MixerAttention requires q, k, and v to be the same tensor object."
            )

    @staticmethod
    def _validate_unsupported_inputs(
        attention_mask: "Tensor | None",
        static_k: "Tensor | None",
        static_v: "Tensor | None",
    ) -> None:
        if attention_mask is not None:
            raise RuntimeError("MixerAttention does not support attention masks.")
        if static_k is not None or static_v is not None:
            raise RuntimeError(
                "MixerAttention does not support static key/value projections."
            )

    @staticmethod
    def _validate_runtime_tensor(
        model: "MixerAttention",
        tensor: "Tensor",
    ) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"MixerAttention input must be a Tensor, got {type(tensor).__name__}."
            )
        if tensor.dim() != 3:
            raise RuntimeError(
                f"MixerAttention input must be rank three, got rank {tensor.dim()}."
            )
        if not torch.is_floating_point(tensor):
            raise RuntimeError("MixerAttention input must be floating point.")
        if any(dimension <= 0 for dimension in tensor.shape):
            raise RuntimeError(
                "MixerAttention batch, sequence, and embedding dimensions "
                "must be non-empty."
            )

        sequence_axis = 0
        if model.batch_first_flag:
            sequence_axis = 1
        actual_sequence_length = tensor.size(sequence_axis)
        if actual_sequence_length > model.sequence_length:
            raise RuntimeError(
                "MixerAttention sequence length must be at most "
                f"{model.sequence_length}, got {actual_sequence_length}."
            )
        actual_embedding_dim = tensor.size(-1)
        if actual_embedding_dim != model.embedding_dim:
            raise RuntimeError(
                "MixerAttention embedding width must be exactly "
                f"{model.embedding_dim}, got {actual_embedding_dim}."
            )

    @staticmethod
    def _validate_padding_mask(
        model: "MixerAttention", tensor: Tensor, padding_mask: Tensor | None
    ) -> None:
        if padding_mask is None:
            return
        if model.batch_first_flag:
            expected_mask_shape = tensor.shape[:2]
        else:
            expected_mask_shape = (tensor.size(1), tensor.size(0))
        if (
            not isinstance(padding_mask, Tensor)
            or padding_mask.shape != expected_mask_shape
        ):
            raise ValueError(
                "MixerAttention key padding mask must match batch and sequence dimensions."
            )
        if padding_mask.device != tensor.device:
            raise ValueError(
                "MixerAttention key padding mask must match the input device."
            )
        if padding_mask.dtype != torch.bool and (
            not padding_mask.is_floating_point()
            or not ((padding_mask == 0) | padding_mask.isneginf()).all()
        ):
            raise ValueError(
                "MixerAttention key padding mask must be Boolean or additive 0/-inf."
            )

    @staticmethod
    def validate_mixing_state(
        state: object,
        flattened_input: "Tensor",
    ) -> None:
        if not isinstance(state, LayerState):
            raise RuntimeError(
                "MixerAttention mixing model must return a LayerState, got "
                f"{type(state).__name__}."
            )
        hidden = state.hidden
        if not torch.is_tensor(hidden):
            raise RuntimeError(
                "MixerAttention mixing model LayerState.hidden must be a Tensor."
            )
        if hidden.shape != flattened_input.shape:
            raise RuntimeError(
                "MixerAttention mixing model must preserve flattened shape "
                f"{tuple(flattened_input.shape)}, got {tuple(hidden.shape)}."
            )
        if hidden.dtype != flattened_input.dtype:
            raise RuntimeError(
                "MixerAttention mixing model must preserve dtype "
                f"{flattened_input.dtype}, got {hidden.dtype}."
            )
        if hidden.device != flattened_input.device:
            raise RuntimeError(
                "MixerAttention mixing model must preserve device "
                f"{flattened_input.device}, got {hidden.device}."
            )
