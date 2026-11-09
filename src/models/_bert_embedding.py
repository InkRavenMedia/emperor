"""Standard BERT byte embedding pipelines composed from scalar Runtime Defaults."""

from types import ModuleType
from typing import Any

import torch

from emperor.attention import SelfAttentionConfig, SelfAttentionProjectionStrategy
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdditiveDynamicBiasConfig,
    WeightDecayScheduleOptions,
)
from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
from emperor.embedding.contextual import (
    ByteContextualEmbeddingConfig,
    CausalPrefixKernelConfig,
)
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
from emperor.embedding.relative import DynamicPositionalBiasConfig
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    RoutingInitializationMode,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, SamplerConfig
from emperor.transformer import (
    FeedForwardConfig,
    TransformerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from model_runtime.packages.runtime_values import positive_runtime_fields

_HIERARCHICAL_DEFAULTS = {
    "hierarchical_embedding_flag": False,
    "hierarchical_byte_embedding_dim": 32,
    "hierarchical_byte_encoder_num_layers": 1,
    "hierarchical_byte_feed_forward_dim": 64,
    "hierarchical_byte_num_heads": 4,
    "hierarchical_max_token_bytes": 64,
}
_CONTEXTUAL_DEFAULTS = {
    "contextual_embedding_flag": False,
    "contextual_embedding_max_token_bytes": 4,
    "contextual_embedding_kernel_dim": 32,
    "contextual_embedding_residual_scale_initial_value": 1e-3,
}


def declare_runtime_defaults(namespace: dict[str, Any]) -> None:
    annotations = namespace.setdefault("__annotations__", {})
    aliases = namespace.setdefault("_CONFIG_FIELD_METADATA_ALIASES", {})
    hidden_keys = namespace.get("CONFIG_SCHEMA_SKIP_KEYS", set())
    positive_keys = []
    for owner, defaults in (
        ("HIERARCHICAL_EMBEDDING_CONFIG", _HIERARCHICAL_DEFAULTS),
        ("CONTEXTUAL_EMBEDDING_CONFIG", _CONTEXTUAL_DEFAULTS),
    ):
        for name, default in defaults.items():
            key = name.upper()
            namespace.setdefault(key, default)
            annotations.setdefault(key, type(default))
            aliases[key] = owner
            hidden_keys.discard(key)
            if type(default) is int:
                positive_keys.append(key)
    # Both byte pipelines require an untied MLM decoder.
    hidden_keys.discard("MLM_DECODER_WEIGHT_TYING_FLAG")
    previous = namespace["RUNTIME_VALUE_CONSTRAINTS"].positive
    namespace["RUNTIME_VALUE_CONSTRAINTS"] = positive_runtime_fields(
        *previous, *positive_keys
    )


def resolve_runtime_defaults(
    flat_kwargs: dict[str, Any], config_module: ModuleType
) -> dict[str, Any]:
    kwargs = dict(flat_kwargs)
    hierarchical = _pop_native_settings(kwargs, config_module, _HIERARCHICAL_DEFAULTS)
    contextual = _pop_native_settings(kwargs, config_module, _CONTEXTUAL_DEFAULTS)
    encoder_options = kwargs.get("encoder_options")
    hidden_dim = kwargs.get(
        "hidden_dim", getattr(encoder_options, "hidden_dim", config_module.HIDDEN_DIM)
    )
    sequence_length = kwargs.get("sequence_length", config_module.SEQUENCE_LENGTH)
    if hierarchical["hierarchical_embedding_flag"] and _uses_native_config(
        kwargs, config_module, "hierarchical_embedding_config"
    ):
        kwargs["hierarchical_embedding_config"] = build_hierarchical_embedding_config(
            batch_size=kwargs.get("batch_size", config_module.BATCH_SIZE),
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            byte_embedding_dim=hierarchical["hierarchical_byte_embedding_dim"],
            byte_encoder_num_layers=hierarchical[
                "hierarchical_byte_encoder_num_layers"
            ],
            byte_feed_forward_dim=hierarchical["hierarchical_byte_feed_forward_dim"],
            byte_num_heads=hierarchical["hierarchical_byte_num_heads"],
            max_token_bytes=hierarchical["hierarchical_max_token_bytes"],
        )
    if contextual["contextual_embedding_flag"] and _uses_native_config(
        kwargs, config_module, "contextual_embedding_config"
    ):
        kwargs["contextual_embedding_config"] = _build_contextual_embedding_config(
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            max_token_bytes=contextual["contextual_embedding_max_token_bytes"],
            kernel_dim=contextual["contextual_embedding_kernel_dim"],
            residual_scale=contextual[
                "contextual_embedding_residual_scale_initial_value"
            ],
        )
    return kwargs


def _pop_native_settings(
    kwargs: dict[str, Any], config_module: ModuleType, defaults: dict[str, Any]
) -> dict[str, Any]:
    return {
        key: kwargs.pop(key, getattr(config_module, key.upper())) for key in defaults
    }


def _uses_native_config(
    kwargs: dict[str, Any], config_module: ModuleType, legacy_key: str
) -> bool:
    default_config = getattr(config_module, legacy_key.upper())
    configured = getattr(kwargs.get("embedding_options"), legacy_key, default_config)
    return kwargs.get(legacy_key, configured) is None


def _linear_stack(
    input_dim: int | None = None,
    output_dim: int | None = None,
    *,
    hidden_dim: int = 32,
    num_layers: int = 1,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.RELU,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def build_hierarchical_embedding_config(
    *,
    batch_size: int,
    sequence_length: int,
    hidden_dim: int,
    byte_embedding_dim: int,
    byte_encoder_num_layers: int,
    byte_feed_forward_dim: int,
    byte_num_heads: int,
    max_token_bytes: int,
) -> HierarchicalByteEmbeddingConfig:
    attention_config = SelfAttentionConfig(
        batch_size=batch_size * sequence_length,
        num_heads=byte_num_heads,
        embedding_dim=byte_embedding_dim,
        query_key_projection_dim=0,
        value_projection_dim=0,
        target_sequence_length=max_token_bytes + 1,
        source_sequence_length=max_token_bytes + 1,
        target_dtype=torch.float32,
        dropout_probability=0.0,
        zero_attention_flag=False,
        causal_attention_mask_flag=False,
        add_key_value_bias_flag=False,
        source_memory_flag=False,
        average_attention_weights_flag=False,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=_linear_stack(byte_embedding_dim, byte_embedding_dim),
        projection_strategy=SelfAttentionProjectionStrategy.FUSED,
    )
    encoder_layer = TransformerEncoderLayerConfig(
        embedding_dim=byte_embedding_dim,
        layer_norm_position=LayerNormPositionOptions.BEFORE,
        dropout_probability=0.0,
        residual_config=AdditiveResidualConfig(),
        attention_config=attention_config,
        feed_forward_config=FeedForwardConfig(
            input_dim=byte_embedding_dim,
            output_dim=byte_embedding_dim,
            stack_config=_linear_stack(
                byte_embedding_dim,
                byte_embedding_dim,
                hidden_dim=byte_feed_forward_dim,
            ),
        ),
    )
    return HierarchicalByteEmbeddingConfig(
        byte_embedding_dim=byte_embedding_dim,
        output_dim=hidden_dim,
        max_token_bytes=max_token_bytes,
        byte_position_config=TextLearnedPositionalEmbeddingConfig(
            num_embeddings=max_token_bytes + 1,
            embedding_dim=byte_embedding_dim,
        ),
        encoder_config=TransformerConfig(
            encoder_stack_config=LayerStackConfig(
                input_dim=byte_embedding_dim,
                hidden_dim=byte_embedding_dim,
                output_dim=byte_embedding_dim,
                num_layers=byte_encoder_num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_postprocessing_flag=True,
                layer_config=TransformerEncoderBlockLayerConfig(
                    activation=ActivationOptions.DISABLED,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    layer_model_config=encoder_layer,
                ),
            ),
        ),
        projection_config=_linear_stack(byte_embedding_dim, hidden_dim),
    )


def _build_contextual_embedding_config(
    *,
    sequence_length: int,
    hidden_dim: int,
    max_token_bytes: int,
    kernel_dim: int,
    residual_scale: float,
) -> ByteContextualEmbeddingConfig:
    return ByteContextualEmbeddingConfig(
        max_token_bytes=max_token_bytes,
        hidden_dim=hidden_dim,
        byte_moe_config=_contextual_moe_config(max_token_bytes * 9, hidden_dim),
        positional_embedding_config=TextLearnedPositionalEmbeddingConfig(
            num_embeddings=sequence_length,
            embedding_dim=hidden_dim,
            padding_idx=0,
        ),
        prefix_kernel_config=CausalPrefixKernelConfig(
            hidden_dim=hidden_dim,
            kernel_dim=kernel_dim,
            relative_position_config=DynamicPositionalBiasConfig(
                num_heads=1,
                embedding_dim=kernel_dim,
                max_positions=max(1, sequence_length - 1),
            ),
        ),
        context_moe_config=_contextual_moe_config(hidden_dim * 2, hidden_dim),
        residual_scale_initial_value=residual_scale,
    )


def _contextual_moe_config(input_dim: int, hidden_dim: int) -> MixtureOfExpertsConfig:
    # Each contextual MoE owns a token-local router and two-layer adaptive experts.
    num_experts, top_k = 3, 2
    experts = _linear_stack(hidden_dim=hidden_dim, num_layers=2)
    experts.layer_config.layer_model_config = AdaptiveLinearLayerConfig(
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            bias_config=AdditiveDynamicBiasConfig(
                decay_schedule=WeightDecayScheduleOptions.DISABLED,
                decay_rate=0.0,
                decay_warmup_batches=0,
                model_config=_linear_stack(hidden_dim=hidden_dim),
            ),
        ),
    )
    return MixtureOfExpertsConfig(
        input_dim=input_dim,
        output_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        capacity_factor=0.0,
        dropped_token_behavior=DroppedTokenOptions.ZEROS,
        compute_expert_mixture_flag=True,
        weighted_parameters_flag=True,
        weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
        routing_initialization_mode=RoutingInitializationMode.LAYER,
        sampler_config=SamplerConfig(
            top_k=top_k,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=True,
            noisy_topk_flag=False,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
            router_config=RouterConfig(
                input_dim=input_dim,
                num_experts=num_experts,
                noisy_topk_flag=False,
                model_config=_linear_stack(
                    input_dim, num_experts, hidden_dim=max(input_dim, num_experts)
                ),
            ),
        ),
        expert_model_config=experts,
    )
