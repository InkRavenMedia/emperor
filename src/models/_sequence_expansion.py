"""Shared Sequence Expansion options for transformer model packages."""

from copy import deepcopy
from dataclasses import dataclass, fields, replace
from types import ModuleType

from emperor.attention import MixerAttentionConfig
from emperor.config import ConfigBase
from emperor.embedding.absolute import (
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.layers import (
    HierarchicalReasoningModelRecurrentConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
)
from emperor.transformer import (
    SequenceExpansionConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerEncoderBlockLayerConfig,
)
from emperor.transformer.sequence_expansion import SequenceExpansion
from models._sequence_expansion_controls import (
    SequenceExpansionFlatOptions,
    prefer_scoped_sequence_controls,
    resolve_sequence_expansion_model,
    resolve_sequence_expansion_positions,
)


@dataclass(frozen=True, kw_only=True)
class SequenceExpansionOptions(SequenceExpansionFlatOptions):
    sequence_expansion_flag: bool = False
    sequence_expansion_config: SequenceExpansionConfig | None = None
    sequence_expansion_max_new_tokens: int | None = None
    sequence_expansion_model_config: ConfigBase | None = None
    sequence_expansion_positional_embedding_config: (
        TextLearnedPositionalEmbeddingConfig
        | TextSinusoidalPositionalEmbeddingConfig
        | None
    ) = None


SEQUENCE_EXPANSION_FIELDS = tuple(
    field.name for field in fields(SequenceExpansionOptions)
)


def sequence_expansion_defaults(
    config_module: ModuleType, prefix: str = ""
) -> dict[str, object]:
    defaults = {}
    for name in SEQUENCE_EXPANSION_FIELDS:
        value = getattr(config_module, f"{prefix}{name.upper()}")
        if prefix and value is None:
            value = getattr(config_module, name.upper())
        defaults[name] = value
    return defaults


def scoped_sequence_expansion_updates(updates: dict[str, object]) -> dict[str, object]:
    """Inherit omitted path settings; prefer fields in an explicit path config."""
    resolved = {
        name: value
        for name, value in updates.items()
        if name not in SEQUENCE_EXPANSION_FIELDS or value is not None
    }
    config = resolved.get("sequence_expansion_config")
    has_scoped_model = resolved.get("sequence_expansion_model_config") is not None
    has_scoped_positions = (
        resolved.get("sequence_expansion_positional_embedding_config") is not None
    )
    if isinstance(config, SequenceExpansionConfig):
        for name in ("max_new_tokens", "model_config", "positional_embedding_config"):
            if getattr(config, name) is not None:
                resolved.setdefault(f"sequence_expansion_{name}", None)
        has_scoped_model |= config.model_config is not None
        has_scoped_positions |= config.positional_embedding_config is not None
    prefer_scoped_sequence_controls(
        resolved,
        has_model_config=has_scoped_model,
        has_position_config=has_scoped_positions,
    )
    return resolved


def build_sequence_expansion_config(
    options: SequenceExpansionOptions, embedding_dim: int
) -> SequenceExpansionConfig | None:
    if not options.sequence_expansion_flag:
        return None
    config = options.sequence_expansion_config
    if config is not None and not isinstance(config, SequenceExpansionConfig):
        raise TypeError("sequence_expansion_config must be SequenceExpansionConfig.")
    config = deepcopy(config) if config is not None else SequenceExpansionConfig()
    max_new_tokens = options.sequence_expansion_max_new_tokens
    model_config = options.sequence_expansion_model_config
    if model_config is None:
        model_config = config.model_config
    else:
        model_config = deepcopy(model_config)
    position_config = options.sequence_expansion_positional_embedding_config
    if position_config is None:
        position_config = config.positional_embedding_config
    else:
        position_config = deepcopy(position_config)
    resolved = replace(
        config,
        embedding_dim=embedding_dim,
        max_new_tokens=config.max_new_tokens
        if max_new_tokens is None
        else max_new_tokens,
        model_config=resolve_sequence_expansion_model(
            model_config, options, embedding_dim
        ),
        positional_embedding_config=resolve_sequence_expansion_positions(
            position_config, options
        ),
    )
    SequenceExpansion.VALIDATOR.validate_config(resolved)
    return resolved


def expanded_sequence_capacity(config: ConfigBase, input_length: int) -> int:
    growth, _ = _expansion_schedule(config)
    return input_length + growth


def with_expansion_attention_capacity(
    config: ConfigBase, input_length: int, source_capacity: int | None = None
) -> ConfigBase:
    """Size existing attention bounds for every configured expansion invocation."""
    growth, _ = _expansion_schedule(config)
    if growth == 0 and source_capacity is None:
        return config
    config = deepcopy(config)
    _, layers = _expansion_schedule(config)
    capacity = input_length + growth
    for layer in layers:
        self_attention = getattr(layer, "self_attention_config", None)
        if self_attention is None:
            self_attention = layer.attention_config
        if isinstance(self_attention, MixerAttentionConfig):
            self_attention.sequence_length = capacity
            self_attention.variable_sequence_length_flag = True
        else:
            self_attention.target_sequence_length = capacity
            self_attention.source_sequence_length = capacity
        cross_attention = getattr(layer, "cross_attention_config", None)
        if cross_attention is not None:
            cross_attention.target_sequence_length = capacity
            if source_capacity is not None:
                cross_attention.source_sequence_length = source_capacity
    return config


def _expansion_schedule(config: ConfigBase) -> tuple[int, list[ConfigBase]]:
    if isinstance(config, LayerStackConfig):
        growth, layers = _expansion_schedule(config.layer_config)
        return config.num_layers * growth, layers
    if isinstance(config, RecurrentLayerConfig):
        growth, layers = _expansion_schedule(config.block_config)
        return config.max_steps * growth, layers
    if isinstance(config, TinyRecursiveModelRecurrentConfig):
        growth, layers = _expansion_schedule(config.block_config)
        calls = config.answer_update_count * (
            config.latent_updates_per_answer_update + 1
        )
        return calls * growth, layers
    if isinstance(config, HierarchicalReasoningModelRecurrentConfig):
        high_growth, high_layers = _expansion_schedule(config.high_block_config)
        low_growth, low_layers = _expansion_schedule(config.low_block_config)
        growth = config.high_cycles * (high_growth + config.low_cycles * low_growth)
        return growth, high_layers + low_layers
    if isinstance(
        config, (TransformerEncoderBlockLayerConfig, TransformerDecoderBlockLayerConfig)
    ):
        expansion = config.sequence_expansion_config
        growth = 0 if expansion is None else expansion.max_new_tokens
        return growth, [config.layer_model_config]
    return 0, []
