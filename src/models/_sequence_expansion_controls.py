"""Flat Runtime Defaults for native Sequence Expansion components."""

from dataclasses import dataclass, fields, replace
from enum import Enum, auto
from typing import Any

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdditiveDynamicBiasConfig,
    WeightDecayScheduleOptions,
)
from emperor.config import ConfigBase
from emperor.embedding.absolute import (
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.linears import LinearLayerConfig
from models._projection_model import (
    ProjectionModelOptions,
    projection_stack_config,
    resolve_projection_model,
)


class SequenceExpansionModelOptions(Enum):
    LINEAR = auto()
    LINEAR_STACK = auto()
    ADAPTIVE_LINEAR = auto()
    ADAPTIVE_LINEAR_STACK = auto()


class SequenceExpansionPositionalEmbeddingOptions(Enum):
    LEARNED = auto()
    SINUSOIDAL = auto()
    DISABLED = auto()


@dataclass(frozen=True, kw_only=True)
class SequenceExpansionFlatOptions:
    sequence_expansion_model_option: SequenceExpansionModelOptions | None = None
    sequence_expansion_model_bias_flag: bool | None = None
    sequence_expansion_model_stack_num_layers: int | None = None
    sequence_expansion_model_stack_hidden_dim: int | None = None
    sequence_expansion_model_adaptive_bias_generator_num_layers: int | None = None
    sequence_expansion_model_adaptive_bias_generator_hidden_dim: int | None = None
    sequence_expansion_positional_embedding_option: (
        SequenceExpansionPositionalEmbeddingOptions | None
    ) = None
    sequence_expansion_positional_embedding_num_embeddings: int | None = None
    sequence_expansion_positional_embedding_padding_idx: int | None = None
    sequence_expansion_positional_embedding_auto_expand_flag: bool | None = None


_MODEL = SequenceExpansionModelOptions
_STACKED = {_MODEL.LINEAR_STACK, _MODEL.ADAPTIVE_LINEAR_STACK}
_ADAPTIVE = {_MODEL.ADAPTIVE_LINEAR, _MODEL.ADAPTIVE_LINEAR_STACK}
_POSITION = SequenceExpansionPositionalEmbeddingOptions
_POSITIONS = {
    _POSITION.LEARNED: TextLearnedPositionalEmbeddingConfig,
    _POSITION.SINUSOIDAL: TextSinusoidalPositionalEmbeddingConfig,
}
_POSITION_PREFIX = "sequence_expansion_positional_embedding_"
_STACK_PREFIX = "sequence_expansion_model_stack_"
_ADAPTIVE_PREFIX = "sequence_expansion_model_adaptive_bias_generator_"
_FLAT_FIELDS = fields(SequenceExpansionFlatOptions)


def declare_sequence_expansion_defaults(namespace: dict[str, Any]) -> None:
    """Add common and scoped flags beside the corresponding structured fields."""
    annotations = namespace.setdefault("__annotations__", {})
    aliases = namespace.setdefault("_CONFIG_FIELD_METADATA_ALIASES", {})
    for owner in tuple(namespace):
        if not owner.endswith("SEQUENCE_EXPANSION_CONFIG"):
            continue
        prefix = owner.removesuffix("SEQUENCE_EXPANSION_CONFIG")
        for field in _FLAT_FIELDS:
            key = prefix + field.name.upper()
            namespace.setdefault(key, None)
            annotations[key] = field.type
            aliases[key] = (
                f"{prefix}SEQUENCE_EXPANSION_POSITIONAL_EMBEDDING_CONFIG"
                if field.name.startswith(_POSITION_PREFIX)
                else f"{prefix}SEQUENCE_EXPANSION_MODEL_CONFIG"
            )


def prefer_scoped_sequence_controls(
    updates: dict[str, object], *, has_model_config: bool, has_position_config: bool
) -> None:
    """Let a scoped object or variant change clear incompatible inherited flags."""
    model_option = updates.get("sequence_expansion_model_option")
    position_option = updates.get(f"{_POSITION_PREFIX}option")
    for field in _FLAT_FIELDS:
        name = field.name
        if name.startswith(_POSITION_PREFIX):
            if has_position_config or (
                position_option is _POSITION.LEARNED
                and name == f"{_POSITION_PREFIX}auto_expand_flag"
            ):
                updates.setdefault(name, None)
        elif (
            has_model_config
            or (
                name.startswith(_STACK_PREFIX)
                and model_option is not None
                and model_option not in _STACKED
            )
            or (
                name.startswith(_ADAPTIVE_PREFIX)
                and model_option is not None
                and model_option not in _ADAPTIVE
            )
            or (
                name == "sequence_expansion_model_bias_flag"
                and model_option in _ADAPTIVE
            )
        ):
            updates.setdefault(name, None)


def _positive(value: int | None, default: int, label: str) -> int:
    if value is None:
        return default
    if type(value) is not int or value < 1:
        raise ValueError(f"Sequence expansion {label} must be a positive integer.")
    return value


def resolve_sequence_expansion_model(
    config: ConfigBase | None, options: SequenceExpansionFlatOptions, embedding_dim: int
) -> ConfigBase | None:
    option = options.sequence_expansion_model_option
    bias_flag = options.sequence_expansion_model_bias_flag
    stack_hidden_dim = options.sequence_expansion_model_stack_hidden_dim
    stack_num_layers = options.sequence_expansion_model_stack_num_layers
    generator_hidden_dim = (
        options.sequence_expansion_model_adaptive_bias_generator_hidden_dim
    )
    generator_num_layers = (
        options.sequence_expansion_model_adaptive_bias_generator_num_layers
    )
    if option not in _STACKED and (
        stack_hidden_dim is not None or stack_num_layers is not None
    ):
        raise ValueError(
            "Sequence expansion model stack settings require a stack option."
        )
    if option not in _ADAPTIVE and (
        generator_hidden_dim is not None or generator_num_layers is not None
    ):
        raise ValueError(
            "Sequence expansion bias generator settings require an adaptive model option."
        )
    if option in (None, _MODEL.LINEAR):
        return resolve_projection_model(
            config, None if option is None else ProjectionModelOptions.LINEAR, bias_flag
        )

    predictor = LinearLayerConfig(bias_flag=True if bias_flag is None else bias_flag)
    if option in _ADAPTIVE:
        if bias_flag is False:
            raise ValueError("Adaptive sequence expansion models require model bias.")
        generator = projection_stack_config(
            LinearLayerConfig(bias_flag=True),
            hidden_dim=_positive(
                generator_hidden_dim, embedding_dim, "bias generator hidden dimension"
            ),
            num_layers=_positive(generator_num_layers, 1, "bias generator layer count"),
        )
        predictor = AdaptiveLinearLayerConfig(
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                bias_config=AdditiveDynamicBiasConfig(
                    decay_schedule=WeightDecayScheduleOptions.DISABLED,
                    decay_rate=0.0,
                    decay_warmup_batches=0,
                    model_config=generator,
                ),
            ),
        )
    if option in _STACKED:
        return projection_stack_config(
            predictor,
            hidden_dim=_positive(
                stack_hidden_dim, embedding_dim, "model stack hidden dimension"
            ),
            num_layers=_positive(stack_num_layers, 2, "model stack layer count"),
        )
    return predictor


_PositionConfig = (
    TextLearnedPositionalEmbeddingConfig | TextSinusoidalPositionalEmbeddingConfig
)


def resolve_sequence_expansion_positions(
    config: _PositionConfig | None, options: SequenceExpansionFlatOptions
) -> _PositionConfig | None:
    option = options.sequence_expansion_positional_embedding_option
    capacity = options.sequence_expansion_positional_embedding_num_embeddings
    padding_idx = options.sequence_expansion_positional_embedding_padding_idx
    auto_expand = options.sequence_expansion_positional_embedding_auto_expand_flag
    if option is _POSITION.DISABLED:
        return None
    if option is None and all(
        value is None for value in (capacity, padding_idx, auto_expand)
    ):
        return config

    if option is not None:
        config = _POSITIONS[option]()
    if config is None:
        raise ValueError(
            "Select a sequence expansion positional embedding option first."
        )
    capacity = config.num_embeddings if capacity is None else capacity
    if type(capacity) is not int or capacity < 1:
        raise ValueError(
            "Sequence expansion positional embedding capacity must be a positive integer."
        )
    padding_idx = config.padding_idx if padding_idx is None else padding_idx
    if padding_idx is not None and not 0 <= padding_idx < capacity:
        raise ValueError(
            "Sequence expansion positional embedding padding index must be within capacity."
        )
    config = replace(config, num_embeddings=capacity, padding_idx=padding_idx)
    if isinstance(config, TextSinusoidalPositionalEmbeddingConfig):
        if auto_expand is None:
            auto_expand = bool(config.auto_expand_flag)
        return replace(config, auto_expand_flag=auto_expand)
    if auto_expand is not None:
        raise ValueError(
            "Sequence expansion positional auto-expand requires SINUSOIDAL positions."
        )
    return config
