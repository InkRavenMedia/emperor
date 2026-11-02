"""Shared source-memory options resolved by each attention model package."""

from copy import deepcopy
from dataclasses import dataclass, fields, replace
from types import ModuleType
from typing import Any

from emperor.attention import AttentionSourceMemoryConfig
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions,
    AttentionGroupingConfig,
    GroupingConfig,
    MeanStdGroupingConfig,
    SumGroupingConfig,
    SummaryNormalizationOptions,
)
from emperor.config import ConfigBase


@dataclass(frozen=True, kw_only=True)
class AttentionSourceMemoryOptions:
    source_memory_config: AttentionSourceMemoryConfig | None = None
    source_memory_model_config: ConfigBase | None = None
    source_memory_chunk_size: int | None = None
    source_memory_grouping_method: type[GroupingConfig] | None = None
    source_memory_grouping_model_config: ConfigBase | None = None
    source_memory_summary_normalization: SummaryNormalizationOptions | None = None
    source_memory_rms_norm_epsilon: float | None = None


SOURCE_MEMORY_FIELDS = tuple(
    field.name for field in fields(AttentionSourceMemoryOptions)
)


def source_memory_defaults(
    config_module: ModuleType, *, prefix: str = "ATTN"
) -> dict[str, Any]:
    return {
        name: getattr(config_module, f"{prefix}_{name.upper()}")
        for name in SOURCE_MEMORY_FIELDS
    }


def build_source_memory_config(
    options: Any,
) -> AttentionSourceMemoryConfig | None:
    # Each package's attention options own the existing enable flag.
    if not options.source_memory_flag:
        return None
    memory_config = _copy_memory_config(options.source_memory_config)
    grouping_config = _resolve_grouping_config(
        memory_config.grouping_config, options.source_memory_grouping_method
    )
    grouping_config = _apply_grouping_overrides(grouping_config, options)
    generator_config = options.source_memory_model_config
    if generator_config is None:
        generator_config = memory_config.model_config
    return replace(
        memory_config,
        grouping_config=grouping_config,
        model_config=generator_config,
    )


def _copy_memory_config(
    config: AttentionSourceMemoryConfig | None,
) -> AttentionSourceMemoryConfig:
    if config is not None and not isinstance(config, AttentionSourceMemoryConfig):
        raise TypeError(
            "source_memory_config must be an AttentionSourceMemoryConfig instance."
        )
    return deepcopy(config) if config is not None else AttentionSourceMemoryConfig()


def _resolve_grouping_config(
    grouping_config: GroupingConfig | None,
    grouping_method: type[GroupingConfig] | None,
) -> GroupingConfig:
    if grouping_method is not None:
        if (
            not isinstance(grouping_method, type)
            or not issubclass(grouping_method, GroupingConfig)
            or grouping_method is GroupingConfig
        ):
            raise TypeError(
                "source_memory_grouping_method must be a concrete GroupingConfig class."
            )
        if grouping_config is None or type(grouping_config) is not grouping_method:
            grouping_config = grouping_method()
    return SumGroupingConfig() if grouping_config is None else grouping_config


def _apply_grouping_overrides(
    grouping_config: GroupingConfig, options: AttentionSourceMemoryOptions
) -> GroupingConfig:
    grouping_overrides: dict[str, Any] = {}
    for option_name, field_name in (
        ("source_memory_chunk_size", "chunk_size"),
        ("source_memory_summary_normalization", "summary_normalization"),
        ("source_memory_rms_norm_epsilon", "rms_norm_epsilon"),
    ):
        override_value = getattr(options, option_name)
        if override_value is not None:
            grouping_overrides[field_name] = override_value
    grouping_model_config = options.source_memory_grouping_model_config
    if grouping_model_config is not None:
        if not isinstance(
            grouping_config, (AttentionGroupingConfig, MeanStdGroupingConfig)
        ):
            raise ValueError(
                "source_memory_grouping_model_config requires attention or mean/std grouping."
            )
        grouping_overrides["model_config"] = grouping_model_config
    resolved_grouping = replace(grouping_config, **grouping_overrides)
    if resolved_grouping.chunk_size is None:
        raise ValueError(
            "Attention source memory requires chunk_size; set "
            "--attn-source-memory-chunk-size or source_memory_config.grouping_config.chunk_size."
        )
    if resolved_grouping.scope is None:
        resolved_grouping = replace(
            resolved_grouping, scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE
        )
    return resolved_grouping
