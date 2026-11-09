"""Resolve native and structured models for learned grouping methods."""

from collections.abc import Mapping
from dataclasses import fields, replace
from enum import Enum, auto
from types import ModuleType
from typing import Any

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
    AttentionGroupingConfig,
    GroupingConfig,
    MeanStdGroupingConfig,
    SumGroupingConfig,
)
from emperor.config import ConfigBase
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from models._projection_model import projection_layer_config


class GroupingModelOptions(Enum):
    LINEAR = auto()


def declare_grouping_model_defaults(namespace: dict[str, Any]) -> None:
    annotations = namespace.setdefault("__annotations__", {})
    aliases = namespace.setdefault("_CONFIG_FIELD_METADATA_ALIASES", {})
    skipped = namespace.get("CONFIG_SCHEMA_SKIP_KEYS", ())
    for owner in tuple(namespace):
        if not owner.endswith("GROUPING_MODEL_CONFIG"):
            continue
        prefix = owner.removesuffix("_CONFIG")
        for suffix, annotation in (
            ("OPTION", GroupingModelOptions | None),
            ("BIAS_FLAG", bool | None),
        ):
            key = f"{prefix}_{suffix}"
            namespace[key] = None
            annotations[key] = annotation
            aliases[key] = owner
            if owner in skipped:
                skipped.add(key)


_LEARNED_GROUPING_TYPES = (AttentionGroupingConfig, MeanStdGroupingConfig)


def prefer_scoped_grouping_model(
    updates: dict[str, Any], *, prefix: str = "grouping", has_config_model: bool = False
) -> None:
    if updates.get(f"{prefix}_model_config") is not None or has_config_model:
        updates.setdefault(f"{prefix}_model_option", None)
        updates.setdefault(f"{prefix}_model_bias_flag", None)


def resolve_grouping_model(
    grouping_type: type[GroupingConfig],
    model_config: ConfigBase | None,
    option: GroupingModelOptions | None,
    bias_flag: bool | None,
    hidden_dim: int | None = None,
    *,
    field_prefix: str = "grouping",
) -> ConfigBase | None:
    if hidden_dim is not None and not issubclass(
        grouping_type, AttentionGroupingConfig
    ):
        raise ValueError("grouping_attention_hidden_dim requires attention grouping.")
    if not issubclass(grouping_type, _LEARNED_GROUPING_TYPES):
        for suffix, value in (
            ("config", model_config),
            ("option", option),
            ("bias_flag", bias_flag),
        ):
            if value is not None:
                raise ValueError(
                    f"{field_prefix}_model_{suffix} requires attention or mean/std grouping."
                )
    if option is not None or (model_config is None and bias_flag is not None):
        if option is not None and option is not GroupingModelOptions.LINEAR:
            raise ValueError("The grouping model option must be LINEAR.")
        layer = projection_layer_config(
            LinearLayerConfig(bias_flag=bias_flag is not False)
        )
        if hidden_dim is None:
            return layer
        return LayerStackConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            apply_output_postprocessing_flag=False,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            layer_config=replace(layer, activation=ActivationOptions.TANH),
        )
    if bias_flag is not None:
        layer = (
            model_config.layer_config
            if isinstance(model_config, LayerStackConfig)
            else model_config
        )
        if (
            not isinstance(layer, LayerConfig)
            or type(layer.layer_model_config) is not LinearLayerConfig
        ):
            raise ValueError(
                "The grouping model bias flag requires a LINEAR grouping model."
            )
        layer = replace(
            layer,
            layer_model_config=replace(layer.layer_model_config, bias_flag=bias_flag),
        )
        model_config = (
            replace(model_config, layer_config=layer)
            if isinstance(model_config, LayerStackConfig)
            else layer
        )
    if hidden_dim is not None:
        if not isinstance(model_config, ConfigBase):
            raise ValueError(
                "grouping_attention_hidden_dim requires a supplied grouping_model_config "
                "or grouping_model_option."
            )
        if "hidden_dim" not in {field.name for field in fields(model_config)}:
            raise ValueError(
                "grouping_attention_hidden_dim requires a model config with a hidden_dim field."
            )
        model_config = replace(model_config, hidden_dim=hidden_dim)
    return model_config


GROUPING_FIELDS = (
    "grouping_scope",
    "group_count",
    "chunk_size",
    "grouping_sequence_length",
    "grouping_input_order",
    "grouping_method",
    "grouping_model_config",
    "grouping_model_option",
    "grouping_model_bias_flag",
    "grouping_summary_normalization",
    "grouping_attention_hidden_dim",
    "grouping_rms_norm_epsilon",
)


def grouping_from_overrides(
    values: Mapping[str, Any], config_module: ModuleType, prefix: str = ""
) -> GroupingConfig | None:
    scoped = {
        name: values[prefix + name]
        for name in GROUPING_FIELDS
        if prefix + name in values
    }
    if prefix:
        prefer_scoped_grouping_model(scoped)
    resolved = {
        name: scoped.get(
            name,
            values[name]
            if prefix in ("attn_", "ff_") and name in values
            else getattr(config_module, (prefix + name).upper(), None),
        )
        for name in GROUPING_FIELDS
    }
    return grouping_from_fields(resolved, prefix=prefix or "main")


def grouping_from_runtime(runtime: Any, prefix: str = "") -> GroupingConfig | None:
    resolved = {name: getattr(runtime, name) for name in GROUPING_FIELDS}
    if prefix:
        scoped = {
            name: value
            for name in GROUPING_FIELDS
            if (value := getattr(runtime, prefix + name)) is not None
        }
        prefer_scoped_grouping_model(scoped)
        resolved.update(scoped)
    return grouping_from_fields(resolved, prefix=prefix or "main")


def grouping_from_fields(
    values: dict[str, Any], *, prefix: str
) -> GroupingConfig | None:
    if all(values.get(field) is None for field in GROUPING_FIELDS):
        return None
    method = values.get("grouping_method")
    config_type = SumGroupingConfig if method is None else method
    if not isinstance(config_type, type) or not issubclass(config_type, GroupingConfig):
        raise ValueError(
            "grouping_method must be a concrete grouping configuration class."
        )
    model_config = resolve_grouping_model(
        config_type,
        values.get("grouping_model_config"),
        values.get("grouping_model_option"),
        values.get("grouping_model_bias_flag"),
        values.get("grouping_attention_hidden_dim"),
    )
    grouping_values = {
        "summary_normalization": values.get("grouping_summary_normalization"),
        "rms_norm_epsilon": values.get("grouping_rms_norm_epsilon"),
    }
    if model_config is not None:
        grouping_values["model_config"] = model_config
    grouping = config_type(
        **grouping_values,
        scope=values.get("grouping_scope"),
        group_count=values.get("group_count"),
        chunk_size=values.get("chunk_size"),
        sequence_length=values.get("grouping_sequence_length"),
        input_order=values.get("grouping_input_order"),
    )
    validator = AdaptiveParameterAugmentationConfig().registry_owner().VALIDATOR
    try:
        validator.validate_grouping_value(grouping)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{prefix} grouping requires complete scope-specific settings; "
            "set all grouping fields to None to disable it. " + str(exc)
        ) from exc
    return grouping
