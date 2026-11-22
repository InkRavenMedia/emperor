from collections.abc import Mapping
from dataclasses import replace
from types import ModuleType
from typing import Any

from models import _learned_grouping

# Weight option, grouping option, and the corresponding flat configuration prefix.
_WEIGHT_ROLES = (
    ("hidden_adaptive_weight_options", "grouping_config", ""),
    ("attention_hidden_adaptive_weight_options", "attention_grouping_config", "attn_"),
    (
        "feed_forward_hidden_adaptive_weight_options",
        "feed_forward_grouping_config",
        "ff_",
    ),
    ("router_adaptive_weight_options", "router_grouping_config", "router_"),
)
_PROJECTION_ROLES = (
    ("input_boundary_options", "input_layer_"),
    ("output_boundary_options", "output_layer_"),
)


def grouping_keys(config_module: ModuleType) -> set[str]:
    return {
        prefix + field
        for _, _, prefix in _WEIGHT_ROLES
        for field in _learned_grouping.GROUPING_FIELDS
        if hasattr(config_module, (prefix + field).upper())
    }


def apply_grouping_options(
    options: dict[str, Any],
    values: Mapping[str, object],
    config_module: ModuleType,
    *,
    include_declared_scopes: bool = False,
) -> dict[str, Any]:
    resolved = dict(options)
    for weight_key, grouping_key, prefix in _WEIGHT_ROLES:
        if weight_key in resolved or (
            include_declared_scopes
            and prefix
            and hasattr(config_module, prefix.upper() + "GROUPING_SCOPE")
        ):
            resolved[grouping_key] = _learned_grouping.grouping_from_overrides(
                values, config_module, prefix
            )
    for role, prefix in _PROJECTION_ROLES:
        projection = resolved.get(role)
        if projection is not None:
            grouping = _learned_grouping.grouping_from_overrides(
                values, config_module, prefix
            )
            resolved[role] = replace(projection, grouping_config=grouping)
    return resolved


def apply_declared_grouping_options(
    options: dict[str, Any], values: Mapping[str, object], config_module: ModuleType
) -> dict[str, Any]:
    return apply_grouping_options(
        options, values, config_module, include_declared_scopes=True
    )
