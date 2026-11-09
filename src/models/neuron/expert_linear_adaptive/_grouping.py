"""Package-owned scalar grouping resolution."""

from dataclasses import replace

from models import _learned_grouping


def grouping_keys(config_module):
    return {
        name.lower()
        for name in vars(config_module)
        if name.isupper()
        and any(
            name.lower().endswith(field) for field in _learned_grouping.GROUPING_FIELDS
        )
    }


def apply_grouping_options(options, values, config_module):
    result = dict(options)
    for role, prefix in (
        ("", ""),
        ("attention_", "attn_"),
        ("feed_forward_", "ff_"),
        ("router_", "router_"),
    ):
        weight_key = role + "hidden_adaptive_weight_options"
        if role == "router_":
            weight_key = "router_adaptive_weight_options"
        if weight_key in result:
            result[role + "grouping_config"] = (
                _learned_grouping.grouping_from_overrides(values, config_module, prefix)
            )
    for role, prefix in (
        ("input_boundary_options", "input_layer_"),
        ("output_boundary_options", "output_layer_"),
    ):
        if result.get(role) is not None:
            result[role] = replace(
                result[role],
                grouping_config=_learned_grouping.grouping_from_overrides(
                    values, config_module, prefix
                ),
            )
    return result
