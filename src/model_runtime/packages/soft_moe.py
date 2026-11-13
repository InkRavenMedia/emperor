"""Shared CLI defaults and binding for package-declared Soft MoE components."""

from collections.abc import Mapping, MutableMapping
from math import isfinite
from types import ModuleType
from typing import Any

from emperor.experts import SoftMoEConfig
from emperor.layers import (
    ActivationOptions,
    AttentionResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    ResidualConfig,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig

# Dimension and layer-model fields are supplied by the mixture owner. Routers
# must emit raw logits; noisy top-k and context controllers are not Soft MoE options.
_DEFAULTS: dict[str, tuple[Any, Any]] = {
    "flag": (bool, False),
    "num_experts": (int | None, None),
    "slots_per_expert": (int, 2),
    "scoring_hidden_dim": (int, 32),
    "scoring_num_layers": (int, 2),
    "scoring_activation": (ActivationOptions, ActivationOptions.RELU),
    "scoring_dropout_probability": (float, 0.0),
    "scoring_layer_norm_position": (
        LayerNormPositionOptions,
        LayerNormPositionOptions.DISABLED,
    ),
    "scoring_normalization": (NormalizationOptions, NormalizationOptions.RMS_NORM),
    "scoring_last_layer_bias_option": (
        LastLayerBiasOptions,
        LastLayerBiasOptions.DEFAULT,
    ),
    "scoring_residual_connection_option": (type[ResidualConfig] | None, None),
    "scoring_residual_model_flag": (bool, False),
}
_POSITIVE_FIELDS = (
    "num_experts",
    "slots_per_expert",
    "scoring_hidden_dim",
    "scoring_num_layers",
)


def _component_prefixes(namespace: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        key.removesuffix("_CONFIG").lower()
        for key in namespace
        if key == "SOFT_MOE_CONFIG" or key.endswith("_SOFT_MOE_CONFIG")
    )


def declare_soft_moe_runtime_defaults(namespace: MutableMapping[str, Any]) -> None:
    """Expose controls for each declared *_SOFT_MOE_CONFIG, including metadata.

    Call at the end of a package's config.py. Role-specific controls inherit the
    common SOFT_MOE_* values unless explicitly overridden; enable flags do not.
    """
    prefixes = _component_prefixes(namespace)
    if not prefixes:
        return
    annotations = namespace.setdefault("__annotations__", {})
    aliases = namespace.setdefault("_CONFIG_FIELD_METADATA_ALIASES", {})
    hidden_keys = namespace.get("CONFIG_SCHEMA_SKIP_KEYS", set())
    for prefix in dict.fromkeys(("soft_moe", *prefixes)):
        owner_key = f"{prefix.upper()}_CONFIG"
        if owner_key not in namespace:
            owner_key = f"{prefixes[0].upper()}_CONFIG"
        for suffix, (annotation, default) in _DEFAULTS.items():
            if suffix == "flag" and prefix not in prefixes:
                continue
            key = f"{prefix}_{suffix}".upper()
            if prefix != "soft_moe" and suffix != "flag":
                annotation = annotation | None
                default = None
            namespace.setdefault(key, default)
            annotations.setdefault(key, annotation)
            aliases[key] = owner_key
            hidden_keys.discard(key)


def validate_soft_moe_runtime_values(
    values: Mapping[str, Any], config_module: ModuleType
) -> None:
    """Reject invalid scalar controls even when the corresponding branch is off."""
    prefixes = _component_prefixes(vars(config_module))
    if not prefixes:
        return
    for prefix in dict.fromkeys(("soft_moe", *prefixes)):
        for suffix in _POSITIVE_FIELDS:
            key = f"{prefix}_{suffix}"
            value = values.get(key)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be positive; got {value!r}")
        dropout_key = f"{prefix}_scoring_dropout_probability"
        dropout = values.get(dropout_key)
        if dropout is not None and not 0 <= dropout <= 1:
            raise ValueError(f"{dropout_key} must be between 0 and 1")


def _setting(
    values: Mapping[str, Any], config_module: ModuleType, prefix: str, suffix: str
) -> Any:
    # Explicit role > explicit common > explicit legacy router stack > defaults.
    keys = tuple(dict.fromkeys((f"{prefix}_{suffix}", f"soft_moe_{suffix}")))
    for key in keys:
        if key in values and values[key] is not None:
            return values[key]
    if suffix.startswith("scoring_"):
        stack_field = suffix.removeprefix("scoring_")
        router_key = f"router_stack_{stack_field}"
        if values.get(router_key) is not None:
            return values[router_key]
        router_options = values.get("router_stack_options")
        if getattr(router_options, stack_field, None) is not None:
            return getattr(router_options, stack_field)
    for key in keys:
        default = getattr(config_module, key.upper(), None)
        if default is not None:
            return default
    return _DEFAULTS[suffix][1]


def _existing_config(
    values: Mapping[str, Any], config_module: ModuleType, prefix: str
) -> SoftMoEConfig | None:
    key = f"{prefix}_config"
    if key in values:
        return values[key]
    options_key = "feed_forward_options"
    if prefix == "attention_soft_moe":
        options_key = "attention_options"
    options = values.get(options_key)
    return getattr(options, key, getattr(config_module, key.upper(), None))


def _build_config(
    values: Mapping[str, Any], config_module: ModuleType, prefix: str
) -> SoftMoEConfig:
    settings = {
        suffix: _setting(values, config_module, prefix, suffix)
        for suffix in _DEFAULTS
        if suffix != "flag"
    }
    num_experts = settings["num_experts"]
    if num_experts is None:
        mixture_options = values.get("mixture_options")
        default_experts = getattr(config_module, "NUM_EXPERTS", 4)
        num_experts = values.get(
            "num_experts", getattr(mixture_options, "num_experts", default_experts)
        )
    residual_config = None
    residual_option = settings["scoring_residual_connection_option"]
    residual_model_flag = settings["scoring_residual_model_flag"]
    if residual_model_flag and (
        not isinstance(residual_option, type)
        or not issubclass(
            residual_option, (WeightedResidualConfig, WeightedBlendResidualConfig)
        )
    ):
        raise ValueError(
            f"{prefix.upper()}_SCORING_RESIDUAL_MODEL_FLAG=True requires "
            f"{prefix.upper()}_SCORING_RESIDUAL_CONNECTION_OPTION to select "
            "WeightedResidualConfig or WeightedBlendResidualConfig."
        )
    if residual_option is not None:
        if issubclass(residual_option, AttentionResidualConfig):
            raise ValueError(
                "Soft MoE scoring cannot use AttentionResidualConfig: it requires "
                "output postprocessing, but the router requires raw logits."
            )
        if issubclass(
            residual_option, (WeightedResidualConfig, WeightedBlendResidualConfig)
        ):
            residual_config = residual_option(
                coefficient_activation=values.get(
                    "residual_coefficient_activation",
                    getattr(
                        config_module,
                        "RESIDUAL_COEFFICIENT_ACTIVATION",
                        ActivationOptions.SIGMOID,
                    ),
                ),
                model_config=(
                    _build_coefficient_stack(values, config_module)
                    if residual_model_flag
                    else None
                ),
            )
        else:
            residual_config = residual_option()
    layer_config = LayerConfig(
        activation=settings["scoring_activation"],
        dropout_probability=settings["scoring_dropout_probability"],
        layer_norm_position=settings["scoring_layer_norm_position"],
        normalization=settings["scoring_normalization"],
        residual_config=residual_config,
        layer_model_config=None,
    )
    scoring_stack = LayerStackConfig(
        hidden_dim=settings["scoring_hidden_dim"],
        num_layers=settings["scoring_num_layers"],
        last_layer_bias_option=settings["scoring_last_layer_bias_option"],
        apply_output_postprocessing_flag=False,
        layer_config=layer_config,
    )
    router_config = RouterConfig(noisy_topk_flag=False, model_config=scoring_stack)
    return SoftMoEConfig(
        num_experts=num_experts,
        slots_per_expert=settings["slots_per_expert"],
        router_config=router_config,
    )


def _residual_stack_setting(
    values: Mapping[str, Any], config_module: ModuleType, field: str
) -> Any:
    """Apply residual-stack overrides to the shared submodule-stack settings."""
    independent = values.get(
        "residual_stack_independent_flag", config_module.RESIDUAL_STACK_INDEPENDENT_FLAG
    )
    residual_key = f"residual_stack_{field}"
    override = values.get(residual_key, getattr(config_module, residual_key.upper()))
    if (
        independent or field in {"residual_block_size", "residual_rms_norm_epsilon"}
    ) and override is not None:
        return override
    submodule_key = f"submodule_stack_{field}"
    if submodule_key in values:
        return values[submodule_key]
    submodule_options = values.get("submodule_stack_options")
    if submodule_options is not None:
        return getattr(submodule_options, field)
    return getattr(config_module, submodule_key.upper())


def _build_coefficient_stack(
    values: Mapping[str, Any], config_module: ModuleType
) -> LayerStackConfig:
    """Build the scoring residual's coefficient model from Residual Stack Options."""
    if values.get(
        "residual_stack_residual_model_flag",
        config_module.RESIDUAL_STACK_RESIDUAL_MODEL_FLAG,
    ):
        raise ValueError(
            "RESIDUAL_STACK_RESIDUAL_MODEL_FLAG=True would recursively define "
            "the residual coefficient stack in itself."
        )
    fields = (
        "hidden_dim",
        "num_layers",
        "last_layer_bias_option",
        "apply_output_postprocessing_flag",
        "activation",
        "layer_norm_position",
        "normalization",
        "residual_connection_option",
        "residual_block_size",
        "residual_rms_norm_epsilon",
        "dropout_probability",
        "bias_flag",
    )
    settings = {
        field: _residual_stack_setting(values, config_module, field) for field in fields
    }
    residual_config = None
    residual_option = settings["residual_connection_option"]
    if residual_option is not None:
        if issubclass(
            residual_option, (WeightedResidualConfig, WeightedBlendResidualConfig)
        ):
            residual_config = residual_option(
                coefficient_activation=values.get(
                    "residual_coefficient_activation",
                    config_module.RESIDUAL_COEFFICIENT_ACTIVATION,
                )
            )
        elif issubclass(residual_option, AttentionResidualConfig):
            residual_config = residual_option(
                block_size=settings["residual_block_size"],
                rms_norm_epsilon=settings["residual_rms_norm_epsilon"],
            )
        else:
            residual_config = residual_option()
    return LayerStackConfig(
        hidden_dim=settings["hidden_dim"],
        num_layers=settings["num_layers"],
        last_layer_bias_option=settings["last_layer_bias_option"],
        apply_output_postprocessing_flag=settings["apply_output_postprocessing_flag"],
        layer_config=LayerConfig(
            activation=settings["activation"],
            layer_norm_position=settings["layer_norm_position"],
            normalization=settings["normalization"],
            residual_config=residual_config,
            dropout_probability=settings["dropout_probability"],
            layer_model_config=LinearLayerConfig(bias_flag=settings["bias_flag"]),
        ),
    )


def resolve_soft_moe_runtime_defaults(
    values: Mapping[str, Any], config_module: ModuleType
) -> dict[str, Any]:
    """Bind scalar controls to the typed fields consumed by every model builder."""
    resolved = dict(values)
    prefixes = _component_prefixes(vars(config_module))
    if not prefixes:
        return resolved
    for prefix in prefixes:
        flag_key = f"{prefix}_flag"
        enabled = values.get(flag_key, getattr(config_module, flag_key.upper(), False))
        if enabled and _existing_config(values, config_module, prefix) is None:
            resolved[f"{prefix}_config"] = _build_config(values, config_module, prefix)
    for prefix in dict.fromkeys(("soft_moe", *prefixes)):
        for suffix in _DEFAULTS:
            resolved.pop(f"{prefix}_{suffix}", None)
    return resolved
