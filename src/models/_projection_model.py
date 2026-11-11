"""Flat CLI controls shared by source memory and sequence expansion."""

from dataclasses import replace
from enum import Enum, auto
from typing import Any

from emperor.config import ConfigBase
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig


class ProjectionModelOptions(Enum):
    LINEAR = auto()
    LINEAR_STACK = auto()
    RECURRENT_LINEAR = auto()


def declare_projection_model_defaults(namespace: dict[str, Any]) -> None:
    """Place model controls beside each component's existing model config."""
    annotations = namespace.setdefault("__annotations__", {})
    aliases = namespace.setdefault("_CONFIG_FIELD_METADATA_ALIASES", {})
    for owner in tuple(namespace):
        if not owner.endswith(
            (
                "ATTN_SOURCE_MEMORY_MODEL_CONFIG",
                "TOKEN_MIXER_SOURCE_MEMORY_MODEL_CONFIG",
            )
        ):
            continue
        prefix = owner.removesuffix("_CONFIG")
        for suffix, annotation in (
            ("OPTION", ProjectionModelOptions | None),
            ("BIAS_FLAG", bool | None),
            ("STACK_NUM_LAYERS", int | None),
            ("STACK_HIDDEN_DIM", int | None),
            ("RECURRENT_MAX_STEPS", int | None),
        ):
            key = f"{prefix}_{suffix}"
            namespace[key] = None
            annotations[key] = annotation
            aliases[key] = owner


def projection_layer_config(model: ConfigBase) -> LayerConfig:
    return LayerConfig(
        activation=ActivationOptions.DISABLED,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        layer_model_config=model,
    )


def projection_stack_config(
    model: ConfigBase, *, num_layers: int, hidden_dim: int
) -> LayerStackConfig:
    return LayerStackConfig(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        apply_output_postprocessing_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        layer_config=projection_layer_config(model),
    )


def resolve_projection_model(
    config: ConfigBase | None,
    option: ProjectionModelOptions | None,
    bias_flag: bool | None,
    *,
    stack_num_layers: int | None = None,
    stack_hidden_dim: int | None = None,
    recurrent_max_steps: int | None = None,
) -> ConfigBase | None:
    if option is ProjectionModelOptions.LINEAR_STACK:
        return projection_stack_config(
            LinearLayerConfig(bias_flag=True if bias_flag is None else bias_flag),
            num_layers=2 if stack_num_layers is None else stack_num_layers,
            hidden_dim=32 if stack_hidden_dim is None else stack_hidden_dim,
        )
    if option is ProjectionModelOptions.RECURRENT_LINEAR:
        steps = 2 if recurrent_max_steps is None else recurrent_max_steps
        return RecurrentLayerConfig(
            max_steps=steps,
            initial_iterations=steps,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=projection_layer_config(
                LinearLayerConfig(bias_flag=True if bias_flag is None else bias_flag)
            ),
        )
    if option is ProjectionModelOptions.LINEAR:
        return LinearLayerConfig(bias_flag=True if bias_flag is None else bias_flag)
    if option is not None:
        raise ValueError("Unsupported projection model option.")
    if bias_flag is None:
        return config
    if config is None:
        return LinearLayerConfig(bias_flag=bias_flag)
    if type(config) is not LinearLayerConfig:
        raise ValueError("The model bias flag requires a LINEAR projection model.")
    return replace(config, bias_flag=bias_flag)


def prefer_scoped_projection_model(
    updates: dict[str, Any], prefix: str, *, has_scoped_model: bool
) -> None:
    """Clear inherited flat controls when a path supplies a legacy model config."""
    if has_scoped_model:
        updates.setdefault(f"{prefix}_model_option", None)
        updates.setdefault(f"{prefix}_model_bias_flag", None)
