"""Flat CLI controls shared by source memory and sequence expansion."""

from copy import deepcopy
from dataclasses import replace
from enum import Enum, auto
from typing import Any

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdditiveDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    WeightDecayScheduleOptions,
)
from emperor.config import ConfigBase
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig


class ProjectionModelOptions(Enum):
    LINEAR = auto()
    LINEAR_STACK = auto()
    RECURRENT_LINEAR = auto()
    ADAPTIVE_LINEAR = auto()


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
    layer_config = projection_layer_config(model)
    return LayerStackConfig(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        apply_output_postprocessing_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        layer_config=layer_config,
    )


def default_adaptive_projection(
    *, hidden_dim: int, num_layers: int, bias_flag: bool = True
) -> AdaptiveLinearLayerConfig:
    generator = projection_stack_config(
        LinearLayerConfig(bias_flag=True), hidden_dim=hidden_dim, num_layers=num_layers
    )
    if bias_flag:
        augmentation = AdaptiveParameterAugmentationConfig(
            bias_config=AdditiveDynamicBiasConfig(
                decay_schedule=WeightDecayScheduleOptions.DISABLED,
                decay_rate=0.0,
                decay_warmup_batches=0,
                model_config=generator,
            ),
        )
    else:
        augmentation = AdaptiveParameterAugmentationConfig(
            weight_config=SingleModelDynamicWeightConfig(
                generator_depth=1,
                decay_schedule=WeightDecayScheduleOptions.DISABLED,
                decay_rate=0.0,
                decay_warmup_batches=0,
                model_config=generator,
            ),
        )
    return AdaptiveLinearLayerConfig(
        bias_flag=bias_flag, adaptive_augmentation_config=augmentation
    )


def _adaptive_memory_projection(
    template_source: ConfigBase | None, bias_flag: bool | None
) -> AdaptiveLinearLayerConfig:
    template = (
        LayerConfig.find_processing_linear_template(template_source)
        if template_source is not None
        else None
    )
    if isinstance(template, AdaptiveLinearLayerConfig):
        augmentation = template.adaptive_augmentation_config
        if augmentation is not None:
            # Each memory row is already a chunk summary.
            augmentation = replace(deepcopy(augmentation), grouping_config=None)
            return replace(
                template,
                input_dim=None,
                output_dim=None,
                bias_flag=template.bias_flag if bias_flag is None else bias_flag,
                adaptive_augmentation_config=augmentation,
            )
    return default_adaptive_projection(
        hidden_dim=32, num_layers=1, bias_flag=bias_flag is not False
    )


def resolve_projection_model(
    config: ConfigBase | None,
    option: ProjectionModelOptions | None,
    bias_flag: bool | None,
    *,
    stack_num_layers: int | None = None,
    stack_hidden_dim: int | None = None,
    recurrent_max_steps: int | None = None,
    template_source: ConfigBase | None = None,
) -> ConfigBase | None:
    if option is ProjectionModelOptions.ADAPTIVE_LINEAR:
        if isinstance(config, AdaptiveLinearLayerConfig):
            return config if bias_flag is None else replace(config, bias_flag=bias_flag)
        return _adaptive_memory_projection(template_source, bias_flag)
    projection_bias = True
    if bias_flag is not None:
        projection_bias = bias_flag
    if option is ProjectionModelOptions.LINEAR_STACK:
        linear_config = LinearLayerConfig(bias_flag=projection_bias)
        if stack_num_layers is None:
            stack_num_layers = 2
        if stack_hidden_dim is None:
            stack_hidden_dim = 32
        return projection_stack_config(
            linear_config,
            num_layers=stack_num_layers,
            hidden_dim=stack_hidden_dim,
        )
    if option is ProjectionModelOptions.RECURRENT_LINEAR:
        if recurrent_max_steps is None:
            recurrent_max_steps = 2
        linear_config = LinearLayerConfig(bias_flag=projection_bias)
        block_config = projection_layer_config(linear_config)
        return RecurrentLayerConfig(
            max_steps=recurrent_max_steps,
            initial_iterations=recurrent_max_steps,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            recurrent_normalization=NormalizationOptions.LAYER_NORM,
            block_config=block_config,
        )
    if option is ProjectionModelOptions.LINEAR:
        return LinearLayerConfig(bias_flag=projection_bias)
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
    """Use the scoped component configuration for this projection."""
    if has_scoped_model:
        updates.setdefault(f"{prefix}_model_option", None)
        updates.setdefault(f"{prefix}_model_bias_flag", None)
