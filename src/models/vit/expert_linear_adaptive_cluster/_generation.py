"""Package-owned options for adaptive parameter generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any

from emperor.augmentations.adaptive_parameters import (
    DynamicWeightConfig,
    LowRankFactorSourceOptions,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.sampler import SamplerConfig
from models import _attention_settings
from models._adaptive_generation_options import GenerationSettingsReader
from models._adaptive_generation_options import generation_keys as generation_keys
from models._adaptive_generator_stack import (
    build_mixture_generation,
    build_weight_generation,
)

if TYPE_CHECKING:
    from models.vit.expert_linear_adaptive_cluster._adaptive_generator_stack_config_factory import (
        AdaptiveGeneratorStackConfigFactory,
    )


@dataclass(frozen=True)
class AdaptiveGeneratorStackSource:
    independent_flag: bool
    hidden_dim: int | None
    layer_norm_position: LayerNormPositionOptions | None
    normalization: NormalizationOptions | None = field(default=None, kw_only=True)
    num_layers: int | None
    activation: ActivationOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    bias_flag: bool | None

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        _attention_settings.validate_stack_source(validation, self)


@dataclass(frozen=True)
class WeightGenerationOptions:
    input_factor_source: LowRankFactorSourceOptions | None = None
    output_factor_source: LowRankFactorSourceOptions | None = None
    mixture_num_experts: int | None = None
    input_factor_generator_stack: AdaptiveGeneratorStackSource | None = None
    output_factor_generator_stack: AdaptiveGeneratorStackSource | None = None
    coefficient_generator_stack: AdaptiveGeneratorStackSource | None = None
    mixture_router_generator_stack: AdaptiveGeneratorStackSource | None = None


@dataclass(frozen=True)
class BiasGenerationOptions:
    mixture_num_experts: int | None = None
    mixture_top_k: int | None = None
    mixture_normalize_probabilities_flag: bool | None = None
    mixture_router_generator_stack: AdaptiveGeneratorStackSource | None = None


@dataclass(frozen=True, slots=True)
class MixtureGenerationConfig:
    num_experts: int | None = None
    top_k: int | None = None
    sampler_config: SamplerConfig | None = None


@dataclass(frozen=True, slots=True)
class WeightGenerationConfig:
    input_factor_source: LowRankFactorSourceOptions | None = None
    output_factor_source: LowRankFactorSourceOptions | None = None
    input_factor_model_config: LayerStackConfig | None = None
    output_factor_model_config: LayerStackConfig | None = None
    coefficient_model_config: LayerStackConfig | None = None
    num_experts: int | None = None
    router_model_config: LayerStackConfig | None = None


class _GenerationSettings(
    GenerationSettingsReader[WeightGenerationOptions, BiasGenerationOptions]
):
    stack_source_type = AdaptiveGeneratorStackSource
    weight_options_type = WeightGenerationOptions
    bias_options_type = BiasGenerationOptions


def apply_generation_options(
    options: dict[str, Any], values: Mapping[str, Any], config_module: ModuleType
) -> dict[str, Any]:
    return _GenerationSettings.apply(options, values, config_module)


def build_weight_generation_config(
    options: WeightGenerationOptions,
    factory: AdaptiveGeneratorStackConfigFactory,
    weight_option: type[DynamicWeightConfig],
    weight_source: AdaptiveGeneratorStackSource | None = None,
) -> WeightGenerationConfig:
    return build_weight_generation(
        options,
        factory,
        weight_option,
        weight_source,
        weight_type=WeightGenerationConfig,
        default=WeightGenerationConfig,
    )


def build_mixture_generation_config(
    options: BiasGenerationOptions,
    factory: AdaptiveGeneratorStackConfigFactory,
) -> MixtureGenerationConfig:
    return build_mixture_generation(
        options, factory.build_config_from_source, MixtureGenerationConfig
    )
