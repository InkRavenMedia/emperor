from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.parametric import ClipParameterOptions
from model_runtime.packages.runtime_values import (
    ResolvedRuntimeOptions,
    validate_runtime_default_value_types,
)
from models._recurrent_variants import ParametricRecurrentOptions

from . import config

if TYPE_CHECKING:
    from ._runtime_construction import ParametricVectorConstructionOptions


from models.parametric.parametric_vector import SETTINGS_VALIDATION

_ROLE_OPTION_KEYS = frozenset(
    {
        "stack_options",
        "residual_stack_options",
        "mixture_options",
        "sampler_options",
        "router_options",
    }
)


@dataclass(frozen=True)
class ParametricStackOptions:
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float
    layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
    apply_output_postprocessing_flag: bool = True
    bias_flag: bool = True

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.positive_integer("stack_hidden_dim", self.hidden_dim)
        SETTINGS_VALIDATION.stack(self)


@dataclass(frozen=True)
class ParametricMixtureOptions:
    top_k: int
    num_experts: int
    weighted_parameters_flag: bool
    clip_parameter_option: ClipParameterOptions
    clip_range: float


@dataclass(frozen=True)
class ParametricSamplerOptions:
    threshold: float
    filter_above_threshold: bool
    num_topk_samples: int
    normalize_probabilities_flag: bool
    noisy_topk_flag: bool
    coefficient_of_variation_loss_weight: float
    switch_loss_weight: float
    zero_centred_loss_weight: float
    mutual_information_loss_weight: float


@dataclass(frozen=True)
class ParametricRouterOptions:
    activation: ActivationOptions
    noisy_topk_flag: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeOptions(ResolvedRuntimeOptions, ParametricRecurrentOptions):
    """Validated flat values with one deep package-construction projection."""

    def __post_init__(self) -> None:
        ResolvedRuntimeOptions.__post_init__(self)
        flat_values = {
            key: value
            for key, value in self._values.items()
            if key not in _ROLE_OPTION_KEYS
        }
        validate_runtime_default_value_types(
            flat_values,
            package="models.parametric.parametric_vector",
            config_module=config,
        )
        self.construction_options()

    def construction_options(self) -> ParametricVectorConstructionOptions:
        from ._runtime_construction import resolve_runtime_construction

        return resolve_runtime_construction(self)
