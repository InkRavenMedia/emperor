from dataclasses import dataclass, field, replace

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.parametric import ClipParameterOptions, GeneratorBiasMixtureConfig
from model_runtime.packages.runtime_values import ResolvedRuntimeOptions
from models._recurrent_variants import ParametricRecurrentOptions
from models.parametric.parametric_generator import SETTINGS_VALIDATION

from . import config
from ._residual import (
    ResidualStackOptions,
    ResidualStackSource,
    resolve_residual_stack_options,
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


@dataclass(frozen=True)
class ParametricGeneratorStackOptions:
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    dropout_probability: float

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.positive_integer(
            "generator_stack_hidden_dim", self.hidden_dim
        )
        SETTINGS_VALIDATION.positive_integer(
            "generator_stack_num_layers", self.num_layers
        )
        SETTINGS_VALIDATION.dropout(
            "generator_stack_dropout_probability", self.dropout_probability
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _ParametricGeneratorConfigBuilderImplementation:
    """Bind the legacy builder arguments under their original Python error name."""

    batch_size: int = config.BATCH_SIZE
    learning_rate: float = config.LEARNING_RATE
    input_dim: int = config.INPUT_DIM
    hidden_dim: int = config.HIDDEN_DIM
    output_dim: int = config.OUTPUT_DIM
    stack_num_layers: int = config.STACK_NUM_LAYERS
    stack_activation: ActivationOptions = config.STACK_ACTIVATION
    stack_residual_connection_option: type[ResidualConfig] | None = (
        config.STACK_RESIDUAL_CONNECTION_OPTION
    )
    stack_residual_model_flag: bool = config.STACK_RESIDUAL_MODEL_FLAG
    stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY
    residual_coefficient_activation: ActivationOptions = (
        config.RESIDUAL_COEFFICIENT_ACTIVATION
    )
    residual_stack_independent_flag: bool = config.RESIDUAL_STACK_INDEPENDENT_FLAG
    residual_stack_hidden_dim: int | None = config.RESIDUAL_STACK_HIDDEN_DIM
    residual_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.RESIDUAL_STACK_LAYER_NORM_POSITION
    )
    residual_stack_num_layers: int | None = config.RESIDUAL_STACK_NUM_LAYERS
    residual_stack_activation: ActivationOptions | None = (
        config.RESIDUAL_STACK_ACTIVATION
    )
    residual_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION
    )
    residual_stack_residual_model_flag: bool = config.RESIDUAL_STACK_RESIDUAL_MODEL_FLAG
    residual_stack_dropout_probability: float | None = (
        config.RESIDUAL_STACK_DROPOUT_PROBABILITY
    )
    residual_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION
    )
    residual_stack_apply_output_postprocessing_flag: bool | None = (
        config.RESIDUAL_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    residual_stack_bias_flag: bool | None = config.RESIDUAL_STACK_BIAS_FLAG
    adaptive_mixture_top_k: int = config.ADAPTIVE_MIXTURE_TOP_K
    adaptive_mixture_num_experts: int = config.ADAPTIVE_MIXTURE_NUM_EXPERTS
    adaptive_mixture_weighted_parameters_flag: bool = (
        config.ADAPTIVE_MIXTURE_WEIGHTED_PARAMETERS_FLAG
    )
    adaptive_mixture_clip_parameter_option: ClipParameterOptions = (
        config.ADAPTIVE_MIXTURE_CLIP_PARAMETER_OPTION
    )
    adaptive_mixture_clip_range: float = config.ADAPTIVE_MIXTURE_CLIP_RANGE
    adaptive_bias_option: type[GeneratorBiasMixtureConfig] | None = (
        config.ADAPTIVE_BIAS_OPTION
    )
    sampler_threshold: float = config.SAMPLER_THRESHOLD
    sampler_filter_above_threshold: bool = config.SAMPLER_FILTER_ABOVE_THRESHOLD
    sampler_num_topk_samples: int = config.SAMPLER_NUM_TOPK_SAMPLES
    sampler_normalize_probabilities_flag: bool = (
        config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG
    )
    sampler_noisy_topk_flag: bool = config.SAMPLER_NOISY_TOPK_FLAG
    sampler_coefficient_of_variation_loss_weight: float = (
        config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT
    )
    sampler_switch_loss_weight: float = config.SAMPLER_SWITCH_LOSS_WEIGHT
    sampler_zero_centred_loss_weight: float = config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT
    sampler_mutual_information_loss_weight: float = (
        config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT
    )
    generator_stack_num_layers: int = config.GENERATOR_STACK_NUM_LAYERS
    generator_stack_hidden_dim: int = config.GENERATOR_STACK_HIDDEN_DIM
    generator_stack_activation: ActivationOptions = config.GENERATOR_STACK_ACTIVATION
    generator_stack_dropout_probability: float = (
        config.GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    stack_options: ParametricStackOptions | None = None
    mixture_options: ParametricMixtureOptions | None = None
    sampler_options: ParametricSamplerOptions | None = None
    router_options: ParametricRouterOptions | None = None
    generator_stack_options: ParametricGeneratorStackOptions | None = None
    residual_stack_options: ResidualStackOptions | None = None
    residual_stack_residual_block_size: int | None = None
    residual_stack_residual_rms_norm_epsilon: float | None = None
    stack_residual_block_size: int | None = None
    stack_residual_rms_norm_epsilon: float | None = None
    residual_stack_normalization: NormalizationOptions | None = (
        config.RESIDUAL_STACK_NORMALIZATION
    )


@dataclass(frozen=True, slots=True)
class ParametricGeneratorConstructionOptions:
    batch_size: int
    learning_rate: float
    input_dim: int
    output_dim: int
    stack: ParametricStackOptions
    mixture: ParametricMixtureOptions
    sampler: ParametricSamplerOptions
    router: ParametricRouterOptions
    generator_stack: ParametricGeneratorStackOptions
    residual_stack: ResidualStackOptions
    adaptive_bias_option: type[GeneratorBiasMixtureConfig] | None

    @classmethod
    def from_inputs(
        cls, values: _ParametricGeneratorConfigBuilderImplementation
    ) -> "ParametricGeneratorConstructionOptions":
        stack = values.stack_options or ParametricStackOptions(
            hidden_dim=values.hidden_dim,
            num_layers=values.stack_num_layers,
            activation=values.stack_activation,
            residual_connection_option=values.stack_residual_connection_option,
            residual_block_size=values.stack_residual_block_size,
            residual_rms_norm_epsilon=values.stack_residual_rms_norm_epsilon,
            residual_model_flag=values.stack_residual_model_flag,
            dropout_probability=values.stack_dropout_probability,
        )
        residual_stack = (
            values.residual_stack_options
            or resolve_residual_stack_options(
                ResidualStackSource(
                    coefficient_activation=values.residual_coefficient_activation,
                    independent_flag=values.residual_stack_independent_flag,
                    hidden_dim=values.residual_stack_hidden_dim,
                    num_layers=values.residual_stack_num_layers,
                    activation=values.residual_stack_activation,
                    layer_norm_position=values.residual_stack_layer_norm_position,
                    normalization=values.residual_stack_normalization,
                    residual_connection_option=values.residual_stack_residual_connection_option,
                    residual_block_size=values.residual_stack_residual_block_size,
                    residual_rms_norm_epsilon=values.residual_stack_residual_rms_norm_epsilon,
                    residual_model_flag=values.residual_stack_residual_model_flag,
                    dropout_probability=values.residual_stack_dropout_probability,
                    last_layer_bias_option=values.residual_stack_last_layer_bias_option,
                    apply_output_postprocessing_flag=values.residual_stack_apply_output_postprocessing_flag,
                    bias_flag=values.residual_stack_bias_flag,
                ),
                stack,
            )
        )
        mixture = values.mixture_options or ParametricMixtureOptions(
            top_k=values.adaptive_mixture_top_k,
            num_experts=values.adaptive_mixture_num_experts,
            weighted_parameters_flag=values.adaptive_mixture_weighted_parameters_flag,
            clip_parameter_option=values.adaptive_mixture_clip_parameter_option,
            clip_range=values.adaptive_mixture_clip_range,
        )
        sampler = values.sampler_options or ParametricSamplerOptions(
            threshold=values.sampler_threshold,
            filter_above_threshold=values.sampler_filter_above_threshold,
            num_topk_samples=values.sampler_num_topk_samples,
            normalize_probabilities_flag=values.sampler_normalize_probabilities_flag,
            noisy_topk_flag=values.sampler_noisy_topk_flag,
            coefficient_of_variation_loss_weight=values.sampler_coefficient_of_variation_loss_weight,
            switch_loss_weight=values.sampler_switch_loss_weight,
            zero_centred_loss_weight=values.sampler_zero_centred_loss_weight,
            mutual_information_loss_weight=values.sampler_mutual_information_loss_weight,
        )
        router = values.router_options or ParametricRouterOptions(
            activation=stack.activation
        )
        generator_stack = (
            values.generator_stack_options
            or ParametricGeneratorStackOptions(
                hidden_dim=values.generator_stack_hidden_dim,
                num_layers=values.generator_stack_num_layers,
                activation=values.generator_stack_activation,
                dropout_probability=values.generator_stack_dropout_probability,
            )
        )
        return cls(
            batch_size=values.batch_size,
            learning_rate=values.learning_rate,
            input_dim=values.input_dim,
            output_dim=values.output_dim,
            stack=stack,
            mixture=mixture,
            sampler=sampler,
            router=router,
            generator_stack=generator_stack,
            residual_stack=residual_stack,
            adaptive_bias_option=values.adaptive_bias_option,
        )

    @property
    def hidden_dim(self) -> int:
        return self.stack.hidden_dim

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.dimensions(self)


@dataclass(frozen=True, slots=True)
class RuntimeOptions(ResolvedRuntimeOptions, ParametricRecurrentOptions):
    def construction_options(self) -> ParametricGeneratorConstructionOptions:
        inputs = replace(_DEFAULT_CONSTRUCTION_INPUTS, **self._values)
        return ParametricGeneratorConstructionOptions.from_inputs(inputs)

    def __post_init__(self) -> None:
        ResolvedRuntimeOptions.__post_init__(self)
        self.construction_options()


_DEFAULT_CONSTRUCTION_INPUTS = _ParametricGeneratorConfigBuilderImplementation()
