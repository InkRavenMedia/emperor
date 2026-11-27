from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Final, Protocol, TypeVar

from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    BroadcastSumDynamicWeightConfig,
    DiagonalAxisMaskConfig,
    DiagonallyModulatedLowRankDynamicWeightConfig,
    DualModelDynamicWeightConfig,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    LowRankFactorSourceOptions,
    MaskDimensionOptions,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    PerAxisScoreMaskConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    TopSliceAxisMaskConfig,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
    WeightInformedScoreAxisMaskConfig,
    WeightNormalizationConfig,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.layers import LayerStackConfig
from emperor.sampler import RouterConfig, SamplerConfig


@dataclass(frozen=True, slots=True)
class MixtureGeneration:
    num_experts: int | None = None
    top_k: int | None = None
    sampler_config: SamplerConfig | None = None


@dataclass(frozen=True, slots=True)
class WeightGeneration:
    input_factor_source: LowRankFactorSourceOptions | None = None
    output_factor_source: LowRankFactorSourceOptions | None = None
    input_factor_model_config: LayerStackConfig | None = None
    output_factor_model_config: LayerStackConfig | None = None
    coefficient_model_config: LayerStackConfig | None = None
    num_experts: int | None = None
    router_model_config: LayerStackConfig | None = None


class _Mixture(Protocol):
    num_experts: Final[int | None]
    top_k: Final[int | None]
    sampler_config: Final[SamplerConfig | None]


class _Generation(Protocol):
    input_factor_source: Final[LowRankFactorSourceOptions | None]
    output_factor_source: Final[LowRankFactorSourceOptions | None]
    input_factor_model_config: Final[LayerStackConfig | None]
    output_factor_model_config: Final[LayerStackConfig | None]
    coefficient_model_config: Final[LayerStackConfig | None]
    num_experts: Final[int | None]
    router_model_config: Final[LayerStackConfig | None]


class _MaskDimensions(Protocol):
    @property
    def mask_dimension_option(self) -> MaskDimensionOptions: ...

    @property
    def mask_transition_width(self) -> float: ...


class MaskOptions(_MaskDimensions, Protocol):
    row_mask_option: Final[type[AxisMaskConfig] | None]
    mask_threshold: Final[float]
    mask_surrogate_scale: Final[float]
    mask_floor: Final[float]


class ScopedMaskOptions(Protocol):
    dimension_option: Final[MaskDimensionOptions]
    transition_width: Final[float]
    threshold: Final[float]
    surrogate_scale: Final[float]
    floor: Final[float]


class _ScopedDimensions:
    def __init__(self, options: ScopedMaskOptions) -> None:
        self.options = options

    @property
    def mask_dimension_option(self) -> MaskDimensionOptions:
        return self.options.dimension_option

    @property
    def mask_transition_width(self) -> float:
        return self.options.transition_width


def build_weight_config(
    weight_option: type[DynamicWeightConfig] | None,
    *,
    generation: _Generation | None = None,
    build_generation: Callable[[], _Generation | None] | None = None,
    generator_depth: int,
    decay_schedule: WeightDecayScheduleOptions,
    decay_rate: float,
    decay_warmup_batches: int,
    normalization_option: WeightNormalizationOptions,
    normalization_learnable_scale_flag: bool = False,
    normalization_position_option: WeightNormalizationPositionOptions,
    bank_expansion_factor: BankExpansionFactorOptions,
    model_config: LayerStackConfig | None = None,
) -> DynamicWeightConfig | None:
    if weight_option is None:
        return None
    if (
        weight_option
        in (
            LayeredWeightedBankDynamicWeightConfig,
            SoftWeightedBankDynamicWeightConfig,
            MatrixWeightsMixtureConfig,
        )
        and normalization_option is not WeightNormalizationOptions.DISABLED
    ):
        raise ValueError(
            f"{weight_option.__name__} does not support weight normalization; "
            f"normalization_option={normalization_option!r} is not a supported combination."
        )
    if generation is None and build_generation is not None:
        generation = build_generation()
    if weight_option is MatrixWeightsMixtureConfig:
        if generation is None:
            generation = WeightGeneration()
        return MatrixWeightsMixtureConfig(
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
            model_config=generation.router_model_config or model_config,
            num_experts=generation.num_experts,
        )
    if (
        weight_option is LayeredWeightedBankDynamicWeightConfig
        or weight_option is SoftWeightedBankDynamicWeightConfig
    ):
        return weight_option(
            generator_depth=generator_depth,
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
            model_config=model_config,
            bank_expansion_factor=bank_expansion_factor,
        )
    if weight_option is DiagonallyModulatedLowRankDynamicWeightConfig:
        if generation is None:
            generation = WeightGeneration()
        normalization_config = _normalization_config(
            normalization_option, normalization_learnable_scale_flag
        )
        return DiagonallyModulatedLowRankDynamicWeightConfig(
            generator_depth=generator_depth,
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
            model_config=model_config,
            normalization_config=normalization_config,
            input_factor_source=generation.input_factor_source,
            output_factor_source=generation.output_factor_source,
            input_factor_model_config=generation.input_factor_model_config,
            output_factor_model_config=generation.output_factor_model_config,
            coefficient_model_config=generation.coefficient_model_config,
        )
    if weight_option in (
        BroadcastSumDynamicWeightConfig,
        SingleModelDynamicWeightConfig,
        DualModelDynamicWeightConfig,
        LowRankDynamicWeightConfig,
        HypernetworkDynamicWeightConfig,
    ):
        position = WeightNormalizationPositionOptions.DISABLED
        if weight_option in (
            SingleModelDynamicWeightConfig,
            DualModelDynamicWeightConfig,
        ):
            position = normalization_position_option
        normalization_config = _normalization_config(
            normalization_option, normalization_learnable_scale_flag, position
        )
        return weight_option(
            generator_depth=generator_depth,
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
            model_config=model_config,
            normalization_config=normalization_config,
        )
    return weight_option(
        generator_depth=generator_depth,
        decay_schedule=decay_schedule,
        decay_rate=decay_rate,
        decay_warmup_batches=decay_warmup_batches,
        model_config=model_config,
    )


def build_bias_config(
    bias_option: type[DynamicBiasConfig] | None,
    *,
    generation: _Mixture | None = None,
    build_generation: Callable[[], _Mixture] | None = None,
    decay_schedule: WeightDecayScheduleOptions,
    decay_rate: float,
    decay_warmup_batches: int,
    bank_expansion_factor: BankExpansionFactorOptions,
    model_config: LayerStackConfig | None = None,
) -> DynamicBiasConfig | None:
    if bias_option is None:
        return None
    build_bias = bias_option
    if bias_option is MatrixBiasMixtureConfig:
        if generation is None and build_generation is not None:
            generation = build_generation()
        if generation is None:
            generation = MixtureGeneration()
        build_bias = partial(
            MatrixBiasMixtureConfig,
            num_experts=generation.num_experts,
            top_k=generation.top_k,
            sampler_config=generation.sampler_config,
        )
    elif bias_option is WeightedBankDynamicBiasConfig:
        build_bias = partial(
            WeightedBankDynamicBiasConfig, bank_expansion_factor=bank_expansion_factor
        )
    return build_bias(
        decay_schedule=decay_schedule,
        decay_rate=decay_rate,
        decay_warmup_batches=decay_warmup_batches,
        model_config=model_config,
    )


def build_diagonal_config(
    diagonal_option: type[DynamicDiagonalConfig] | None,
    *,
    model_config: LayerStackConfig | None = None,
) -> DynamicDiagonalConfig | None:
    if diagonal_option is None:
        return None
    return diagonal_option(model_config=model_config)


def build_mask_config(
    options: MaskOptions,
    *,
    model_config: LayerStackConfig | None = None,
) -> AxisMaskConfig | None:
    option = options.row_mask_option
    if option is None:
        return None
    build_mask = axis_mask_constructor(option, options)
    return build_mask(
        mask_threshold=options.mask_threshold,
        mask_surrogate_scale=options.mask_surrogate_scale,
        mask_floor=options.mask_floor,
        model_config=model_config,
    )


def build_scoped_mask(
    option: type[AxisMaskConfig],
    options: ScopedMaskOptions,
    *,
    model_config: LayerStackConfig | None,
) -> AxisMaskConfig:
    build_mask = axis_mask_constructor(option, _ScopedDimensions(options))
    return build_mask(
        mask_threshold=options.threshold,
        mask_surrogate_scale=options.surrogate_scale,
        mask_floor=options.floor,
        model_config=model_config,
    )


class _BuildMask(Protocol):
    def __call__(
        self,
        *,
        mask_threshold: float,
        mask_surrogate_scale: float,
        mask_floor: float,
        model_config: LayerStackConfig | None,
    ) -> AxisMaskConfig: ...


def axis_mask_constructor(
    option: type[AxisMaskConfig],
    options: _MaskDimensions,
) -> _BuildMask:
    build_mask = option
    if option is TopSliceAxisMaskConfig:
        build_mask = partial(
            TopSliceAxisMaskConfig,
            mask_dimension_option=options.mask_dimension_option,
            mask_transition_width=options.mask_transition_width,
        )
    elif option is DiagonalAxisMaskConfig:
        build_mask = partial(
            DiagonalAxisMaskConfig, mask_transition_width=options.mask_transition_width
        )
    elif (
        option is PerAxisScoreMaskConfig or option is WeightInformedScoreAxisMaskConfig
    ):
        build_mask = partial(
            option, mask_dimension_option=options.mask_dimension_option
        )
    return build_mask


def _normalization_config(
    method: WeightNormalizationOptions,
    learnable_scale: bool,
    position: WeightNormalizationPositionOptions = WeightNormalizationPositionOptions.DISABLED,
) -> WeightNormalizationConfig | None:
    if method is WeightNormalizationOptions.DISABLED:
        return None
    return WeightNormalizationConfig(
        normalization_option=method,
        normalization_position_option=position,
        normalization_learnable_scale_flag=learnable_scale,
    )


def build_mixture_sampler(
    router_model_config: LayerStackConfig | None,
    num_experts: int | None,
    top_k: int | None,
    normalize_probabilities: bool | None,
) -> SamplerConfig:
    router = RouterConfig(
        num_experts=num_experts,
        noisy_topk_flag=False,
        model_config=router_model_config,
    )
    if normalize_probabilities is None:
        normalize_probabilities = (top_k or 0) > 1
    return SamplerConfig(
        num_experts=num_experts,
        top_k=top_k,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        noisy_topk_flag=False,
        coefficient_of_variation_loss_weight=0.0,
        switch_loss_weight=0.0,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        normalize_probabilities_flag=normalize_probabilities,
        router_config=router,
    )


_AdaptiveParameterConfigT = TypeVar("_AdaptiveParameterConfigT")


def resolve_enabled_adaptive_parameter_option(
    *,
    option_flag: bool,
    option: type[_AdaptiveParameterConfigT] | None,
    option_flag_name: str,
    option_name: str,
) -> type[_AdaptiveParameterConfigT] | None:
    if not option_flag:
        return None
    if option is None:
        raise ValueError(f"{option_name} must be set when {option_flag_name} is True.")
    return option
