from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Final, Generic, Protocol, TypeVar

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    AntiDynamicDiagonalConfig,
    AxisMaskConfig,
    BankExpansionFactorOptions,
    BroadcastSumDynamicWeightConfig,
    CombinedDynamicDiagonalConfig,
    DiagonalAxisMaskConfig,
    DiagonallyModulatedLowRankDynamicWeightConfig,
    DualModelDynamicWeightConfig,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GeneratorDynamicBiasConfig,
    GroupingConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    LowRankFactorSourceOptions,
    MaskDimensionOptions,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    MultiplicativeDynamicBiasConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    SigmoidGatedDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    StandardDynamicDiagonalConfig,
    TanhGatedDynamicBiasConfig,
    TopSliceAxisMaskConfig,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
    WeightInformedScoreAxisMaskConfig,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.config import ConfigBase
from emperor.layers import LayerStackConfig
from emperor.sampler import SamplerConfig
from models._adaptive_generator_stack import build_generator_stack
from models._adaptive_parameters import (
    MixtureGeneration,
    WeightGeneration,
    axis_mask_constructor,
    build_bias_config,
    build_mixture_sampler,
    build_weight_config,
)
from models._residual_stack import ResidualStackSettings, StackDefaults


class _GeneratorSource(Protocol):
    independent_flag: Final[bool]


_Source = TypeVar("_Source", bound=_GeneratorSource)
_Stack = TypeVar("_Stack", bound=StackDefaults)
_SourceCo = TypeVar("_SourceCo", bound=_GeneratorSource, covariant=True)
_StackCo = TypeVar("_StackCo", bound=StackDefaults, covariant=True)


class _ParameterSettings(Protocol[_SourceCo, _StackCo]):
    weight_input_factor_source: Final[LowRankFactorSourceOptions | None]
    weight_output_factor_source: Final[LowRankFactorSourceOptions | None]
    weight_mixture_num_experts: Final[int | None]
    bias_mixture_num_experts: Final[int | None]
    weight_mixture_top_k: Final[int | None]
    bias_mixture_top_k: Final[int | None]
    weight_mixture_normalize_probabilities_flag: Final[bool | None]
    bias_mixture_normalize_probabilities_flag: Final[bool | None]
    weight_input_factor_generator_stack_options: Final[_SourceCo]
    weight_output_factor_generator_stack_options: Final[_SourceCo]
    weight_coefficient_generator_stack_options: Final[_SourceCo]
    weight_mixture_router_generator_stack_options: Final[_SourceCo]
    bias_mixture_router_generator_stack_options: Final[_SourceCo]
    grouping_config: Final[GroupingConfig | None]
    weight_option_flag: Final[bool]
    weight_option: Final[type[DynamicWeightConfig] | None]
    generator_depth: Final[int]
    weight_decay_schedule: Final[WeightDecayScheduleOptions]
    weight_decay_rate: Final[float]
    weight_decay_warmup_batches: Final[int]
    weight_normalization_option: Final[WeightNormalizationOptions]
    weight_normalization_position_option: Final[WeightNormalizationPositionOptions]
    weight_bank_expansion_factor: Final[BankExpansionFactorOptions]
    bias_option_flag: Final[bool]
    bias_option: Final[type[DynamicBiasConfig] | None]
    bias_decay_schedule: Final[WeightDecayScheduleOptions]
    bias_decay_rate: Final[float]
    bias_decay_warmup_batches: Final[int]
    bias_bank_expansion_factor: Final[BankExpansionFactorOptions]
    diagonal_option_flag: Final[bool]
    diagonal_option: Final[type[DynamicDiagonalConfig] | None]
    mask_option_flag: Final[bool]
    row_mask_option: Final[type[AxisMaskConfig] | None]
    mask_threshold: Final[float]
    mask_surrogate_scale: Final[float]
    mask_floor: Final[float]
    mask_dimension_option: Final[MaskDimensionOptions]
    mask_transition_width: Final[float]
    generator_stack_options: Final[_StackCo]
    weight_generator_stack_options: Final[_SourceCo]
    bias_generator_stack_options: Final[_SourceCo]
    diagonal_generator_stack_options: Final[_SourceCo]
    mask_generator_stack_options: Final[_SourceCo]
    weight_normalization_learnable_scale_flag: Final[bool]


class AdaptiveAugmentationFactory(Generic[_Source, _Stack]):
    def __init__(
        self,
        resolve_stack: Callable[[_Source, _Stack], _Stack],
        residual_stack: ResidualStackSettings,
    ) -> None:
        self.__resolve_stack = resolve_stack
        self.__residual_stack = residual_stack

    def build(
        self, options: _ParameterSettings[_Source, _Stack]
    ) -> AdaptiveParameterAugmentationConfig:
        generator = build_generator_stack(
            options.generator_stack_options,
            self.__residual_stack,
        )
        weight = self.__weight(options)
        bias = self.__bias(options)
        diagonal = self.__diagonal(options)
        mask = self.__mask(options)
        return AdaptiveParameterAugmentationConfig(
            grouping_config=options.grouping_config,
            weight_config=weight,
            bias_config=bias,
            diagonal_config=diagonal,
            mask_config=mask,
            model_config=generator,
        )

    def __weight(
        self, options: _ParameterSettings[_Source, _Stack]
    ) -> DynamicWeightConfig | None:
        weight_stack_options = self.__resolve_stack(
            options.weight_generator_stack_options, options.generator_stack_options
        )
        weight_input_factor_model_config = self.__component_generator(
            options.weight_input_factor_generator_stack_options,
            weight_stack_options,
        )
        weight_output_factor_model_config = self.__component_generator(
            options.weight_output_factor_generator_stack_options,
            weight_stack_options,
        )
        weight_coefficient_model_config = self.__component_generator(
            options.weight_coefficient_generator_stack_options,
            weight_stack_options,
        )
        weight_model_config = None
        if options.weight_generator_stack_options.independent_flag:
            weight_model_config = build_generator_stack(
                weight_stack_options, self.__residual_stack
            )
        weight_option = None
        if options.weight_option_flag:
            weight_option = options.weight_option
        weight_sampler_config = None
        if weight_option is MatrixWeightsMixtureConfig:
            weight_sampler_config = self.__mixture_sampler(
                options.weight_mixture_num_experts,
                options.weight_mixture_top_k,
                options.weight_mixture_normalize_probabilities_flag,
                options.weight_mixture_router_generator_stack_options,
                weight_stack_options,
            )
        weight_inputs = _WeightInputs(
            num_experts=options.weight_mixture_num_experts,
            top_k=options.weight_mixture_top_k,
            sampler_config=weight_sampler_config,
            generator_depth=options.generator_depth,
            input_factor_source=options.weight_input_factor_source,
            output_factor_source=options.weight_output_factor_source,
            input_factor_model_config=weight_input_factor_model_config,
            output_factor_model_config=weight_output_factor_model_config,
            coefficient_model_config=weight_coefficient_model_config,
            decay_schedule=options.weight_decay_schedule,
            decay_rate=options.weight_decay_rate,
            decay_warmup_batches=options.weight_decay_warmup_batches,
            normalization_option=options.weight_normalization_option,
            normalization_learnable_scale_flag=options.weight_normalization_learnable_scale_flag,
            normalization_position_option=options.weight_normalization_position_option,
            bank_expansion_factor=options.weight_bank_expansion_factor,
            model_config=weight_model_config,
        )
        return weight_inputs.build(weight_option)

    def __bias(
        self, options: _ParameterSettings[_Source, _Stack]
    ) -> DynamicBiasConfig | None:
        bias_stack_options = self.__resolve_stack(
            options.bias_generator_stack_options, options.generator_stack_options
        )
        bias_model_config = None
        if options.bias_generator_stack_options.independent_flag:
            bias_model_config = build_generator_stack(
                bias_stack_options, self.__residual_stack
            )
        bias_option = None
        if options.bias_option_flag:
            bias_option = options.bias_option
        bias_sampler_config = None
        if bias_option is MatrixBiasMixtureConfig:
            bias_sampler_config = self.__mixture_sampler(
                options.bias_mixture_num_experts,
                options.bias_mixture_top_k,
                options.bias_mixture_normalize_probabilities_flag,
                options.bias_mixture_router_generator_stack_options,
                bias_stack_options,
            )
        bias_inputs = _BiasInputs(
            num_experts=options.bias_mixture_num_experts,
            top_k=options.bias_mixture_top_k,
            sampler_config=bias_sampler_config,
            decay_schedule=options.bias_decay_schedule,
            decay_rate=options.bias_decay_rate,
            decay_warmup_batches=options.bias_decay_warmup_batches,
            bank_expansion_factor=options.bias_bank_expansion_factor,
            model_config=bias_model_config,
        )
        return bias_inputs.build(bias_option)

    def __diagonal(
        self, options: _ParameterSettings[_Source, _Stack]
    ) -> DynamicDiagonalConfig | None:
        diagonal_model_config = self.__component_generator(
            options.diagonal_generator_stack_options,
            options.generator_stack_options,
        )
        diagonal_option = None
        if options.diagonal_option_flag:
            diagonal_option = options.diagonal_option
        diagonal_inputs = _DiagonalInputs(
            model_config=diagonal_model_config,
        )
        return diagonal_inputs.build(diagonal_option)

    def __mask(
        self, options: _ParameterSettings[_Source, _Stack]
    ) -> AxisMaskConfig | None:
        mask_model_config = self.__component_generator(
            options.mask_generator_stack_options,
            options.generator_stack_options,
        )
        mask_option = None
        if options.mask_option_flag:
            mask_option = options.row_mask_option
        mask_inputs = _MaskInputs(
            mask_threshold=options.mask_threshold,
            mask_surrogate_scale=options.mask_surrogate_scale,
            mask_floor=options.mask_floor,
            mask_dimension_option=options.mask_dimension_option,
            mask_transition_width=options.mask_transition_width,
            model_config=mask_model_config,
        )
        return mask_inputs.build(mask_option)

    def __component_generator(
        self, source: _Source, defaults: _Stack
    ) -> LayerStackConfig | None:
        if not source.independent_flag:
            return None
        stack = self.__resolve_stack(source, defaults)
        return build_generator_stack(stack, self.__residual_stack)

    def __mixture_sampler(
        self,
        num_experts: int | None,
        top_k: int | None,
        normalize: bool | None,
        router_stack: _Source,
        stack_defaults: _Stack,
    ) -> SamplerConfig:
        router = self.__component_generator(router_stack, stack_defaults)
        return build_mixture_sampler(router, num_experts, top_k, normalize)


_WEIGHT_VARIANTS = (
    LayeredWeightedBankDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    MatrixWeightsMixtureConfig,
    SingleModelDynamicWeightConfig,
    DualModelDynamicWeightConfig,
    BroadcastSumDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    DiagonallyModulatedLowRankDynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
)


@dataclass(frozen=True, slots=True)
class _WeightInputs:
    num_experts: int | None
    top_k: int | None
    sampler_config: SamplerConfig | None
    generator_depth: int
    input_factor_source: LowRankFactorSourceOptions | None
    output_factor_source: LowRankFactorSourceOptions | None
    input_factor_model_config: LayerStackConfig | None
    output_factor_model_config: LayerStackConfig | None
    coefficient_model_config: LayerStackConfig | None
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    normalization_option: WeightNormalizationOptions
    normalization_learnable_scale_flag: bool
    normalization_position_option: WeightNormalizationPositionOptions
    bank_expansion_factor: BankExpansionFactorOptions
    model_config: LayerStackConfig | None

    def build(
        self, option: type[DynamicWeightConfig] | None
    ) -> DynamicWeightConfig | None:
        if option is None:
            return None
        if option is not DynamicWeightConfig and option not in _WEIGHT_VARIANTS:
            return _custom_parameter_config(option, self)
        generation = WeightGeneration(
            input_factor_source=self.input_factor_source,
            output_factor_source=self.output_factor_source,
            input_factor_model_config=self.input_factor_model_config,
            output_factor_model_config=self.output_factor_model_config,
            coefficient_model_config=self.coefficient_model_config,
            mixture=MixtureGeneration(
                self.num_experts, self.top_k, self.sampler_config
            ),
        )
        return build_weight_config(
            option,
            generation=generation,
            generator_depth=self.generator_depth,
            decay_schedule=self.decay_schedule,
            decay_rate=self.decay_rate,
            decay_warmup_batches=self.decay_warmup_batches,
            normalization_option=self.normalization_option,
            normalization_learnable_scale_flag=self.normalization_learnable_scale_flag,
            normalization_position_option=self.normalization_position_option,
            bank_expansion_factor=self.bank_expansion_factor,
            model_config=self.model_config,
        )


@dataclass(frozen=True, slots=True)
class _BiasInputs:
    num_experts: int | None
    top_k: int | None
    sampler_config: SamplerConfig | None
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    model_config: LayerStackConfig | None

    def build(self, option: type[DynamicBiasConfig] | None) -> DynamicBiasConfig | None:
        if option is None:
            return None
        if option not in (
            MatrixBiasMixtureConfig,
            WeightedBankDynamicBiasConfig,
            DynamicBiasConfig,
            AffineTransformDynamicBiasConfig,
            AdditiveDynamicBiasConfig,
            MultiplicativeDynamicBiasConfig,
            SigmoidGatedDynamicBiasConfig,
            TanhGatedDynamicBiasConfig,
            GeneratorDynamicBiasConfig,
        ):
            return _custom_parameter_config(option, self)
        generation = MixtureGeneration(
            self.num_experts, self.top_k, self.sampler_config
        )
        return build_bias_config(
            option,
            generation=generation,
            decay_schedule=self.decay_schedule,
            decay_rate=self.decay_rate,
            decay_warmup_batches=self.decay_warmup_batches,
            bank_expansion_factor=self.bank_expansion_factor,
            model_config=self.model_config,
        )


@dataclass(frozen=True, slots=True)
class _DiagonalInputs:
    model_config: LayerStackConfig | None

    def build(
        self, option: type[DynamicDiagonalConfig] | None
    ) -> DynamicDiagonalConfig | None:
        if option is None:
            return None
        if option in (
            DynamicDiagonalConfig,
            StandardDynamicDiagonalConfig,
            AntiDynamicDiagonalConfig,
            CombinedDynamicDiagonalConfig,
        ):
            return option(model_config=self.model_config)
        return _custom_parameter_config(option, self)


@dataclass(frozen=True, slots=True)
class _MaskInputs:
    mask_threshold: float
    mask_surrogate_scale: float
    mask_floor: float
    mask_dimension_option: MaskDimensionOptions
    mask_transition_width: float
    model_config: LayerStackConfig | None

    def build(self, option: type[AxisMaskConfig] | None) -> AxisMaskConfig | None:
        if option is None:
            return None
        if option not in (
            TopSliceAxisMaskConfig,
            DiagonalAxisMaskConfig,
            PerAxisScoreMaskConfig,
            WeightInformedScoreAxisMaskConfig,
            AxisMaskConfig,
            OuterProductMaskConfig,
        ):
            return _custom_parameter_config(option, self)
        build_mask = axis_mask_constructor(option, self)
        return build_mask(
            mask_threshold=self.mask_threshold,
            mask_surrogate_scale=self.mask_surrogate_scale,
            mask_floor=self.mask_floor,
            model_config=self.model_config,
        )


_Parameter = TypeVar("_Parameter", bound=ConfigBase)


def _custom_parameter_config(
    option: type[_Parameter],
    inputs: _WeightInputs | _BiasInputs | _DiagonalInputs | _MaskInputs,
) -> _Parameter:
    if not isinstance(inputs, _WeightInputs) and option in _WEIGHT_VARIANTS:
        raise KeyError("normalization_option")
    accepted = {field.name for field in fields(option)}
    arguments = {
        field.name: getattr(inputs, field.name)
        for field in fields(inputs)
        if field.name in accepted
    }
    return option(**arguments)
