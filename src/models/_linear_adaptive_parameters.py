from __future__ import annotations

from functools import partial
from typing import Final, Protocol

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DiagonallyModulatedLowRankDynamicWeightConfig,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GroupingConfig,
    LowRankFactorSourceOptions,
    MaskDimensionOptions,
    MatrixWeightsMixtureConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.layers import LayerStackConfig
from models._adaptive_generator_stack import build_generator_stack
from models._adaptive_parameters import (
    MixtureGeneration,
    WeightGeneration,
    build_bias_config,
    build_diagonal_config,
    build_mixture_sampler,
    build_scoped_mask,
    build_weight_config,
)
from models._residual_stack import ResidualStackSettings, StackDefaults


class GeneratorStackOptions(Protocol):
    independent: Final[bool]
    stack: Final[StackDefaults]


class AdaptiveGenerationOptions(Protocol):
    weight_input_factor_source: Final[LowRankFactorSourceOptions | None]
    weight_output_factor_source: Final[LowRankFactorSourceOptions | None]
    weight_mixture_num_experts: Final[int | None]
    bias_mixture_num_experts: Final[int | None]
    bias_mixture_top_k: Final[int | None]
    bias_mixture_normalize_probabilities_flag: Final[bool | None]
    weight_input_factor_generator_stack: Final[GeneratorStackOptions | None]
    weight_output_factor_generator_stack: Final[GeneratorStackOptions | None]
    weight_coefficient_generator_stack: Final[GeneratorStackOptions | None]
    weight_mixture_router_generator_stack: Final[GeneratorStackOptions | None]
    bias_mixture_router_generator_stack: Final[GeneratorStackOptions | None]


class AdaptiveWeightOptions(Protocol):
    enabled: Final[bool]
    option: Final[type[DynamicWeightConfig] | None]
    generator_depth: Final[int]
    normalization_option: Final[WeightNormalizationOptions]
    normalization_position_option: Final[WeightNormalizationPositionOptions]
    decay_schedule: Final[WeightDecayScheduleOptions]
    decay_rate: Final[float]
    decay_warmup_batches: Final[int]
    bank_expansion_factor: Final[BankExpansionFactorOptions]
    generator_stack: Final[GeneratorStackOptions]
    normalization_learnable_scale_flag: Final[bool]


class AdaptiveBiasOptions(Protocol):
    enabled: Final[bool]
    option: Final[type[DynamicBiasConfig] | None]
    decay_schedule: Final[WeightDecayScheduleOptions]
    decay_rate: Final[float]
    decay_warmup_batches: Final[int]
    bank_expansion_factor: Final[BankExpansionFactorOptions]
    generator_stack: Final[GeneratorStackOptions]


class AdaptiveDiagonalOptions(Protocol):
    enabled: Final[bool]
    option: Final[type[DynamicDiagonalConfig] | None]
    generator_stack: Final[GeneratorStackOptions]


class ProjectionMaskOptions(Protocol):
    row_mask_option: Final[type[AxisMaskConfig] | None]
    dimension_option: Final[MaskDimensionOptions]
    threshold: Final[float]
    surrogate_scale: Final[float]
    floor: Final[float]
    transition_width: Final[float]


class AdaptiveMaskOptions(ProjectionMaskOptions, Protocol):
    enabled: Final[bool]
    generator_stack: Final[GeneratorStackOptions]


class AdaptiveProjectionOptions(Protocol):
    grouping_config: Final[GroupingConfig | None]
    generation: Final[AdaptiveGenerationOptions]
    weight_option: Final[type[DynamicWeightConfig] | None]
    generator_depth: Final[int]
    weight_decay_schedule: Final[WeightDecayScheduleOptions]
    weight_decay_rate: Final[float]
    weight_decay_warmup_batches: Final[int]
    weight_normalization_option: Final[WeightNormalizationOptions]
    weight_normalization_position_option: Final[WeightNormalizationPositionOptions]
    weight_bank_expansion_factor: Final[BankExpansionFactorOptions]
    bias_option: Final[type[DynamicBiasConfig] | None]
    bias_decay_schedule: Final[WeightDecayScheduleOptions]
    bias_decay_rate: Final[float]
    bias_decay_warmup_batches: Final[int]
    bias_bank_expansion_factor: Final[BankExpansionFactorOptions]
    diagonal_option: Final[type[DynamicDiagonalConfig] | None]
    weight_normalization_learnable_scale_flag: Final[bool]

    @property
    def mask(self) -> ProjectionMaskOptions: ...


class _LayerSettings(Protocol):
    grouping_config: Final[GroupingConfig | None]
    adaptive_generator_stack: Final[StackDefaults]
    residual_stack: Final[ResidualStackSettings]
    generation: Final[AdaptiveGenerationOptions]
    weight: Final[AdaptiveWeightOptions]
    bias: Final[AdaptiveBiasOptions]
    diagonal: Final[AdaptiveDiagonalOptions]
    mask: Final[AdaptiveMaskOptions]


class AdaptiveParameterConfigFactory:
    def __init__(self, runtime: _LayerSettings) -> None:
        self._build_settings = runtime

    def build_hidden_config(self) -> AdaptiveParameterAugmentationConfig:
        weight_config = self.__build_hidden_weight()
        bias_config = self.__build_hidden_bias()
        diagonal_config = self.__build_hidden_diagonal()
        mask_config = self.__build_hidden_mask()
        generator_config = build_generator_stack(
            self._build_settings.adaptive_generator_stack,
            self._build_settings.residual_stack,
        )
        return AdaptiveParameterAugmentationConfig(
            grouping_config=self._build_settings.grouping_config,
            weight_config=weight_config,
            bias_config=bias_config,
            diagonal_config=diagonal_config,
            mask_config=mask_config,
            model_config=generator_config,
        )

    def build_projection_config(
        self,
        options: AdaptiveProjectionOptions,
    ) -> AdaptiveParameterAugmentationConfig:
        weight_option = options.weight_option
        weight_config = build_weight_config(
            weight_option,
            build_generation=partial(
                self.__weight_generation, weight_option, options.generation
            ),
            generator_depth=options.generator_depth,
            decay_schedule=options.weight_decay_schedule,
            decay_rate=options.weight_decay_rate,
            decay_warmup_batches=options.weight_decay_warmup_batches,
            normalization_option=options.weight_normalization_option,
            normalization_learnable_scale_flag=options.weight_normalization_learnable_scale_flag,
            normalization_position_option=(
                options.weight_normalization_position_option
            ),
            bank_expansion_factor=options.weight_bank_expansion_factor,
        )
        bias_config = build_bias_config(
            options.bias_option,
            build_generation=partial(self.__bias_mixture, options.generation),
            decay_schedule=options.bias_decay_schedule,
            decay_rate=options.bias_decay_rate,
            decay_warmup_batches=options.bias_decay_warmup_batches,
            bank_expansion_factor=options.bias_bank_expansion_factor,
        )
        diagonal_config = build_diagonal_config(options.diagonal_option)
        mask_config = self.__mask_config(options.mask)
        generator_config = build_generator_stack(
            self._build_settings.adaptive_generator_stack,
            self._build_settings.residual_stack,
        )
        return AdaptiveParameterAugmentationConfig(
            grouping_config=options.grouping_config,
            weight_config=weight_config,
            bias_config=bias_config,
            diagonal_config=diagonal_config,
            mask_config=mask_config,
            model_config=generator_config,
        )

    def __build_hidden_weight(self) -> DynamicWeightConfig | None:
        options = self._build_settings.weight
        if not options.enabled:
            return None
        generator = self.__build_independent_generator_stack(options.generator_stack)
        weight_option = options.option
        return build_weight_config(
            weight_option,
            build_generation=partial(
                self.__weight_generation,
                weight_option,
                self._build_settings.generation,
            ),
            generator_depth=options.generator_depth,
            decay_schedule=options.decay_schedule,
            decay_rate=options.decay_rate,
            decay_warmup_batches=options.decay_warmup_batches,
            normalization_option=options.normalization_option,
            normalization_learnable_scale_flag=options.normalization_learnable_scale_flag,
            normalization_position_option=options.normalization_position_option,
            bank_expansion_factor=options.bank_expansion_factor,
            model_config=generator,
        )

    def __build_hidden_bias(self) -> DynamicBiasConfig | None:
        options = self._build_settings.bias
        if not options.enabled:
            return None
        generator = self.__build_independent_generator_stack(options.generator_stack)
        return build_bias_config(
            options.option,
            build_generation=partial(
                self.__bias_mixture, self._build_settings.generation
            ),
            decay_schedule=options.decay_schedule,
            decay_rate=options.decay_rate,
            decay_warmup_batches=options.decay_warmup_batches,
            bank_expansion_factor=options.bank_expansion_factor,
            model_config=generator,
        )

    def __build_hidden_diagonal(self) -> DynamicDiagonalConfig | None:
        options = self._build_settings.diagonal
        if not options.enabled:
            return None
        generator = self.__build_independent_generator_stack(options.generator_stack)
        return build_diagonal_config(options.option, model_config=generator)

    def __build_hidden_mask(self) -> AxisMaskConfig | None:
        options = self._build_settings.mask
        if not options.enabled:
            return None
        generator = self.__build_independent_generator_stack(options.generator_stack)
        return self.__mask_config(options, model_config=generator)

    def __build_independent_generator_stack(
        self,
        options: GeneratorStackOptions | None,
    ) -> LayerStackConfig | None:
        if options is None or not options.independent:
            return None
        independent_generator_stack_config = build_generator_stack(
            options.stack, self._build_settings.residual_stack
        )
        return independent_generator_stack_config

    def __weight_generation(
        self,
        option: type[DynamicWeightConfig] | None,
        options: AdaptiveGenerationOptions,
    ) -> WeightGeneration | None:
        if option is MatrixWeightsMixtureConfig:
            return WeightGeneration(
                num_experts=options.weight_mixture_num_experts,
                router_model_config=self.__build_independent_generator_stack(
                    options.weight_mixture_router_generator_stack
                ),
            )
        if option is not DiagonallyModulatedLowRankDynamicWeightConfig:
            return None
        input_factor = self.__build_independent_generator_stack(
            options.weight_input_factor_generator_stack
        )
        output_factor = self.__build_independent_generator_stack(
            options.weight_output_factor_generator_stack
        )
        coefficients = self.__build_independent_generator_stack(
            options.weight_coefficient_generator_stack
        )
        return WeightGeneration(
            input_factor_source=options.weight_input_factor_source,
            output_factor_source=options.weight_output_factor_source,
            input_factor_model_config=input_factor,
            output_factor_model_config=output_factor,
            coefficient_model_config=coefficients,
        )

    def __bias_mixture(self, options: AdaptiveGenerationOptions) -> MixtureGeneration:
        return self.__mixture(
            options.bias_mixture_num_experts,
            options.bias_mixture_top_k,
            options.bias_mixture_normalize_probabilities_flag,
            options.bias_mixture_router_generator_stack,
        )

    def __mixture(
        self,
        num_experts: int | None,
        top_k: int | None,
        normalize_probabilities: bool | None,
        router_stack: GeneratorStackOptions | None,
    ) -> MixtureGeneration:
        router_model = self.__build_independent_generator_stack(router_stack)
        sampler = build_mixture_sampler(
            router_model, num_experts, top_k, normalize_probabilities
        )
        return MixtureGeneration(
            num_experts=sampler.num_experts, top_k=sampler.top_k, sampler_config=sampler
        )

    @staticmethod
    def __mask_config(
        options: ProjectionMaskOptions,
        *,
        model_config: LayerStackConfig | None = None,
    ) -> AxisMaskConfig | None:
        option = options.row_mask_option
        if option is None:
            return None
        return build_scoped_mask(option, options, model_config=model_config)
