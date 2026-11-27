from collections.abc import Callable
from typing import Final, Protocol

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GroupingConfig,
    MatrixBiasMixtureConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from models._adaptive_generator_stack import (
    GeneratorStackSource,
    MixtureGenerationSettings,
    StackGenerator,
    WeightGenerationSettings,
    build_mixture_generation,
    build_weight_generation,
)
from models._adaptive_parameters import (
    MaskOptions as MaskSettings,
)
from models._adaptive_parameters import (
    MixtureGeneration,
    WeightGeneration,
    build_bias_config,
    build_diagonal_config,
    build_mask_config,
    build_weight_config,
    resolve_enabled_adaptive_parameter_option,
)


class _AdaptiveParameter(Protocol):
    option_flag: Final[bool]
    generator_stack_source: Final[GeneratorStackSource]


class _DecaySettings(_AdaptiveParameter, Protocol):
    decay_schedule: Final[WeightDecayScheduleOptions]
    decay_rate: Final[float]
    decay_warmup_batches: Final[int]
    bank_expansion_factor: Final[BankExpansionFactorOptions]


class WeightOptions(_DecaySettings, Protocol):
    option: Final[type[DynamicWeightConfig] | None]
    generation: Final[WeightGenerationSettings]
    generator_depth: Final[int]
    normalization_option: Final[WeightNormalizationOptions]
    normalization_position_option: Final[WeightNormalizationPositionOptions]
    normalization_learnable_scale_flag: Final[bool]


class BiasOptions(_DecaySettings, Protocol):
    option: Final[type[DynamicBiasConfig] | None]
    generation: Final[MixtureGenerationSettings[GeneratorStackSource]]


class DiagonalOptions(_AdaptiveParameter, Protocol):
    option: Final[type[DynamicDiagonalConfig] | None]


class MaskOptions(_AdaptiveParameter, MaskSettings, Protocol):
    pass


class AdaptiveAugmentationFactory:
    def __init__(
        self,
        generators: StackGenerator,
        *,
        default_weight_generation: Callable[
            [], WeightGeneration | None
        ] = WeightGeneration,
    ) -> None:
        self.generators = generators
        self.default_weight_generation = default_weight_generation

    def build(
        self,
        *,
        grouping: GroupingConfig | None,
        weight: WeightOptions,
        bias: BiasOptions,
        diagonal: DiagonalOptions,
        mask: MaskOptions,
    ) -> AdaptiveParameterAugmentationConfig:
        weight_config = self.__build_weight_config(weight)
        bias_config = self.__build_bias_config(bias)
        diagonal_config = self.__build_diagonal_config(diagonal)
        mask_config = self.__build_mask_config(mask)
        model_config = self.generators.build_shared_config()
        return AdaptiveParameterAugmentationConfig(
            grouping_config=grouping,
            weight_config=weight_config,
            bias_config=bias_config,
            diagonal_config=diagonal_config,
            mask_config=mask_config,
            model_config=model_config,
        )

    def __build_weight_config(
        self, options: WeightOptions
    ) -> DynamicWeightConfig | None:
        weight_option = resolve_enabled_adaptive_parameter_option(
            option_flag=options.option_flag,
            option=options.option,
            option_flag_name="weight_option_flag",
            option_name="weight_option",
        )
        if weight_option is None:
            return None
        model_config = self.generators.build_config_from_source(
            options.generator_stack_source
        )
        weight_generation = build_weight_generation(
            options.generation,
            self.generators,
            weight_option,
            options.generator_stack_source,
            weight_type=WeightGeneration,
            default=self.default_weight_generation,
        )
        weight_config = build_weight_config(
            weight_option,
            generation=weight_generation,
            generator_depth=options.generator_depth,
            decay_schedule=options.decay_schedule,
            decay_rate=options.decay_rate,
            decay_warmup_batches=(options.decay_warmup_batches),
            normalization_option=(options.normalization_option),
            normalization_learnable_scale_flag=options.normalization_learnable_scale_flag,
            normalization_position_option=(options.normalization_position_option),
            bank_expansion_factor=(options.bank_expansion_factor),
            model_config=model_config,
        )
        return weight_config

    def __build_bias_config(self, options: BiasOptions) -> DynamicBiasConfig | None:
        bias_option = resolve_enabled_adaptive_parameter_option(
            option_flag=options.option_flag,
            option=options.option,
            option_flag_name="bias_option_flag",
            option_name="bias_option",
        )
        if bias_option is None:
            return None
        model_config = self.generators.build_config_from_source(
            options.generator_stack_source
        )
        bias_generation = None
        if bias_option is MatrixBiasMixtureConfig:
            bias_generation = build_mixture_generation(
                options.generation,
                self.generators.build_config_from_source,
                MixtureGeneration,
            )
        bias_config = build_bias_config(
            bias_option,
            generation=bias_generation,
            decay_schedule=options.decay_schedule,
            decay_rate=options.decay_rate,
            decay_warmup_batches=(options.decay_warmup_batches),
            bank_expansion_factor=(options.bank_expansion_factor),
            model_config=model_config,
        )
        return bias_config

    def __build_diagonal_config(
        self, options: DiagonalOptions
    ) -> DynamicDiagonalConfig | None:
        diagonal_option = resolve_enabled_adaptive_parameter_option(
            option_flag=options.option_flag,
            option=options.option,
            option_flag_name="diagonal_option_flag",
            option_name="diagonal_option",
        )
        if diagonal_option is None:
            return None
        model_config = self.generators.build_config_from_source(
            options.generator_stack_source
        )
        diagonal_config = build_diagonal_config(
            diagonal_option,
            model_config=model_config,
        )
        return diagonal_config

    def __build_mask_config(self, options: MaskOptions) -> AxisMaskConfig | None:
        row_mask_option = resolve_enabled_adaptive_parameter_option(
            option_flag=options.option_flag,
            option=options.row_mask_option,
            option_flag_name="mask_option_flag",
            option_name="row_mask_option",
        )
        if row_mask_option is None:
            return None
        model_config = self.generators.build_config_from_source(
            options.generator_stack_source
        )
        mask_config = build_mask_config(options, model_config=model_config)
        return mask_config
