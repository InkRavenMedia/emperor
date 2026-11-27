from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Final, Protocol

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
    AxisMaskConfig,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    LowRankFactorSourceOptions,
    MatrixWeightsMixtureConfig,
)
from emperor.layers import LayerStackConfig
from models import _learned_grouping
from models._adaptive_linear_template import AdaptiveLinearLayerTemplateConfig
from models._adaptive_parameters import (
    MixtureGeneration,
    WeightGeneration,
    build_bias_config,
    build_mixture_sampler,
    build_scoped_mask,
    build_weight_config,
)

from ._adaptive_options import (
    AdaptiveSettings,
    GeneratorStackSource,
    WeightOptions,
    adaptive_options,
    bias_mixture_router_generator_stack_source,
    weight_coefficient_generator_stack_source,
    weight_input_factor_generator_stack_source,
    weight_mixture_router_generator_stack_source,
    weight_output_factor_generator_stack_source,
)
from ._control_options import StackOptions


class _Settings(AdaptiveSettings, Protocol):
    weight_input_factor_source: Final[LowRankFactorSourceOptions | None]
    weight_output_factor_source: Final[LowRankFactorSourceOptions | None]
    weight_mixture_num_experts: Final[int | None]
    bias_mixture_num_experts: Final[int | None]
    bias_mixture_top_k: Final[int | None]
    bias_mixture_normalize_probabilities_flag: Final[bool | None]


class AdaptiveLayerFactory:
    def __init__(
        self,
        runtime: _Settings,
        build_generator: Callable[[StackOptions], LayerStackConfig],
        *,
        template_type: type[AdaptiveLinearLayerTemplateConfig],
    ) -> None:
        self._runtime = runtime
        self._options = adaptive_options(runtime)
        self._build_generator = build_generator
        self._template_type = template_type

    def build(
        self, *, bias_flag: bool, grouping_prefix: str = ""
    ) -> AdaptiveLinearLayerTemplateConfig:
        bias = self.__bias()
        grouping = _learned_grouping.grouping_from_runtime(
            self._runtime, grouping_prefix
        )
        diagonal = self.__diagonal()
        weight = self.__weight()
        mask = self.__mask()
        generator = self._build_generator(self._options.generator_stack)
        augmentation = AdaptiveParameterAugmentationConfig(
            grouping_config=grouping,
            diagonal_config=diagonal,
            weight_config=weight,
            bias_config=bias,
            mask_config=mask,
            model_config=generator,
        )
        return self._template_type(
            bias_flag=bias_flag or bias is not None,
            adaptive_augmentation_config=augmentation,
        )

    def __independent_generator(
        self, source: GeneratorStackSource, defaults: StackOptions
    ) -> LayerStackConfig | None:
        options = source.resolve(defaults)
        if options is None:
            return None
        return self._build_generator(options)

    def __weight(self) -> DynamicWeightConfig | None:
        options = self._options.weight
        if not options.enabled:
            return None
        implementation = options.implementation
        if implementation is None:
            raise ValueError(
                "weight_option must be set when weight_option_flag is True"
            )
        generator = self.__independent_generator(
            options.generator_stack, self._options.generator_stack
        )
        return build_weight_config(
            implementation,
            build_generation=partial(self.__weight_generation, implementation, options),
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

    def __weight_generation(
        self, implementation: type[DynamicWeightConfig], options: WeightOptions
    ) -> WeightGeneration:
        runtime = self._runtime
        if implementation is MatrixWeightsMixtureConfig:
            router_source = weight_mixture_router_generator_stack_source(runtime)
            return WeightGeneration(
                num_experts=runtime.weight_mixture_num_experts,
                router_model_config=self.__independent_generator(
                    router_source, self._options.generator_stack
                ),
            )
        defaults = (
            options.generator_stack.resolve(self._options.generator_stack)
            or self._options.generator_stack
        )
        input_factor = self.__independent_generator(
            weight_input_factor_generator_stack_source(runtime), defaults
        )
        output_factor = self.__independent_generator(
            weight_output_factor_generator_stack_source(runtime), defaults
        )
        coefficient = self.__independent_generator(
            weight_coefficient_generator_stack_source(runtime), defaults
        )
        return WeightGeneration(
            input_factor_source=runtime.weight_input_factor_source,
            output_factor_source=runtime.weight_output_factor_source,
            input_factor_model_config=input_factor,
            output_factor_model_config=output_factor,
            coefficient_model_config=coefficient,
        )

    def __bias(self) -> DynamicBiasConfig | None:
        options = self._options.bias
        if not options.enabled:
            return None
        implementation = options.implementation
        if implementation is None:
            raise ValueError("bias_option must be set when bias_option_flag is True")
        generator = self.__independent_generator(
            options.generator_stack, self._options.generator_stack
        )
        return build_bias_config(
            implementation,
            build_generation=self.__bias_mixture,
            decay_schedule=options.decay_schedule,
            decay_rate=options.decay_rate,
            decay_warmup_batches=options.decay_warmup_batches,
            bank_expansion_factor=options.bank_expansion_factor,
            model_config=generator,
        )

    def __bias_mixture(self) -> MixtureGeneration:
        runtime = self._runtime
        router_source = bias_mixture_router_generator_stack_source(runtime)
        return self.__mixture(
            router_source,
            runtime.bias_mixture_num_experts,
            runtime.bias_mixture_top_k,
            runtime.bias_mixture_normalize_probabilities_flag,
        )

    def __mixture(
        self,
        source: GeneratorStackSource,
        num_experts: int | None,
        top_k: int | None,
        normalize: bool | None,
    ) -> MixtureGeneration:
        router = self.__independent_generator(source, self._options.generator_stack)
        sampler = build_mixture_sampler(router, num_experts, top_k, normalize)
        return MixtureGeneration(
            num_experts=sampler.num_experts, top_k=sampler.top_k, sampler_config=sampler
        )

    def __diagonal(self) -> DynamicDiagonalConfig | None:
        options = self._options.diagonal
        if not options.enabled:
            return None
        implementation = options.implementation
        if implementation is None:
            raise ValueError(
                "diagonal_option must be set when diagonal_option_flag is True"
            )
        generator = self.__independent_generator(
            options.generator_stack, self._options.generator_stack
        )
        return implementation(model_config=generator)

    def __mask(self) -> AxisMaskConfig | None:
        options = self._options.mask
        if not options.enabled:
            return None
        implementation = options.implementation
        if implementation is None:
            raise ValueError(
                "row_mask_option must be set when mask_option_flag is True"
            )
        generator = self.__independent_generator(
            options.generator_stack, self._options.generator_stack
        )
        return build_scoped_mask(implementation, options, model_config=generator)
