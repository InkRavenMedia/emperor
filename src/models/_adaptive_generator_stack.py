from collections.abc import Callable
from typing import Final, Generic, Protocol, Self, TypeVar

from emperor.augmentations.adaptive_parameters import (
    DiagonallyModulatedLowRankDynamicWeightConfig,
    DynamicWeightConfig,
    LowRankFactorSourceOptions,
    MatrixWeightsMixtureConfig,
)
from emperor.layers import (
    LayerConfig,
    LayerStackConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import SamplerConfig
from models._adaptive_parameters import build_mixture_sampler
from models._residual_stack import (
    ResidualStackSettings,
    StackDefaults,
    build_residual_config,
    inherit_if_unset,
)
from models._stack_options import StackSource as GeneratorStackSource


class StackGenerator(Protocol):
    def build_shared_config(self) -> LayerStackConfig: ...

    def for_weight_source(self, source: GeneratorStackSource | None) -> Self: ...

    def build_config_from_source(
        self, source: GeneratorStackSource | None
    ) -> LayerStackConfig | None: ...


class _GeneratorStackOptions(StackDefaults, Protocol):
    residual_stack_options: Final[ResidualStackSettings | None]


_Options = TypeVar("_Options", bound=_GeneratorStackOptions)


class GeneratorStackFactory(Generic[_Options]):
    options_type: Callable[..., _Options]

    def __init__(self, shared_options: _Options) -> None:
        self.shared_options = shared_options

    def build_shared_config(self) -> LayerStackConfig:
        model_config = build_generator_stack(
            self.shared_options, self.shared_options.residual_stack_options
        )
        return model_config

    def for_weight_source(self, source: GeneratorStackSource | None) -> Self:
        options = self.__resolve_options(source)
        if options is None:
            return self
        return type(self)(options)

    def build_config_from_source(
        self,
        source: GeneratorStackSource | None,
    ) -> LayerStackConfig | None:
        options = self.__resolve_options(source)
        if options is None:
            return None
        model_config = build_generator_stack(options, options.residual_stack_options)
        return model_config

    def __resolve_options(
        self,
        source: GeneratorStackSource | None,
    ) -> _Options | None:
        if source is None or not source.independent_flag:
            return None
        defaults = self.shared_options
        return self.options_type(
            hidden_dim=inherit_if_unset(source.hidden_dim, defaults.hidden_dim),
            layer_norm_position=inherit_if_unset(
                source.layer_norm_position, defaults.layer_norm_position
            ),
            normalization=inherit_if_unset(
                source.normalization, defaults.normalization
            ),
            num_layers=inherit_if_unset(source.num_layers, defaults.num_layers),
            activation=inherit_if_unset(source.activation, defaults.activation),
            residual_connection_option=inherit_if_unset(
                source.residual_connection_option, defaults.residual_connection_option
            ),
            residual_block_size=inherit_if_unset(
                source.residual_block_size, defaults.residual_block_size
            ),
            residual_rms_norm_epsilon=inherit_if_unset(
                source.residual_rms_norm_epsilon, defaults.residual_rms_norm_epsilon
            ),
            residual_model_flag=inherit_if_unset(
                source.residual_model_flag, defaults.residual_model_flag
            ),
            dropout_probability=inherit_if_unset(
                source.dropout_probability, defaults.dropout_probability
            ),
            last_layer_bias_option=inherit_if_unset(
                source.last_layer_bias_option, defaults.last_layer_bias_option
            ),
            apply_output_postprocessing_flag=inherit_if_unset(
                source.apply_output_postprocessing_flag,
                defaults.apply_output_postprocessing_flag,
            ),
            bias_flag=inherit_if_unset(source.bias_flag, defaults.bias_flag),
            residual_stack_options=defaults.residual_stack_options,
        )


def build_generator_stack(
    options: StackDefaults,
    residual_stack: ResidualStackSettings | None,
) -> LayerStackConfig:
    residual_config = build_residual_config(
        options.residual_connection_option,
        options.residual_model_flag,
        residual_stack,
        residual_block_size=options.residual_block_size,
        residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
    )
    return build_generator_layers(options, residual_config)


def build_generator_layers(
    options: StackDefaults, residual_config: ResidualConfig | None
) -> LayerStackConfig:
    layer_model_config = LinearLayerConfig(
        bias_flag=options.bias_flag,
    )
    layer_config = LayerConfig(
        activation=options.activation,
        layer_norm_position=options.layer_norm_position,
        normalization=options.normalization,
        residual_config=residual_config,
        dropout_probability=options.dropout_probability,
        gate_config=None,
        halting_config=None,
        layer_model_config=layer_model_config,
    )
    return LayerStackConfig(
        hidden_dim=options.hidden_dim,
        num_layers=options.num_layers,
        last_layer_bias_option=options.last_layer_bias_option,
        apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
        layer_config=layer_config,
    )


_Source = TypeVar("_Source")
_SourceCo = TypeVar("_SourceCo", covariant=True)
_Generation = TypeVar("_Generation")
_GenerationCo = TypeVar("_GenerationCo", covariant=True)


class _FactorSources(Protocol[_SourceCo]):
    input_factor_source: Final[LowRankFactorSourceOptions | None]
    output_factor_source: Final[LowRankFactorSourceOptions | None]
    input_factor_generator_stack: Final[_SourceCo | None]
    output_factor_generator_stack: Final[_SourceCo | None]
    coefficient_generator_stack: Final[_SourceCo | None]


class _BuildFactors(Protocol[_GenerationCo]):
    def __call__(
        self,
        *,
        input_factor_source: LowRankFactorSourceOptions | None,
        output_factor_source: LowRankFactorSourceOptions | None,
        input_factor_model_config: LayerStackConfig | None,
        output_factor_model_config: LayerStackConfig | None,
        coefficient_model_config: LayerStackConfig | None,
    ) -> _GenerationCo: ...


def build_factor_generation(
    options: _FactorSources[_Source],
    build_stack: Callable[[_Source | None], LayerStackConfig | None],
    result_type: _BuildFactors[_Generation],
) -> _Generation:
    input_factor = build_stack(options.input_factor_generator_stack)
    output_factor = build_stack(options.output_factor_generator_stack)
    coefficient = build_stack(options.coefficient_generator_stack)
    return result_type(
        input_factor_source=options.input_factor_source,
        output_factor_source=options.output_factor_source,
        input_factor_model_config=input_factor,
        output_factor_model_config=output_factor,
        coefficient_model_config=coefficient,
    )


class MixtureGenerationSettings(Protocol[_SourceCo]):
    mixture_num_experts: Final[int | None]
    mixture_top_k: Final[int | None]
    mixture_normalize_probabilities_flag: Final[bool | None]
    mixture_router_generator_stack: Final[_SourceCo | None]


class _BuildMixture(Protocol[_GenerationCo]):
    def __call__(
        self,
        *,
        num_experts: int | None,
        top_k: int | None,
        sampler_config: SamplerConfig | None,
    ) -> _GenerationCo: ...


def build_mixture_generation(
    options: MixtureGenerationSettings[_Source],
    build_stack: Callable[[_Source | None], LayerStackConfig | None],
    result_type: _BuildMixture[_Generation],
) -> _Generation:
    router = build_stack(options.mixture_router_generator_stack)
    sampler = build_mixture_sampler(
        router,
        options.mixture_num_experts,
        options.mixture_top_k,
        options.mixture_normalize_probabilities_flag,
    )
    return result_type(
        num_experts=options.mixture_num_experts,
        top_k=options.mixture_top_k,
        sampler_config=sampler,
    )


class WeightGenerationSettings(
    _FactorSources[GeneratorStackSource],
    MixtureGenerationSettings[GeneratorStackSource],
    Protocol,
):
    pass


_Mixture = TypeVar("_Mixture")
_MixtureIn = TypeVar("_MixtureIn", contravariant=True)
_Default = TypeVar("_Default")


class _BuildWeight(Protocol[_MixtureIn, _GenerationCo]):
    def __call__(
        self,
        *,
        input_factor_source: LowRankFactorSourceOptions | None = ...,
        output_factor_source: LowRankFactorSourceOptions | None = ...,
        input_factor_model_config: LayerStackConfig | None = ...,
        output_factor_model_config: LayerStackConfig | None = ...,
        coefficient_model_config: LayerStackConfig | None = ...,
        mixture: _MixtureIn = ...,
    ) -> _GenerationCo: ...


def build_weight_generation(
    options: WeightGenerationSettings,
    factory: StackGenerator,
    weight_option: type[DynamicWeightConfig] | None,
    weight_source: GeneratorStackSource | None,
    *,
    weight_type: _BuildWeight[_Mixture, _Generation],
    mixture_type: _BuildMixture[_Mixture],
    default: Callable[[], _Default],
) -> _Generation | _Default:
    if weight_option is MatrixWeightsMixtureConfig:
        mixture = build_mixture_generation(
            options, factory.build_config_from_source, mixture_type
        )
        return weight_type(mixture=mixture)
    if weight_option is DiagonallyModulatedLowRankDynamicWeightConfig:
        factor_factory = factory.for_weight_source(weight_source)
        return build_factor_generation(
            options, factor_factory.build_config_from_source, weight_type
        )
    return default()
