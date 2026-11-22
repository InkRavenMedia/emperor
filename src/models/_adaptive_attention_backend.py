from abc import ABC
from collections.abc import Callable
from types import ModuleType
from typing import Final, Generic, Protocol, TypeVar

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    GroupingConfig,
)
from emperor.linears import LinearLayerConfig
from models._adaptive_augmentation import (
    BiasOptions,
    DiagonalOptions,
    MaskOptions,
    WeightOptions,
)
from models._linear_stack import LinearStackFactory


class AdaptiveBackendStackFactory(LinearStackFactory, ABC):
    layer_template_type: type[AdaptiveLinearLayerConfig]
    adaptive_augmentation_config: AdaptiveParameterAugmentationConfig | None
    enable_bias_for_adaptive_bias = False

    def build_backend_linear_layer_config(
        self, *, bias_flag: bool
    ) -> LinearLayerConfig:
        augmentation = self.adaptive_augmentation_config
        if augmentation is None:
            return self.build_plain_linear_layer_config(bias_flag=bias_flag)
        if self.enable_bias_for_adaptive_bias:
            bias_flag = bias_flag or augmentation.bias_config is not None
        return self.layer_template_type(
            bias_flag=bias_flag, adaptive_augmentation_config=augmentation
        )


_GeneratorCo = TypeVar("_GeneratorCo", covariant=True)
_Generator = TypeVar("_Generator")
_HiddenRecord = TypeVar("_HiddenRecord")
_Option = TypeVar("_Option")


class AdaptiveOptions(Protocol[_GeneratorCo]):
    hidden_dim: Final[int]
    output_dim: Final[int]
    grouping_config: Final[GroupingConfig | None]
    hidden_adaptive_weight_options: Final[WeightOptions | None]
    hidden_adaptive_bias_options: Final[BiasOptions | None]
    hidden_adaptive_diagonal_options: Final[DiagonalOptions | None]
    hidden_adaptive_mask_options: Final[MaskOptions | None]
    adaptive_generator_stack_options: Final[_GeneratorCo | None]


class AdaptiveOptionDefaults(Protocol[_GeneratorCo]):
    def hidden_adaptive_weight_options(
        self, config: ModuleType, /
    ) -> WeightOptions: ...
    def hidden_adaptive_bias_options(self, config: ModuleType, /) -> BiasOptions: ...
    def hidden_adaptive_diagonal_options(
        self, config: ModuleType, /
    ) -> DiagonalOptions: ...
    def hidden_adaptive_mask_options(self, config: ModuleType, /) -> MaskOptions: ...
    def adaptive_generator_stack_options(
        self, config: ModuleType, /
    ) -> _GeneratorCo: ...


class _HiddenFactory(Protocol):
    @property
    def adaptive_augmentation_config(self) -> AdaptiveParameterAugmentationConfig: ...


class AttentionAugmentationFactory(Generic[_Generator, _HiddenRecord]):
    hidden_options_type: Callable[..., _HiddenRecord]
    hidden_factory_type: Callable[[_HiddenRecord], _HiddenFactory]
    use_defaults_for_falsey_options = False

    def __init__(
        self,
        dependencies: AdaptiveOptions[_Generator],
        config: ModuleType,
        defaults: AdaptiveOptionDefaults[_Generator],
    ) -> None:
        self.dependencies = dependencies
        self.adaptive_generator_stack_options = self.__select(
            dependencies.adaptive_generator_stack_options,
            defaults.adaptive_generator_stack_options,
            config,
        )
        self.hidden_adaptive_weight_options = self.__select(
            dependencies.hidden_adaptive_weight_options,
            defaults.hidden_adaptive_weight_options,
            config,
        )
        self.grouping_config = dependencies.grouping_config
        self.hidden_adaptive_bias_options = self.__select(
            dependencies.hidden_adaptive_bias_options,
            defaults.hidden_adaptive_bias_options,
            config,
        )
        self.hidden_adaptive_diagonal_options = self.__select(
            dependencies.hidden_adaptive_diagonal_options,
            defaults.hidden_adaptive_diagonal_options,
            config,
        )
        self.hidden_adaptive_mask_options = self.__select(
            dependencies.hidden_adaptive_mask_options,
            defaults.hidden_adaptive_mask_options,
            config,
        )

    def __select(
        self,
        selected: _Option | None,
        default: Callable[[ModuleType], _Option],
        config: ModuleType,
    ) -> _Option:
        if selected is None or (self.use_defaults_for_falsey_options and not selected):
            return default(config)
        return selected

    def build_adaptive_augmentation_config(self) -> AdaptiveParameterAugmentationConfig:
        options = self.hidden_options_type(
            hidden_dim=self.dependencies.hidden_dim,
            stack_options=None,
            submodule_stack_options=None,
            layer_controller_options=None,
            dynamic_memory_options=None,
            recurrent_controller_options=None,
            grouping_config=self.grouping_config,
            hidden_adaptive_weight_options=self.hidden_adaptive_weight_options,
            hidden_adaptive_bias_options=self.hidden_adaptive_bias_options,
            hidden_adaptive_diagonal_options=self.hidden_adaptive_diagonal_options,
            hidden_adaptive_mask_options=self.hidden_adaptive_mask_options,
            adaptive_generator_stack_options=self.adaptive_generator_stack_options,
            output_dim=self.dependencies.output_dim,
        )
        hidden_factory = self.hidden_factory_type(options)
        return hidden_factory.adaptive_augmentation_config
