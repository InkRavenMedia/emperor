from collections.abc import Callable
from types import ModuleType
from typing import Any, Final, Generic, Protocol, TypeVar

from emperor.augmentations.adaptive_parameters import GroupingConfig
from emperor.layers import LayerStackConfig
from models._adaptive_augmentation import (
    AdaptiveAugmentationFactory,
    BiasOptions,
    DiagonalOptions,
    MaskOptions,
    WeightOptions,
)
from models._adaptive_generator_stack import StackGenerator
from models._controller_stack import build_controller_stack_config
from models._expert_mixture import ExpertSettings, ExpertStackFactory
from models._expert_router import RoutedExpertSettings, RoutedExpertStackFactory
from models._linear_stack import LayerStackSettings


class _AdaptiveInputs(RoutedExpertSettings, Protocol):
    hidden_dim: Final[int]
    adaptive_generator_stack_options: Final[object]
    grouping_config: Final[GroupingConfig | None]
    router_grouping_config: Final[GroupingConfig | None]
    hidden_adaptive_weight_options: Final[WeightOptions]
    router_adaptive_weight_options: Final[WeightOptions]
    hidden_adaptive_bias_options: Final[BiasOptions]
    router_adaptive_bias_options: Final[BiasOptions]
    hidden_adaptive_diagonal_options: Final[DiagonalOptions]
    router_adaptive_diagonal_options: Final[DiagonalOptions]
    hidden_adaptive_mask_options: Final[MaskOptions]
    router_adaptive_mask_options: Final[MaskOptions]


_Adaptive = TypeVar("_Adaptive", bound=_AdaptiveInputs)


class AdaptiveRoutedExpertFactory(RoutedExpertStackFactory, Generic[_Adaptive]):
    _generator: Callable[[Any], StackGenerator]

    def __init__(self, dependencies: _Adaptive) -> None:
        self.stack_options = dependencies.stack_options
        self.submodule_stack_options = dependencies.submodule_stack_options
        self.mixture_options = dependencies.mixture_options
        self.expert_stack_options = dependencies.expert_stack_options
        self.sampler_options = dependencies.sampler_options
        self.router_options = dependencies.router_options
        self.router_stack_options = dependencies.router_stack_options
        self.router_layer_controller_options = (
            dependencies.router_layer_controller_options
        )
        self.router_dynamic_memory_options = dependencies.router_dynamic_memory_options
        self.router_recurrent_controller_options = (
            dependencies.router_recurrent_controller_options
        )
        self.layer_controller_options = dependencies.layer_controller_options
        self.dynamic_memory_options = dependencies.dynamic_memory_options
        self.recurrent_controller_options = dependencies.recurrent_controller_options
        self.expert_layer_controller_options = (
            dependencies.expert_layer_controller_options
        )
        self.expert_dynamic_memory_options = dependencies.expert_dynamic_memory_options
        self.expert_recurrent_controller_options = (
            dependencies.expert_recurrent_controller_options
        )
        self.adaptive_generator_stack_options = (
            dependencies.adaptive_generator_stack_options
        )
        self.adaptive_generator_stack_config_factory = self._generator(
            self.adaptive_generator_stack_options
        )
        self.hidden_adaptive_weight_options = (
            dependencies.hidden_adaptive_weight_options
        )
        self.grouping_config = dependencies.grouping_config
        self.hidden_adaptive_bias_options = dependencies.hidden_adaptive_bias_options
        self.hidden_adaptive_diagonal_options = (
            dependencies.hidden_adaptive_diagonal_options
        )
        self.hidden_adaptive_mask_options = dependencies.hidden_adaptive_mask_options
        self.router_adaptive_weight_options = (
            dependencies.router_adaptive_weight_options
        )
        self.router_grouping_config = dependencies.router_grouping_config
        self.router_adaptive_bias_options = dependencies.router_adaptive_bias_options
        self.router_adaptive_diagonal_options = (
            dependencies.router_adaptive_diagonal_options
        )
        self.router_adaptive_mask_options = dependencies.router_adaptive_mask_options
        augmentation = AdaptiveAugmentationFactory(
            self.adaptive_generator_stack_config_factory
        )
        self.hidden_adaptive_augmentation_config = augmentation.build(
            grouping=self.grouping_config,
            weight=self.hidden_adaptive_weight_options,
            bias=self.hidden_adaptive_bias_options,
            diagonal=self.hidden_adaptive_diagonal_options,
            mask=self.hidden_adaptive_mask_options,
        )
        self.router_adaptive_augmentation_config = augmentation.build(
            grouping=self.router_grouping_config,
            weight=self.router_adaptive_weight_options,
            bias=self.router_adaptive_bias_options,
            diagonal=self.router_adaptive_diagonal_options,
            mask=self.router_adaptive_mask_options,
        )
        self.hidden_dim = dependencies.hidden_dim
        self.output_dim = dependencies.output_dim
        self._initialize_controllers()

    @property
    def _expert_settings(self) -> RoutedExpertSettings:
        return self


class _PlainInputs(Protocol):
    hidden_dim: Final[int]
    stack_options: Final[object]
    submodule_stack_options: Final[object]
    mixture_options: Final[object]
    expert_stack_options: Final[object]
    sampler_options: Final[object]
    router_options: Final[object]
    router_stack_options: Final[object]
    layer_controller_options: Final[object]
    dynamic_memory_options: Final[object]
    recurrent_controller_options: Final[object]
    expert_layer_controller_options: Final[object]
    expert_dynamic_memory_options: Final[object]
    expert_recurrent_controller_options: Final[object]
    output_dim: Final[int]


_Plain = TypeVar("_Plain", bound=_PlainInputs)


class LanguageExpertFactory(ExpertStackFactory, Generic[_Plain]):
    _config: ModuleType
    _defaults: ModuleType

    def __init__(self, dependencies: _Plain) -> None:
        defaults = self._defaults
        config = self._config
        self.stack_options: LayerStackSettings = defaults.resolve_options(
            dependencies.stack_options,
            defaults.experts_stack_options,
            config,
            dependencies.hidden_dim,
        )
        self.submodule_stack_options = defaults.resolve_options(
            dependencies.submodule_stack_options,
            defaults.experts_submodule_stack_options,
            config,
            "base",
        )
        self.mixture_options = defaults.resolve_options(
            dependencies.mixture_options,
            defaults.experts_mixture_options,
            config,
        )
        self.expert_stack_options = defaults.resolve_options(
            dependencies.expert_stack_options,
            defaults.experts_role_stack_options,
            config,
            "expert",
            self.submodule_stack_options,
        )
        self.sampler_options = defaults.resolve_options(
            dependencies.sampler_options,
            defaults.experts_sampler_options,
            config,
        )
        self.router_options = defaults.resolve_options(
            dependencies.router_options, defaults.experts_router_options, config
        )
        self.router_stack_options = defaults.resolve_options(
            dependencies.router_stack_options,
            defaults.experts_submodule_stack_options,
            config,
            "router",
        )
        self.layer_controller_options = defaults.resolve_options(
            dependencies.layer_controller_options,
            defaults.experts_layer_controller_options,
            config,
            "main",
        )
        self.dynamic_memory_options = defaults.resolve_options(
            dependencies.dynamic_memory_options,
            defaults.experts_dynamic_memory_options,
            config,
            "main",
        )
        self.recurrent_controller_options = defaults.resolve_options(
            dependencies.recurrent_controller_options,
            defaults.experts_recurrent_controller_options,
            config,
            "main",
        )
        self.expert_layer_controller_options = defaults.expert_layer_controller_options(
            config,
            dependencies.expert_layer_controller_options,
        )
        self.expert_dynamic_memory_options = defaults.expert_dynamic_memory_options(
            config,
            dependencies.expert_dynamic_memory_options,
        )
        self.expert_recurrent_controller_options = (
            defaults.expert_recurrent_controller_options(
                config,
                dependencies.expert_recurrent_controller_options,
            )
        )
        self.hidden_dim = self.stack_options.hidden_dim
        self.output_dim = dependencies.output_dim
        self._initialize_controllers()

    @property
    def _expert_settings(self) -> ExpertSettings:
        return self

    def _build_router_model(self) -> LayerStackConfig:
        return build_controller_stack_config(self.router_stack_options)
