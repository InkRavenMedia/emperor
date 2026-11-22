from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from types import ModuleType
from typing import Final, Generic, Protocol, TypeVar

from emperor.attention import MixtureOfAttentionHeadsConfig
from emperor.experts import MixtureOfExpertsModelConfig
from emperor.layers import LayerStackConfig, RecurrentLayerConfig
from models._component_defaults import resolve_options
from models._linear_stack import LayerStackSettings


class _Inputs(Protocol):
    encoder_options: Final[object]
    attention_options: Final[object]
    feed_forward_options: Final[object]
    mixture_options: Final[object]
    expert_stack_options: Final[object]
    sampler_options: Final[object]
    router_options: Final[object]
    router_stack_options: Final[object]
    expert_layer_controller_options: Final[object]
    expert_dynamic_memory_options: Final[object]
    expert_recurrent_controller_options: Final[object]
    hidden_dim: Final[int]
    expert_attention_use_kv_expert_models_flag: Final[bool]


class _AdaptiveInputs(_Inputs, Protocol):
    grouping_config: Final[object]
    router_grouping_config: Final[object]
    mixture_submodule_stack_options: Final[object]
    mixture_layer_controller_options: Final[object]
    mixture_dynamic_memory_options: Final[object]
    mixture_recurrent_controller_options: Final[object]
    router_layer_controller_options: Final[object]
    router_dynamic_memory_options: Final[object]
    router_recurrent_controller_options: Final[object]
    adaptive_generator_stack_options: Final[object]
    hidden_adaptive_weight_options: Final[object]
    hidden_adaptive_bias_options: Final[object]
    hidden_adaptive_diagonal_options: Final[object]
    hidden_adaptive_mask_options: Final[object]
    router_adaptive_weight_options: Final[object]
    router_adaptive_bias_options: Final[object]
    router_adaptive_diagonal_options: Final[object]
    router_adaptive_mask_options: Final[object]


_StackIn = TypeVar("_StackIn", bound=LayerStackSettings, contravariant=True)


class _ExpertBuilder(Protocol[_StackIn]):
    def build_feed_forward_base_stack_config(
        self, feed_forward_stack_options: _StackIn
    ) -> MixtureOfExpertsModelConfig: ...
    def build_attention_config(
        self,
        *,
        batch_size: int,
        hidden_dim: int,
        sequence_length: int,
        projection_model_config: LayerStackConfig | RecurrentLayerConfig,
    ) -> MixtureOfAttentionHeadsConfig: ...


_Settings = TypeVar("_Settings", bound=_Inputs)
_Adaptive = TypeVar("_Adaptive", bound=_AdaptiveInputs)
_Resolved = TypeVar("_Resolved")
_Stack = TypeVar("_Stack", bound=LayerStackSettings)


class _ConfiguredExperts(ABC, Generic[_Settings, _Resolved, _Stack]):
    _config: ModuleType
    _defaults: ModuleType
    _resolved_inputs: Callable[..., _Resolved]
    _builder: Callable[[_Resolved], _ExpertBuilder[_Stack]]
    dependencies: _Settings

    def _resolve(
        self, supplied: object, factory: Callable[..., object], *roles: object
    ) -> object:
        return resolve_options(supplied, factory, self._config, *roles)

    @abstractmethod
    def _expert_config_factory(self) -> _ExpertBuilder[_Stack]: ...

    def build_feed_forward_base_stack_config(
        self,
        feed_forward_stack_options: _Stack,
    ) -> MixtureOfExpertsModelConfig:
        feed_forward_base_stack_config = (
            self._expert_config_factory().build_feed_forward_base_stack_config(
                feed_forward_stack_options
            )
        )
        return feed_forward_base_stack_config

    def build_attention_config(
        self,
        *,
        batch_size: int,
        hidden_dim: int,
        sequence_length: int,
        projection_model_config: LayerStackConfig | RecurrentLayerConfig,
    ) -> MixtureOfAttentionHeadsConfig:
        attention_config = self._expert_config_factory().build_attention_config(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            sequence_length=sequence_length,
            projection_model_config=projection_model_config,
        )
        return attention_config

    def _resolve_model_options(self) -> None:
        dependencies = self.dependencies
        defaults = self._defaults
        self.encoder_options = self._resolve(
            dependencies.encoder_options, defaults.vit_encoder_options
        )
        self.attention_options = self._resolve(
            dependencies.attention_options, defaults.vit_attention_options
        )
        self.feed_forward_options = self._resolve(
            dependencies.feed_forward_options, defaults.vit_feed_forward_options
        )
        self.mixture_options = self._resolve(
            dependencies.mixture_options, defaults.experts_mixture_options
        )

    def _resolve_expert_routing(self) -> None:
        dependencies = self.dependencies
        defaults = self._defaults
        self.expert_stack_options = self._resolve(
            dependencies.expert_stack_options,
            defaults.experts_submodule_stack_options,
            defaults.ExpertStackRole.EXPERT,
        )
        self.sampler_options = self._resolve(
            dependencies.sampler_options, defaults.experts_sampler_options
        )
        self.router_options = self._resolve(
            dependencies.router_options, defaults.experts_router_options
        )
        self.router_stack_options = self._resolve(
            dependencies.router_stack_options,
            defaults.experts_submodule_stack_options,
            defaults.ExpertStackRole.ROUTER,
        )

    def _resolve_expert_controllers(self) -> None:
        dependencies = self.dependencies
        defaults = self._defaults
        self.expert_layer_controller_options = self._resolve(
            dependencies.expert_layer_controller_options,
            defaults.experts_layer_controller_options,
            defaults.ExpertControlRole.EXPERT,
        )
        self.expert_dynamic_memory_options = self._resolve(
            dependencies.expert_dynamic_memory_options,
            defaults.experts_dynamic_memory_options,
            defaults.ExpertControlRole.EXPERT,
        )
        self.expert_recurrent_controller_options = self._resolve(
            dependencies.expert_recurrent_controller_options,
            defaults.experts_recurrent_controller_options,
            defaults.ExpertControlRole.EXPERT,
        )


class ExpertDefaults(_ConfiguredExperts[_Settings, _Resolved, _Stack]):
    def __init__(self, dependencies: _Settings) -> None:
        self.dependencies = dependencies
        self._resolve_model_options()
        self._resolve_expert_routing()
        self._resolve_expert_controllers()

    def _expert_config_factory(self) -> _ExpertBuilder[_Stack]:
        dependencies = self.dependencies
        expert_dependencies = self._resolved_inputs(
            hidden_dim=dependencies.hidden_dim,
            encoder_options=self.encoder_options,
            attention_options=self.attention_options,
            feed_forward_options=self.feed_forward_options,
            mixture_options=self.mixture_options,
            expert_stack_options=self.expert_stack_options,
            sampler_options=self.sampler_options,
            router_options=self.router_options,
            router_stack_options=self.router_stack_options,
            expert_layer_controller_options=(self.expert_layer_controller_options),
            expert_dynamic_memory_options=self.expert_dynamic_memory_options,
            expert_recurrent_controller_options=(
                self.expert_recurrent_controller_options
            ),
            expert_attention_use_kv_expert_models_flag=(
                dependencies.expert_attention_use_kv_expert_models_flag
            ),
        )
        return self._builder(expert_dependencies)


class AdaptiveExpertDefaults(_ConfiguredExperts[_Adaptive, _Resolved, _Stack]):
    def __init__(self, dependencies: _Adaptive) -> None:
        self.dependencies = dependencies
        self._resolve_model_options()
        self._resolve_mixture_controllers()
        self._resolve_expert_routing()
        self._resolve_router_controllers()
        self._resolve_expert_controllers()
        self._resolve_adaptive_parameters()

    def _resolve_mixture_controllers(self) -> None:
        dependencies = self.dependencies
        defaults = self._defaults
        self.mixture_submodule_stack_options = self._resolve(
            dependencies.mixture_submodule_stack_options,
            defaults.experts_submodule_stack_options,
            defaults.ExpertStackRole.MAIN,
        )
        self.mixture_layer_controller_options = self._resolve(
            dependencies.mixture_layer_controller_options,
            defaults.experts_layer_controller_options,
            defaults.ExpertControlRole.MAIN,
        )
        self.mixture_dynamic_memory_options = self._resolve(
            dependencies.mixture_dynamic_memory_options,
            defaults.experts_dynamic_memory_options,
            defaults.ExpertControlRole.MAIN,
        )
        self.mixture_recurrent_controller_options = self._resolve(
            dependencies.mixture_recurrent_controller_options,
            defaults.experts_recurrent_controller_options,
            defaults.ExpertControlRole.MAIN,
        )

    def _resolve_router_controllers(self) -> None:
        dependencies = self.dependencies
        defaults = self._defaults
        self.router_layer_controller_options = self._resolve(
            dependencies.router_layer_controller_options,
            defaults.experts_layer_controller_options,
            defaults.ExpertControlRole.ROUTER,
        )
        self.router_dynamic_memory_options = self._resolve(
            dependencies.router_dynamic_memory_options,
            defaults.experts_dynamic_memory_options,
            defaults.ExpertControlRole.ROUTER,
        )
        self.router_recurrent_controller_options = self._resolve(
            dependencies.router_recurrent_controller_options,
            defaults.experts_recurrent_controller_options,
            defaults.ExpertControlRole.ROUTER,
        )

    def _resolve_adaptive_parameters(self) -> None:
        dependencies = self.dependencies
        defaults = self._defaults
        self.adaptive_generator_stack_options = self._resolve(
            dependencies.adaptive_generator_stack_options,
            defaults.adaptive_generator_stack_options,
        )
        self.hidden_adaptive_weight_options = self._resolve(
            dependencies.hidden_adaptive_weight_options,
            defaults.hidden_adaptive_weight_options,
        )
        self.grouping_config = dependencies.grouping_config
        self.hidden_adaptive_bias_options = self._resolve(
            dependencies.hidden_adaptive_bias_options,
            defaults.hidden_adaptive_bias_options,
        )
        self.hidden_adaptive_diagonal_options = self._resolve(
            dependencies.hidden_adaptive_diagonal_options,
            defaults.hidden_adaptive_diagonal_options,
        )
        self.hidden_adaptive_mask_options = self._resolve(
            dependencies.hidden_adaptive_mask_options,
            defaults.hidden_adaptive_mask_options,
        )
        self.router_adaptive_weight_options = self._resolve(
            dependencies.router_adaptive_weight_options,
            defaults.hidden_adaptive_weight_options,
            defaults.AdaptiveRole.ROUTER,
        )
        self.router_grouping_config = dependencies.router_grouping_config
        self.router_adaptive_bias_options = self._resolve(
            dependencies.router_adaptive_bias_options,
            defaults.hidden_adaptive_bias_options,
            defaults.AdaptiveRole.ROUTER,
        )
        self.router_adaptive_diagonal_options = self._resolve(
            dependencies.router_adaptive_diagonal_options,
            defaults.hidden_adaptive_diagonal_options,
            defaults.AdaptiveRole.ROUTER,
        )
        self.router_adaptive_mask_options = self._resolve(
            dependencies.router_adaptive_mask_options,
            defaults.hidden_adaptive_mask_options,
            defaults.AdaptiveRole.ROUTER,
        )

    def _expert_config_factory(self) -> _ExpertBuilder[_Stack]:
        dependencies = self.dependencies
        expert_adaptive_dependencies = self._resolved_inputs(
            hidden_dim=dependencies.hidden_dim,
            encoder_options=self.encoder_options,
            attention_options=self.attention_options,
            feed_forward_options=self.feed_forward_options,
            mixture_options=self.mixture_options,
            expert_stack_options=self.expert_stack_options,
            sampler_options=self.sampler_options,
            router_options=self.router_options,
            router_stack_options=self.router_stack_options,
            expert_layer_controller_options=(self.expert_layer_controller_options),
            expert_dynamic_memory_options=self.expert_dynamic_memory_options,
            expert_recurrent_controller_options=(
                self.expert_recurrent_controller_options
            ),
            expert_attention_use_kv_expert_models_flag=(
                dependencies.expert_attention_use_kv_expert_models_flag
            ),
            mixture_submodule_stack_options=(self.mixture_submodule_stack_options),
            mixture_layer_controller_options=(self.mixture_layer_controller_options),
            mixture_dynamic_memory_options=(self.mixture_dynamic_memory_options),
            mixture_recurrent_controller_options=(
                self.mixture_recurrent_controller_options
            ),
            router_layer_controller_options=(self.router_layer_controller_options),
            router_dynamic_memory_options=self.router_dynamic_memory_options,
            router_recurrent_controller_options=(
                self.router_recurrent_controller_options
            ),
            adaptive_generator_stack_options=(self.adaptive_generator_stack_options),
            grouping_config=self.grouping_config,
            hidden_adaptive_weight_options=self.hidden_adaptive_weight_options,
            hidden_adaptive_bias_options=self.hidden_adaptive_bias_options,
            hidden_adaptive_diagonal_options=(self.hidden_adaptive_diagonal_options),
            hidden_adaptive_mask_options=self.hidden_adaptive_mask_options,
            router_grouping_config=self.router_grouping_config,
            router_adaptive_weight_options=self.router_adaptive_weight_options,
            router_adaptive_bias_options=self.router_adaptive_bias_options,
            router_adaptive_diagonal_options=(self.router_adaptive_diagonal_options),
            router_adaptive_mask_options=self.router_adaptive_mask_options,
        )
        return self._builder(expert_adaptive_dependencies)
