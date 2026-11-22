from abc import ABC, abstractmethod
from types import EllipsisType
from typing import Final, Protocol, TypeVar

from emperor.config import ConfigBase
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
)
from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LayerStackConfig,
    RecurrentCompositionConfig,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import DynamicMemoryConfig
from emperor.sampler import RouterConfig, SamplerConfig
from models._controller_stack import build_controller_layer, build_controller_residual
from models._linear_stack import LayerStackSettings
from models._stack_controllers import (
    ExpertMemoryConstructor,
    GateConstructor,
    HaltingConstructor,
    RecurrentConstructor,
)


class _ExpertCounts(Protocol):
    @property
    def top_k(self) -> int: ...

    @property
    def num_experts(self) -> int: ...


class _ExpertModelOptions(Protocol):
    @property
    def top_k(self) -> int: ...

    @property
    def routing_initialization_mode(self) -> RoutingInitializationMode: ...


class _ExpertMixtureOptions(_ExpertCounts, _ExpertModelOptions, Protocol):
    @property
    def capacity_factor(self) -> float: ...

    @property
    def dropped_token_behavior(self) -> DroppedTokenOptions: ...

    @property
    def compute_expert_mixture_flag(self) -> bool: ...

    @property
    def weighted_parameters_flag(self) -> bool: ...

    @property
    def weighting_position_option(self) -> ExpertWeightingPositionOptions: ...


class SamplerSettings(Protocol):
    @property
    def threshold(self) -> float: ...

    @property
    def filter_above_threshold(self) -> bool: ...

    @property
    def num_topk_samples(self) -> int: ...

    @property
    def normalize_probabilities_flag(self) -> bool: ...

    @property
    def noisy_topk_flag(self) -> bool: ...

    @property
    def coefficient_of_variation_loss_weight(self) -> float: ...

    @property
    def switch_loss_weight(self) -> float: ...

    @property
    def zero_centred_loss_weight(self) -> float: ...

    @property
    def mutual_information_loss_weight(self) -> float: ...


def build_routed_expert_sampler(
    mixture: _ExpertCounts,
    sampler: SamplerSettings,
    *,
    input_dim: int | None,
    router_noisy_topk_flag: bool,
    router_model_config: ConfigBase,
) -> SamplerConfig:
    router_config = RouterConfig(
        input_dim=input_dim,
        num_experts=mixture.num_experts,
        noisy_topk_flag=router_noisy_topk_flag,
        model_config=router_model_config,
    )
    return build_expert_sampler(mixture, sampler, router_config=router_config)


def build_expert_sampler(
    mixture: _ExpertCounts,
    sampler: SamplerSettings,
    *,
    router_config: RouterConfig | None = None,
) -> SamplerConfig:
    return SamplerConfig(
        top_k=mixture.top_k,
        threshold=sampler.threshold,
        filter_above_threshold=sampler.filter_above_threshold,
        num_topk_samples=sampler.num_topk_samples,
        normalize_probabilities_flag=sampler.normalize_probabilities_flag,
        noisy_topk_flag=sampler.noisy_topk_flag,
        num_experts=mixture.num_experts,
        coefficient_of_variation_loss_weight=sampler.coefficient_of_variation_loss_weight,
        switch_loss_weight=sampler.switch_loss_weight,
        zero_centred_loss_weight=sampler.zero_centred_loss_weight,
        mutual_information_loss_weight=sampler.mutual_information_loss_weight,
        router_config=router_config,
    )


def build_expert_mixture(
    options: _ExpertMixtureOptions,
    *,
    input_dim: int | None,
    output_dim: int | None,
    sampler_config: SamplerConfig | None,
    expert_model_config: LayerStackConfig | RecurrentCompositionConfig,
) -> MixtureOfExpertsConfig:
    return MixtureOfExpertsConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=options.top_k,
        num_experts=options.num_experts,
        capacity_factor=options.capacity_factor,
        dropped_token_behavior=options.dropped_token_behavior,
        compute_expert_mixture_flag=options.compute_expert_mixture_flag,
        weighted_parameters_flag=options.weighted_parameters_flag,
        weighting_position_option=options.weighting_position_option,
        routing_initialization_mode=options.routing_initialization_mode,
        sampler_config=sampler_config,
        expert_model_config=expert_model_config,
    )


def build_expert_model(
    options: _ExpertModelOptions,
    *,
    input_dim: int | None,
    output_dim: int | None,
    sampler_config: SamplerConfig | None,
    stack_config: LayerStackConfig,
) -> MixtureOfExpertsModelConfig:
    return MixtureOfExpertsModelConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=options.top_k,
        routing_initialization_mode=options.routing_initialization_mode,
        sampler_config=sampler_config,
        stack_config=stack_config,
    )


class _MemorySettings(Protocol):
    memory_flag: Final[bool]


class _RecurrentSettings(Protocol):
    recurrent_flag: Final[bool]


class _SharedGate(Protocol):
    stack_gate_flag: Final[bool]
    stack_halting_flag: Final[bool]
    shared_gate_config: Final[GateConfig | None]


class ExpertRouterOptions(Protocol):
    noisy_topk_flag: Final[bool]


class ExpertSettings(Protocol):
    output_dim: Final[int]
    submodule_stack_options: Final[LayerStackSettings]
    dynamic_memory_options: Final[_MemorySettings]
    recurrent_controller_options: Final[_RecurrentSettings]
    expert_dynamic_memory_options: Final[_MemorySettings]
    expert_recurrent_controller_options: Final[_RecurrentSettings]
    router_stack_options: Final[LayerStackSettings]
    sampler_options: Final[SamplerSettings]
    router_options: Final[ExpertRouterOptions]
    stack_options: Final[LayerStackSettings]
    layer_controller_options: Final[_SharedGate]
    expert_stack_options: Final[LayerStackSettings]
    expert_layer_controller_options: Final[_SharedGate]
    mixture_options: Final[_ExpertMixtureOptions]


class GateBuilder(Protocol):
    def build_gate_config(self) -> GateConfig | None: ...
    def build_recurrent_gate_config(self) -> GateConfig | None: ...


class HaltingBuilder(Protocol):
    def build_halting_config(self) -> HaltingConfig | None: ...
    def build_recurrent_halting_config(self) -> HaltingConfig | None: ...


class MemoryBuilder(Protocol):
    def build_memory_config(self) -> DynamicMemoryConfig | None: ...


_Block = TypeVar("_Block", bound=ConfigBase)


class RecurrentBuilder(Protocol):
    def build_config(self, block_config: _Block) -> _Block | RecurrentLayerConfig: ...


class ExpertStackFactory(ABC):
    _gate_type: GateConstructor[GateBuilder]
    _halting_type: HaltingConstructor[HaltingBuilder]
    _memory_type: ExpertMemoryConstructor[MemoryBuilder]
    _recurrent_type: RecurrentConstructor[RecurrentBuilder]

    hidden_dim: int
    gate_config_factory: GateBuilder
    halting_config_factory: HaltingBuilder
    memory_config_factory: MemoryBuilder
    recurrent_config_factory: RecurrentBuilder
    expert_gate_config_factory: GateBuilder
    expert_halting_config_factory: HaltingBuilder
    expert_memory_config_factory: MemoryBuilder
    expert_recurrent_config_factory: RecurrentBuilder

    @property
    @abstractmethod
    def _expert_settings(self) -> ExpertSettings: ...

    @abstractmethod
    def _build_router_model(self) -> ConfigBase: ...

    def _build_sampler_config(self) -> SamplerConfig:
        settings = self._expert_settings
        mixture_options = settings.mixture_options
        sampler_options = settings.sampler_options
        router_options = settings.router_options
        router_model = self._build_router_model()
        return build_routed_expert_sampler(
            mixture_options,
            sampler_options,
            input_dim=self.hidden_dim,
            router_noisy_topk_flag=router_options.noisy_topk_flag,
            router_model_config=router_model,
        )

    def _initialize_controllers(self) -> None:
        settings = self._expert_settings
        self.gate_config_factory = self._gate_type(
            layer_controller_options=settings.layer_controller_options,
            recurrent_controller_options=settings.recurrent_controller_options,
            submodule_stack_options=settings.submodule_stack_options,
        )
        self.halting_config_factory = self._halting_type(
            layer_controller_options=settings.layer_controller_options,
            recurrent_controller_options=settings.recurrent_controller_options,
            submodule_stack_options=settings.submodule_stack_options,
            output_dim=settings.output_dim,
        )
        self.memory_config_factory = self._memory_type(
            stack_options=settings.stack_options,
            dynamic_memory_options=settings.dynamic_memory_options,
            submodule_stack_options=settings.submodule_stack_options,
        )
        self.recurrent_config_factory = self._recurrent_type(
            recurrent_controller_options=settings.recurrent_controller_options,
            gate_config_factory=self.gate_config_factory,
            halting_config_factory=self.halting_config_factory,
        )
        self.expert_gate_config_factory = self._gate_type(
            layer_controller_options=settings.expert_layer_controller_options,
            recurrent_controller_options=settings.expert_recurrent_controller_options,
            submodule_stack_options=settings.expert_stack_options,
            recurrent_stack_inherits_gate_stack=False,
        )
        self.expert_halting_config_factory = self._halting_type(
            layer_controller_options=settings.expert_layer_controller_options,
            recurrent_controller_options=settings.expert_recurrent_controller_options,
            submodule_stack_options=settings.expert_stack_options,
            output_dim=settings.expert_stack_options.hidden_dim,
            halting_stack_defaults=settings.expert_stack_options,
            recurrent_stack_inherits_halting_stack=False,
        )
        self.expert_memory_config_factory = self._memory_type(
            stack_options=settings.expert_stack_options,
            dynamic_memory_options=settings.expert_dynamic_memory_options,
            submodule_stack_options=settings.expert_stack_options,
        )
        self.expert_recurrent_config_factory = self._recurrent_type(
            recurrent_controller_options=settings.expert_recurrent_controller_options,
            gate_config_factory=self.expert_gate_config_factory,
            halting_config_factory=self.expert_halting_config_factory,
        )

    def build(self) -> MixtureOfExpertsModelConfig | RecurrentLayerConfig:
        options = self._expert_settings.mixture_options
        sampler_config = self._build_sampler_config()
        stack_config = self.__build_main_stack()
        model_config = build_expert_model(
            options,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            sampler_config=sampler_config,
            stack_config=stack_config,
        )
        return self.recurrent_config_factory.build_config(model_config)

    def __build_main_stack(self) -> LayerStackConfig:
        options = self._expert_settings.stack_options
        controller = self._expert_settings.layer_controller_options
        gate_config = self.gate_config_factory.build_gate_config()
        halting_config = self.halting_config_factory.build_halting_config()
        memory_config = self.memory_config_factory.build_memory_config()
        layer_config = self.__build_mixture_layer(gate_config, halting_config)
        return LayerStackConfig(
            input_dim=options.hidden_dim,
            hidden_dim=options.hidden_dim,
            output_dim=options.hidden_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
            shared_gate_config=controller.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )

    def __build_mixture_layer(
        self, gate_config: GateConfig | None, halting_config: HaltingConfig | None
    ) -> MixtureOfExpertsLayerConfig:
        options = self._expert_settings.stack_options
        residual_config = build_controller_residual(options)
        mixture_config = self.__build_mixture()
        return MixtureOfExpertsLayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            residual_config=residual_config,
            dropout_probability=options.dropout_probability,
            gate_config=gate_config,
            halting_config=halting_config,
            layer_model_config=mixture_config,
        )

    def __build_mixture(self) -> MixtureOfExpertsConfig:
        options = self._expert_settings.mixture_options
        sampler_config = self._build_sampler_config()
        expert_stack = self._build_expert_stack()
        expert_model = self.expert_recurrent_config_factory.build_config(expert_stack)
        return build_expert_mixture(
            options,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            sampler_config=sampler_config,
            expert_model_config=expert_model,
        )

    def _build_expert_stack(self) -> LayerStackConfig:
        options = self._expert_settings.expert_stack_options
        return self._assemble_expert_stack(options)

    def _assemble_expert_stack(
        self,
        options: LayerStackSettings,
        layer_model_config: ConfigBase | EllipsisType = ...,
    ) -> LayerStackConfig:
        controller = self._expert_settings.expert_layer_controller_options
        gate_config = self.expert_gate_config_factory.build_gate_config()
        halting_config = self.expert_halting_config_factory.build_halting_config()
        memory_config = self.expert_memory_config_factory.build_memory_config()
        residual_config = build_controller_residual(options)
        if layer_model_config is ...:
            layer_model_config = LinearLayerConfig(bias_flag=options.bias_flag)
        layer_config = build_controller_layer(
            options,
            layer_model_config,
            residual_config,
            gate_config=gate_config,
            halting_config=halting_config,
        )
        return LayerStackConfig(
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
            shared_gate_config=controller.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )
