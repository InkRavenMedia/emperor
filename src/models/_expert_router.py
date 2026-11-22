from abc import abstractmethod
from typing import Final, Protocol

from emperor.config import ConfigBase
from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
)
from models._controller_stack import (
    build_controller_layer,
    build_controller_residual,
    build_controller_stack_config,
)
from models._expert_mixture import (
    ExpertSettings,
    ExpertStackFactory,
    GateBuilder,
    HaltingBuilder,
    MemoryBuilder,
    RecurrentBuilder,
)
from models._router_controller_config import RouterControllerModelConfig


class _LayerFlags(Protocol):
    stack_gate_flag: Final[bool]
    stack_halting_flag: Final[bool]


class _MemoryFlag(Protocol):
    memory_flag: Final[bool]


class _RecurrentFlag(Protocol):
    recurrent_flag: Final[bool]


class RoutedExpertSettings(ExpertSettings, Protocol):
    router_layer_controller_options: Final[_LayerFlags]
    router_dynamic_memory_options: Final[_MemoryFlag]
    router_recurrent_controller_options: Final[_RecurrentFlag]


class RoutedExpertStackFactory(ExpertStackFactory):
    _router_config: type[RouterControllerModelConfig]
    _build_standalone_stack = staticmethod(build_controller_stack_config)
    router_gate_config_factory: GateBuilder
    router_halting_config_factory: HaltingBuilder
    router_memory_config_factory: MemoryBuilder
    router_recurrent_config_factory: RecurrentBuilder

    @property
    @abstractmethod
    def _expert_settings(self) -> RoutedExpertSettings: ...

    @abstractmethod
    def build_router_adaptive_linear_layer_config(self) -> ConfigBase: ...

    def _initialize_controllers(self) -> None:
        super()._initialize_controllers()
        settings = self._expert_settings
        self.router_gate_config_factory = self._gate_type(
            layer_controller_options=settings.router_layer_controller_options,
            recurrent_controller_options=settings.router_recurrent_controller_options,
            submodule_stack_options=settings.submodule_stack_options,
            recurrent_stack_inherits_gate_stack=False,
        )
        self.router_halting_config_factory = self._halting_type(
            layer_controller_options=settings.router_layer_controller_options,
            recurrent_controller_options=settings.router_recurrent_controller_options,
            submodule_stack_options=settings.submodule_stack_options,
            output_dim=settings.router_stack_options.hidden_dim,
            recurrent_stack_inherits_halting_stack=False,
        )
        self.router_memory_config_factory = self._memory_type(
            stack_options=settings.router_stack_options,
            dynamic_memory_options=settings.router_dynamic_memory_options,
            submodule_stack_options=settings.submodule_stack_options,
        )
        self.router_recurrent_config_factory = self._recurrent_type(
            recurrent_controller_options=settings.router_recurrent_controller_options,
            gate_config_factory=self.router_gate_config_factory,
            halting_config_factory=self.router_halting_config_factory,
        )

    def _build_router_model(self) -> ConfigBase:
        stack_options = self._expert_settings.router_stack_options
        layer = self.build_router_adaptive_linear_layer_config()
        if self.__controller_enabled():
            return self.__build_controlled_router(layer)
        return self._build_standalone_stack(stack_options, layer_model_config=layer)

    def __controller_enabled(self) -> bool:
        settings = self._expert_settings
        return (
            settings.router_layer_controller_options.stack_gate_flag
            or settings.router_layer_controller_options.stack_halting_flag
            or settings.router_dynamic_memory_options.memory_flag
            or settings.router_recurrent_controller_options.recurrent_flag
        )

    def __build_controlled_router(
        self, layer: ConfigBase
    ) -> RouterControllerModelConfig:
        settings = self._expert_settings
        trunk_dim = settings.router_stack_options.hidden_dim
        stack = self.__build_trunk(layer)
        trunk = self.router_recurrent_config_factory.build_config(stack)
        output_dim = settings.mixture_options.num_experts
        if settings.router_options.noisy_topk_flag:
            output_dim = 2 * output_dim
        adapter = self.__build_projection(layer)
        head = self.__build_projection(layer)
        return self._router_config(
            input_dim=self.hidden_dim,
            hidden_dim=trunk_dim,
            output_dim=output_dim,
            adapter_config=adapter,
            trunk_config=trunk,
            head_config=head,
        )

    def __build_trunk(self, layer: ConfigBase) -> LayerStackConfig:
        options = self._expert_settings.router_stack_options
        gate = self.router_gate_config_factory.build_gate_config()
        halting = self.router_halting_config_factory.build_halting_config()
        memory = self.router_memory_config_factory.build_memory_config()
        residual_config = build_controller_residual(options)
        layer_config = build_controller_layer(
            options, layer, residual_config, gate_config=gate, halting_config=halting
        )
        return LayerStackConfig(
            input_dim=options.hidden_dim,
            hidden_dim=options.hidden_dim,
            output_dim=options.hidden_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
            shared_memory_config=memory,
            layer_config=layer_config,
        )

    @staticmethod
    def __build_projection(layer: ConfigBase) -> LayerConfig:
        return LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=layer,
        )
