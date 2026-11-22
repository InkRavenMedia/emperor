from collections.abc import Callable
from types import ModuleType
from typing import Final, Generic, Protocol, TypeVar

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    GroupingConfig,
)
from emperor.halting import HaltingConfig
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig
from models._adaptive_attention_backend import AdaptiveOptionDefaults, AdaptiveOptions
from models._adaptive_augmentation import (
    AdaptiveAugmentationFactory,
    BiasOptions,
    DiagonalOptions,
    MaskOptions,
    WeightOptions,
)
from models._adaptive_generator_stack import (
    StackGenerator,
)
from models._component_defaults import resolve_options
from models._residual_stack import ResidualStackSettings, build_residual_config


class _AdaptiveStack(Protocol):
    bias_flag: Final[bool]
    activation: Final[ActivationOptions]
    layer_norm_position: Final[LayerNormPositionOptions]
    normalization: Final[NormalizationOptions]
    dropout_probability: Final[float]
    num_layers: Final[int]
    last_layer_bias_option: Final[LastLayerBiasOptions]
    apply_output_postprocessing_flag: Final[bool]
    residual_connection_option: Final[type[ResidualConfig] | None]
    residual_model_flag: Final[bool]
    residual_block_size: Final[int | None]
    residual_rms_norm_epsilon: Final[float | None]
    residual_stack_options: Final[ResidualStackSettings | None]


class _Controller(Protocol):
    shared_gate_config: Final[GateConfig | None]


class _GateFactory(Protocol):
    def build_gate_config(self) -> GateConfig | None: ...


class _HaltingFactory(Protocol):
    def build_halting_config(self) -> HaltingConfig | None: ...


class _MemoryFactory(Protocol):
    def build_memory_config(self) -> DynamicMemoryConfig | None: ...


class _RecurrentFactory(Protocol):
    def build_config(
        self, block_config: LayerStackConfig
    ) -> LayerStackConfig | RecurrentLayerConfig: ...


_Submodule = TypeVar("_Submodule")
_MemoryOptions = TypeVar("_MemoryOptions")
_RecurrentOptions = TypeVar("_RecurrentOptions")
_GeneratorOptions = TypeVar("_GeneratorOptions")
_SubmoduleCo = TypeVar("_SubmoduleCo", covariant=True)
_MemoryCo = TypeVar("_MemoryCo", covariant=True)
_RecurrentCo = TypeVar("_RecurrentCo", covariant=True)
_GeneratorCo = TypeVar("_GeneratorCo", covariant=True)
_Role = TypeVar("_Role")
_RoleIn = TypeVar("_RoleIn", contravariant=True)


class _HiddenOptions(
    AdaptiveOptions[_GeneratorCo],
    Protocol[_SubmoduleCo, _MemoryCo, _RecurrentCo, _GeneratorCo],
):
    stack_options: Final[_AdaptiveStack | None]
    submodule_stack_options: Final[_SubmoduleCo | None]
    layer_controller_options: Final[_Controller | None]
    dynamic_memory_options: Final[_MemoryCo | None]
    recurrent_controller_options: Final[_RecurrentCo | None]


class _HiddenDefaults(
    AdaptiveOptionDefaults[_GeneratorCo],
    Protocol[_RoleIn, _SubmoduleCo, _MemoryCo, _RecurrentCo, _GeneratorCo],
):
    def main_layer_stack_options(self, config: ModuleType, /) -> _AdaptiveStack: ...
    def linears_submodule_stack_options(
        self, config: ModuleType, role: _RoleIn, /
    ) -> _SubmoduleCo: ...
    def linears_layer_controller_options(
        self, config: ModuleType, role: _RoleIn, /
    ) -> _Controller: ...
    def linears_dynamic_memory_options(
        self, config: ModuleType, role: _RoleIn, /
    ) -> _MemoryCo: ...
    def linears_recurrent_controller_options(
        self, config: ModuleType, role: _RoleIn, /
    ) -> _RecurrentCo: ...


class AdaptiveHiddenModelFactory(
    Generic[_Submodule, _MemoryOptions, _RecurrentOptions, _GeneratorOptions]
):
    gate_type: Callable[..., _GateFactory]
    halting_type: Callable[..., _HaltingFactory]
    memory_type: Callable[..., _MemoryFactory]
    recurrent_type: Callable[..., _RecurrentFactory]
    generator_type: Callable[[_GeneratorOptions], StackGenerator]
    layer_template_type: type[AdaptiveLinearLayerConfig]
    _hidden_dim: int
    stack_options: _AdaptiveStack
    layer_controller_options: _Controller
    grouping_config: GroupingConfig | None
    gate_config_factory: _GateFactory
    halting_config_factory: _HaltingFactory
    memory_config_factory: _MemoryFactory
    recurrent_config_factory: _RecurrentFactory
    hidden_adaptive_weight_options: WeightOptions
    hidden_adaptive_bias_options: BiasOptions
    hidden_adaptive_diagonal_options: DiagonalOptions
    hidden_adaptive_mask_options: MaskOptions
    adaptive_generator_stack_config_factory: StackGenerator
    adaptive_augmentation_config: AdaptiveParameterAugmentationConfig

    def __init__(
        self,
        dependencies: _HiddenOptions[
            _Submodule, _MemoryOptions, _RecurrentOptions, _GeneratorOptions
        ],
        config: ModuleType,
        defaults: _HiddenDefaults[
            _Role, _Submodule, _MemoryOptions, _RecurrentOptions, _GeneratorOptions
        ],
        main_role: _Role,
    ) -> None:
        hidden_dim = dependencies.hidden_dim
        stack_options = dependencies.stack_options
        submodule_stack_options = dependencies.submodule_stack_options
        layer_controller_options = dependencies.layer_controller_options
        dynamic_memory_options = dependencies.dynamic_memory_options
        recurrent_controller_options = dependencies.recurrent_controller_options
        hidden_adaptive_weight_options = dependencies.hidden_adaptive_weight_options
        grouping_config = dependencies.grouping_config
        hidden_adaptive_bias_options = dependencies.hidden_adaptive_bias_options
        hidden_adaptive_diagonal_options = dependencies.hidden_adaptive_diagonal_options
        hidden_adaptive_mask_options = dependencies.hidden_adaptive_mask_options
        adaptive_generator_stack_options = dependencies.adaptive_generator_stack_options
        output_dim = dependencies.output_dim

        self._hidden_dim = hidden_dim
        self.stack_options = resolve_options(
            stack_options, defaults.main_layer_stack_options, config
        )
        self.submodule_stack_options = resolve_options(
            submodule_stack_options,
            defaults.linears_submodule_stack_options,
            config,
            main_role,
        )
        self.layer_controller_options = resolve_options(
            layer_controller_options,
            defaults.linears_layer_controller_options,
            config,
            main_role,
        )
        self.dynamic_memory_options = resolve_options(
            dynamic_memory_options,
            defaults.linears_dynamic_memory_options,
            config,
            main_role,
        )
        self.recurrent_controller_options = resolve_options(
            recurrent_controller_options,
            defaults.linears_recurrent_controller_options,
            config,
            main_role,
        )
        self.gate_config_factory = self.gate_type(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
        )
        self.halting_config_factory = self.halting_type(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
            output_dim=output_dim,
        )
        self.memory_config_factory = self.memory_type(
            hidden_dim=self.hidden_dim,
            stack_options=self.stack_options,
            dynamic_memory_options=self.dynamic_memory_options,
            submodule_stack_options=self.submodule_stack_options,
        )
        self.recurrent_config_factory = self.recurrent_type(
            recurrent_controller_options=self.recurrent_controller_options,
            gate_config_factory=self.gate_config_factory,
            halting_config_factory=self.halting_config_factory,
        )
        self.hidden_adaptive_weight_options = resolve_options(
            hidden_adaptive_weight_options,
            defaults.hidden_adaptive_weight_options,
            config,
        )
        self.grouping_config = grouping_config
        self.hidden_adaptive_bias_options = resolve_options(
            hidden_adaptive_bias_options,
            defaults.hidden_adaptive_bias_options,
            config,
        )
        self.hidden_adaptive_diagonal_options = resolve_options(
            hidden_adaptive_diagonal_options,
            defaults.hidden_adaptive_diagonal_options,
            config,
        )
        self.hidden_adaptive_mask_options = resolve_options(
            hidden_adaptive_mask_options,
            defaults.hidden_adaptive_mask_options,
            config,
        )
        self.adaptive_generator_stack_options = resolve_options(
            adaptive_generator_stack_options,
            defaults.adaptive_generator_stack_options,
            config,
        )
        self.adaptive_generator_stack_config_factory = self.generator_type(
            self.adaptive_generator_stack_options
        )
        augmentation = AdaptiveAugmentationFactory(
            self.adaptive_generator_stack_config_factory
        )
        self.adaptive_augmentation_config = augmentation.build(
            grouping=self.grouping_config,
            weight=self.hidden_adaptive_weight_options,
            bias=self.hidden_adaptive_bias_options,
            diagonal=self.hidden_adaptive_diagonal_options,
            mask=self.hidden_adaptive_mask_options,
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def build_hidden_model_config(self) -> LayerStackConfig | RecurrentLayerConfig:
        gate_config = self.gate_config_factory.build_gate_config()
        halting_config = self.halting_config_factory.build_halting_config()
        memory_config = self.memory_config_factory.build_memory_config()
        layer_config = self.__build_layer_config(
            gate_config=gate_config,
            halting_config=halting_config,
        )
        layer_stack_config = self.__build_stack_config(
            memory_config=memory_config,
            layer_config=layer_config,
        )
        hidden_model_config = self.recurrent_config_factory.build_config(
            layer_stack_config
        )
        return hidden_model_config

    def __build_stack_config(
        self,
        *,
        memory_config: DynamicMemoryConfig | None,
        layer_config: LayerConfig,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            hidden_dim=self.hidden_dim,
            num_layers=self.stack_options.num_layers,
            last_layer_bias_option=self.stack_options.last_layer_bias_option,
            apply_output_postprocessing_flag=(
                self.stack_options.apply_output_postprocessing_flag
            ),
            shared_gate_config=self.layer_controller_options.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )

    def __build_layer_config(
        self,
        *,
        gate_config: GateConfig | None,
        halting_config: HaltingConfig | None,
    ) -> LayerConfig:
        layer_model_config = self.__build_layer_model_config()
        residual_config = build_residual_config(
            self.stack_options.residual_connection_option,
            self.stack_options.residual_model_flag,
            self.stack_options.residual_stack_options,
            residual_block_size=self.stack_options.residual_block_size,
            residual_rms_norm_epsilon=self.stack_options.residual_rms_norm_epsilon,
        )
        return LayerConfig(
            activation=self.stack_options.activation,
            layer_norm_position=self.stack_options.layer_norm_position,
            normalization=self.stack_options.normalization,
            residual_config=residual_config,
            dropout_probability=self.stack_options.dropout_probability,
            gate_config=gate_config,
            halting_config=halting_config,
            layer_model_config=layer_model_config,
        )

    def __build_layer_model_config(self) -> AdaptiveLinearLayerConfig:
        return self.layer_template_type(
            bias_flag=self.stack_options.bias_flag,
            adaptive_augmentation_config=self.adaptive_augmentation_config,
        )
