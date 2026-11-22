from collections.abc import Callable
from typing import Final, Generic, Protocol, TypeVar, cast, overload

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentCompositionConfig,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import (
    AttentionDynamicMemoryConfig,
    DynamicMemoryConfig,
    MemoryPositionOptions,
)
from models import _controller_stack

_Model = TypeVar("_Model", bound=ConfigBase)
_Source = TypeVar("_Source")
_SourceCo = TypeVar("_SourceCo", covariant=True)
_Defaults = TypeVar("_Defaults")
_DefaultsCo = TypeVar("_DefaultsCo", covariant=True)
_ResidualOptions = TypeVar("_ResidualOptions")
_ResidualIn = TypeVar("_ResidualIn", contravariant=True)


class _BuildResidual(Protocol[_ResidualIn]):
    def __call__(
        self,
        selector: type[ResidualConfig] | None,
        enabled: bool,
        options: _ResidualIn,
        /,
        *,
        residual_block_size: int | None,
        residual_rms_norm_epsilon: float | None,
    ) -> ResidualConfig | None: ...


class StackSettings(Protocol):
    hidden_dim: Final[int]
    num_layers: Final[int]
    last_layer_bias_option: Final[LastLayerBiasOptions]
    apply_output_postprocessing_flag: Final[bool]
    activation: Final[ActivationOptions]
    layer_norm_position: Final[LayerNormPositionOptions]
    normalization: Final[NormalizationOptions]
    residual_connection_option: Final[type[ResidualConfig] | None]
    residual_model_flag: Final[bool]
    residual_block_size: Final[int | None]
    residual_rms_norm_epsilon: Final[float | None]
    dropout_probability: Final[float]
    bias_flag: Final[bool]


class DynamicMemorySettings(Protocol[_SourceCo]):
    memory_flag: Final[bool]
    memory_option: Final[type[DynamicMemoryConfig]]
    memory_position_option: Final[MemoryPositionOptions]
    memory_test_time_training_learning_rate: Final[float | None]
    memory_test_time_training_num_inner_steps: Final[int | None]
    memory_stack_options: Final[_SourceCo]
    memory_num_memory_slots: Final[int]


class LayerControllerSettings(Protocol[_SourceCo]):
    stack_gate_flag: Final[bool]
    gate_option: Final[LayerGateOptions | None]
    gate_activation: Final[ActivationOptions | None]
    gate_stack_options: Final[_SourceCo]
    stack_halting_flag: Final[bool]
    halting_option: Final[type[HaltingConfig]]
    halting_threshold: Final[float | None]
    halting_dropout: Final[float]
    halting_hidden_state_mode: Final[HaltingHiddenStateModeOptions]
    halting_stack_options: Final[_SourceCo]
    shared_gate_config: Final[GateConfig | None]


class RecurrentControllerSettings(Protocol[_SourceCo]):
    recurrent_flag: Final[bool]
    recurrent_max_steps: Final[int]
    recurrent_layer_norm_position: Final[LayerNormPositionOptions]
    recurrent_normalization: Final[NormalizationOptions]
    recurrent_stack_gate_flag: Final[bool]
    recurrent_gate_option: Final[LayerGateOptions | None]
    recurrent_gate_activation: Final[ActivationOptions | None]
    recurrent_gate_stack_options: Final[_SourceCo]
    recurrent_stack_halting_flag: Final[bool]
    recurrent_halting_option: Final[type[HaltingConfig]]
    recurrent_halting_threshold: Final[float | None]
    recurrent_halting_dropout: Final[float]
    recurrent_halting_hidden_state_mode: Final[HaltingHiddenStateModeOptions]
    recurrent_halting_stack_options: Final[_SourceCo]


class PathSettings(Protocol[_SourceCo, _DefaultsCo]):
    stack_options: Final[_DefaultsCo]
    layer_controller_options: Final[LayerControllerSettings[_SourceCo]]
    dynamic_memory_options: Final[DynamicMemorySettings[_SourceCo]]
    recurrent_controller_options: Final[RecurrentControllerSettings[_SourceCo]]


class TransformerSubmoduleFactory(Generic[_Source, _Defaults, _ResidualOptions]):
    def __init__(
        self,
        resolve_stack: Callable[[_Source, _Defaults], StackSettings],
        build_residual: _BuildResidual[_ResidualOptions],
    ) -> None:
        self.__resolve_stack = resolve_stack
        self.__build_residual = build_residual

    def __controller_stack(
        self,
        source: _Source,
        defaults: _Defaults,
        residual_stack_options: _ResidualOptions,
        *,
        output_dim: int | None = None,
    ) -> LayerStackConfig:
        options = self.__resolve_stack(source, defaults)
        residual_config = self.__build_residual(
            options.residual_connection_option,
            options.residual_model_flag,
            residual_stack_options,
            residual_block_size=options.residual_block_size,
            residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
        )
        layer_model_config = LinearLayerConfig(bias_flag=options.bias_flag)
        layer_config = _controller_stack.build_controller_layer(
            options, layer_model_config, residual_config
        )
        last_layer_bias = options.last_layer_bias_option
        if output_dim is not None:
            last_layer_bias = LastLayerBiasOptions.DISABLED
        return LayerStackConfig(
            hidden_dim=options.hidden_dim,
            output_dim=output_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=last_layer_bias,
            apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
            layer_config=layer_config,
        )

    def __halting_config(
        self,
        source: _Source,
        stack_defaults: _Defaults,
        residual_stack_options: _ResidualOptions,
        *,
        implementation: type[HaltingConfig],
        threshold: float | None,
        dropout: float,
        hidden_state_mode: HaltingHiddenStateModeOptions,
    ) -> HaltingConfig:
        halting_gate_config = self.__controller_stack(
            source, stack_defaults, residual_stack_options, output_dim=2
        )
        return implementation(
            threshold=threshold,
            min_steps=1,
            ponder_cost_weight=1.0,
            dropout_probability=dropout,
            hidden_state_mode=hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )

    def __configure_layer_controls(
        self,
        control_stack: LayerStackConfig,
        options: LayerControllerSettings[_Source],
        stack_defaults: _Defaults,
        residual_stack_options: _ResidualOptions,
    ) -> None:
        gate_config = None
        if options.stack_gate_flag:
            gate_model_config = self.__controller_stack(
                options.gate_stack_options, stack_defaults, residual_stack_options
            )
            gate_config = GateConfig(
                option=options.gate_option,
                activation=options.gate_activation,
                model_config=gate_model_config,
            )
        layer = cast(LayerConfig, control_stack.layer_config)
        layer.gate_config = gate_config
        layer.halting_config = None

        halting_config = None
        if options.stack_halting_flag:
            halting_config = self.__halting_config(
                options.halting_stack_options,
                stack_defaults,
                residual_stack_options,
                implementation=options.halting_option,
                threshold=options.halting_threshold,
                dropout=options.halting_dropout,
                hidden_state_mode=options.halting_hidden_state_mode,
            )
        control_stack.shared_halting_config = halting_config
        control_stack.shared_gate_config = options.shared_gate_config

    def __recurrent_controls(
        self,
        options: RecurrentControllerSettings[_Source],
        stack_defaults: _Defaults,
        residual_stack_options: _ResidualOptions,
    ) -> tuple[GateConfig | None, HaltingConfig | None]:
        gate_config = None
        if options.recurrent_stack_gate_flag:
            gate_model_config = self.__controller_stack(
                options.recurrent_gate_stack_options,
                stack_defaults,
                residual_stack_options,
            )
            gate_config = GateConfig(
                option=options.recurrent_gate_option,
                activation=options.recurrent_gate_activation,
                model_config=gate_model_config,
            )
        halting_config = None
        if options.recurrent_stack_halting_flag:
            halting_config = self.__halting_config(
                options.recurrent_halting_stack_options,
                stack_defaults,
                residual_stack_options,
                implementation=options.recurrent_halting_option,
                threshold=options.recurrent_halting_threshold,
                dropout=options.recurrent_halting_dropout,
                hidden_state_mode=options.recurrent_halting_hidden_state_mode,
            )
        return gate_config, halting_config

    def __memory_config(
        self,
        options: DynamicMemorySettings[_Source],
        stack_defaults: _Defaults,
        residual_stack_options: _ResidualOptions,
        *,
        model_dim: int,
    ) -> DynamicMemoryConfig | None:
        if not options.memory_flag:
            return None
        model_config = self.__controller_stack(
            options.memory_stack_options,
            stack_defaults,
            residual_stack_options,
        )
        config = options.memory_option(
            input_dim=model_dim,
            output_dim=model_dim,
            memory_position_option=options.memory_position_option,
            test_time_training_learning_rate=(
                options.memory_test_time_training_learning_rate
            ),
            test_time_training_num_inner_steps=(
                options.memory_test_time_training_num_inner_steps
            ),
            model_config=model_config,
        )

        if isinstance(config, AttentionDynamicMemoryConfig):
            config.num_memory_slots = options.memory_num_memory_slots
        return config

    @overload
    def configure(
        self,
        model_config: _Model,
        *,
        control_stack: LayerStackConfig,
        path_options: PathSettings[_Source, _Defaults],
        model_dim: int,
        residual_stack_options: _ResidualOptions,
    ) -> _Model | RecurrentLayerConfig: ...

    @overload
    def configure(
        self,
        model_config: _Model,
        *,
        control_stack: LayerStackConfig,
        path_options: PathSettings[_Source, _Defaults],
        model_dim: int,
        residual_stack_options: _ResidualOptions,
        wrap_recurrent: Callable[
            [GateConfig | None, HaltingConfig | None], RecurrentCompositionConfig
        ],
    ) -> _Model | RecurrentCompositionConfig: ...

    def configure(
        self,
        model_config: _Model,
        *,
        control_stack: LayerStackConfig,
        path_options: PathSettings[_Source, _Defaults],
        model_dim: int,
        residual_stack_options: _ResidualOptions,
        wrap_recurrent: Callable[
            [GateConfig | None, HaltingConfig | None], RecurrentCompositionConfig
        ]
        | None = None,
    ) -> _Model | RecurrentCompositionConfig:
        stack_defaults = path_options.stack_options
        self.__configure_layer_controls(
            control_stack,
            path_options.layer_controller_options,
            stack_defaults,
            residual_stack_options,
        )
        memory_config = self.__memory_config(
            path_options.dynamic_memory_options,
            stack_defaults,
            residual_stack_options,
            model_dim=model_dim,
        )
        control_stack.shared_memory_config = memory_config
        recurrent = path_options.recurrent_controller_options
        if not recurrent.recurrent_flag:
            return model_config
        recurrent_gate_config, recurrent_halting_config = self.__recurrent_controls(
            recurrent, stack_defaults, residual_stack_options
        )
        if wrap_recurrent is not None:
            return wrap_recurrent(recurrent_gate_config, recurrent_halting_config)
        return RecurrentLayerConfig(
            input_dim=model_dim,
            output_dim=model_dim,
            max_steps=recurrent.recurrent_max_steps,
            gradient_transition_count=None,
            no_gradient_transition_count=None,
            initial_iterations=recurrent.recurrent_max_steps,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=recurrent.recurrent_layer_norm_position,
            recurrent_normalization=recurrent.recurrent_normalization,
            block_config=model_config,
            gate_config=recurrent_gate_config,
            residual_config=None,
            halting_config=recurrent_halting_config,
            memory_config=None,
        )
