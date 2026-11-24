from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Generic, Protocol, Self, TypeVar, cast

from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
)
from emperor.config import ConfigBase
from emperor.experts import MixtureOfExpertsModelConfig, SoftMoEConfig
from emperor.halting import HaltingConfig
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    GateConfig,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig
from emperor.transformer import (
    FeedForwardConfig,
    SequenceExpansionConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from models import _attention_source_memory as _source_memory
from models import _sequence_expansion, _transformer_stack
from models._linear_stack import LayerStackSettings, LinearStackFactory
from models._residual_stack import ResidualStackSettings, build_residual_config


class _EncoderSettings(Protocol):
    hidden_dim: Final[int]
    num_layers: Final[int]
    layer_norm_position: Final[LayerNormPositionOptions]
    normalization: Final[NormalizationOptions]
    dropout_probability: Final[float]


class _AttentionSettings(Protocol):
    source_memory_flag: Final[bool]
    num_heads: Final[int]
    add_key_value_bias_flag: Final[bool]


class _FeedForwardSettings(Protocol):
    soft_moe_config: Final[SoftMoEConfig | None]


class _EncoderStackSettings(Protocol):
    residual_connection_option: Final[type[ResidualConfig] | None]
    residual_model_flag: Final[bool]
    residual_block_size: Final[int | None]
    residual_rms_norm_epsilon: Final[float | None]
    residual_stack_options: Final[ResidualStackSettings | None]
    last_layer_bias_option: Final[LastLayerBiasOptions]
    apply_output_postprocessing_flag: Final[bool]


class _LayerControls(Protocol):
    stack_gate_flag: Final[bool]
    stack_halting_flag: Final[bool]
    shared_gate_config: Final[GateConfig | None]


class _MemorySettings(Protocol):
    memory_flag: Final[bool]


_Encoder = TypeVar("_Encoder", bound=_EncoderSettings)
_MainStack = TypeVar("_MainStack", bound=_EncoderStackSettings)
_LayerOptions = TypeVar("_LayerOptions", bound=_LayerControls)
_MemoryOptions = TypeVar("_MemoryOptions", bound=_MemorySettings)
_Recurrence = TypeVar("_Recurrence")
_Block = TypeVar("_Block", bound=ConfigBase)
_Control = TypeVar("_Control", covariant=True)
_Dependencies = TypeVar("_Dependencies")
_Stack = TypeVar("_Stack", bound=LayerStackSettings)
_StackIn = TypeVar("_StackIn", bound=LayerStackSettings, contravariant=True)


class _ScopedControl(Protocol[_Control]):
    def build_encoder_factory(self) -> _Control | None: ...
    def build_attention_projection_factory(self) -> _Control | None: ...
    def build_feed_forward_factory(self) -> _Control | None: ...


class _Gate(Protocol):
    def build_gate_config(self) -> GateConfig | None: ...


class _Halting(Protocol):
    def build_halting_config(self) -> HaltingConfig | None: ...


class _Memory(Protocol):
    def build_memory_config(self) -> DynamicMemoryConfig | None: ...


class _Recurrent(Protocol):
    def build_config(
        self,
        block_config: _Block,
        *,
        input_dim: int,
        output_dim: int,
    ) -> _Block | RecurrentLayerConfig: ...


class ExpertFactory(Protocol[_StackIn]):
    def build_attention_config(
        self,
        *,
        batch_size: int,
        hidden_dim: int,
        sequence_length: int,
        projection_model_config: LayerStackConfig | RecurrentLayerConfig,
    ) -> MixtureOfAttentionHeadsConfig | None: ...

    def build_feed_forward_base_stack_config(
        self,
        feed_forward_stack_options: _StackIn,
    ) -> MixtureOfExpertsModelConfig: ...


@dataclass(frozen=True)
class EncoderInputs(
    Generic[_Encoder, _Stack, _MainStack, _LayerOptions, _MemoryOptions, _Recurrence]
):
    batch_size: int
    sequence_length: int
    encoder_options: _Encoder
    attention_options: _AttentionSettings
    feed_forward_options: _FeedForwardSettings
    attention_projection_stack_options: _Stack
    attention_projection_layer_controller_options: _LayerOptions | None
    attention_projection_dynamic_memory_options: _MemoryOptions | None
    attention_projection_recurrent_controller_options: _Recurrence | None
    feed_forward_stack_options: _Stack
    feed_forward_layer_controller_options: _LayerOptions | None
    feed_forward_dynamic_memory_options: _MemoryOptions | None
    feed_forward_recurrent_controller_options: _Recurrence | None
    stack_options: _MainStack
    submodule_stack_options: _Stack | None
    layer_controller_options: _LayerOptions | None
    dynamic_memory_options: _MemoryOptions | None
    recurrent_controller_options: _Recurrence | None
    linear_layer_config_factory: LinearStackFactory
    attention_projection_linear_layer_config_factory: LinearStackFactory | None = None
    feed_forward_linear_layer_config_factory: LinearStackFactory | None = None
    expert_config_factory: ExpertFactory[_Stack] | None = None


@dataclass(frozen=True)
class EncoderControls(
    Generic[_Stack, _MainStack, _LayerOptions, _MemoryOptions, _Recurrence]
):
    hidden_dim: int
    encoder_stack_options: _MainStack
    encoder_submodule_stack_options: _Stack | None
    attention_projection_stack_options: _Stack
    feed_forward_stack_options: _Stack
    encoder_layer_controller_options: _LayerOptions | None
    attention_projection_layer_controller_options: _LayerOptions | None
    feed_forward_layer_controller_options: _LayerOptions | None
    encoder_dynamic_memory_options: _MemoryOptions | None
    attention_projection_dynamic_memory_options: _MemoryOptions | None
    feed_forward_dynamic_memory_options: _MemoryOptions | None
    encoder_recurrent_controller_options: _Recurrence | None
    attention_projection_recurrent_controller_options: _Recurrence | None
    feed_forward_recurrent_controller_options: _Recurrence | None

    @classmethod
    def from_encoder(
        cls,
        encoder: VisionEncoderBuilder[
            _Dependencies,
            _Stack,
            _Encoder,
            _MainStack,
            _LayerOptions,
            _MemoryOptions,
            _Recurrence,
        ],
    ) -> Self:
        hidden_dim = encoder.hidden_dim
        main_stack = encoder.encoder_stack_options
        submodule_stack = encoder.encoder_submodule_stack_options
        attention_stack, feed_forward_stack = encoder.control_stack_options()
        return cls(
            hidden_dim=hidden_dim,
            encoder_stack_options=main_stack,
            encoder_submodule_stack_options=submodule_stack,
            attention_projection_stack_options=attention_stack,
            feed_forward_stack_options=feed_forward_stack,
            encoder_layer_controller_options=encoder.encoder_layer_controller_options,
            attention_projection_layer_controller_options=(
                encoder.attention_projection_layer_controller_options
            ),
            feed_forward_layer_controller_options=(
                encoder.feed_forward_layer_controller_options
            ),
            encoder_dynamic_memory_options=encoder.encoder_dynamic_memory_options,
            attention_projection_dynamic_memory_options=(
                encoder.attention_projection_dynamic_memory_options
            ),
            feed_forward_dynamic_memory_options=(
                encoder.feed_forward_dynamic_memory_options
            ),
            encoder_recurrent_controller_options=(
                encoder.encoder_recurrent_controller_options
            ),
            attention_projection_recurrent_controller_options=(
                encoder.attention_projection_recurrent_controller_options
            ),
            feed_forward_recurrent_controller_options=(
                encoder.feed_forward_recurrent_controller_options
            ),
        )


class VisionEncoderBuilder(
    Generic[
        _Dependencies,
        _Stack,
        _Encoder,
        _MainStack,
        _LayerOptions,
        _MemoryOptions,
        _Recurrence,
    ],
):
    _control_inputs: Callable[[Self], _Dependencies]
    _gate: Callable[[_Dependencies], _ScopedControl[_Gate]]
    _halting: Callable[[_Dependencies], _ScopedControl[_Halting]]
    _memory: Callable[[_Dependencies], _ScopedControl[_Memory]]
    _recurrent: Callable[[_Dependencies], _ScopedControl[_Recurrent]]

    def __init__(
        self,
        dependencies: EncoderInputs[
            _Encoder, _Stack, _MainStack, _LayerOptions, _MemoryOptions, _Recurrence
        ],
    ) -> None:
        self.batch_size = dependencies.batch_size
        self.sequence_length = dependencies.sequence_length
        self.encoder_options = dependencies.encoder_options
        self.hidden_dim = dependencies.encoder_options.hidden_dim
        self.attention_options = dependencies.attention_options
        self.feed_forward_options = dependencies.feed_forward_options
        self.attention_projection_stack_options = (
            dependencies.attention_projection_stack_options
        )
        self.attention_projection_layer_controller_options = (
            dependencies.attention_projection_layer_controller_options
        )
        self.attention_projection_dynamic_memory_options = (
            dependencies.attention_projection_dynamic_memory_options
        )
        self.attention_projection_recurrent_controller_options = (
            dependencies.attention_projection_recurrent_controller_options
        )
        self.feed_forward_stack_options = dependencies.feed_forward_stack_options
        self.feed_forward_layer_controller_options = (
            dependencies.feed_forward_layer_controller_options
        )
        self.feed_forward_dynamic_memory_options = (
            dependencies.feed_forward_dynamic_memory_options
        )
        self.feed_forward_recurrent_controller_options = (
            dependencies.feed_forward_recurrent_controller_options
        )
        self.encoder_stack_options = dependencies.stack_options
        self.encoder_submodule_stack_options = dependencies.submodule_stack_options
        self.encoder_layer_controller_options = dependencies.layer_controller_options
        self.encoder_dynamic_memory_options = dependencies.dynamic_memory_options
        self.encoder_recurrent_controller_options = (
            dependencies.recurrent_controller_options
        )
        self.linear_layer_config_factory = dependencies.linear_layer_config_factory
        self.attention_projection_linear_layer_config_factory = (
            dependencies.attention_projection_linear_layer_config_factory
            or self.linear_layer_config_factory
        )
        self.feed_forward_linear_layer_config_factory = (
            dependencies.feed_forward_linear_layer_config_factory
            or self.linear_layer_config_factory
        )
        self.expert_config_factory = dependencies.expert_config_factory

    def control_stack_options(self) -> tuple[_Stack, _Stack]:
        attention_stack = self._effective_attention_projection_stack_options()
        feed_forward_stack = self._effective_feed_forward_stack_options()
        return attention_stack, feed_forward_stack

    def _control_factory_dependencies(self) -> _Dependencies:
        return self._control_inputs(self)

    def _sequence_expansion_config(self) -> SequenceExpansionConfig | None:
        encoder = cast(
            _sequence_expansion.SequenceExpansionOptions, self.encoder_options
        )
        return _sequence_expansion.build_sequence_expansion_config(
            encoder, self.hidden_dim
        )

    def build_encoder_config(self) -> LayerStackConfig | RecurrentLayerConfig:
        return self._build_encoder_config()

    def _build_encoder_config(self) -> LayerStackConfig | RecurrentLayerConfig:
        options = self.encoder_options
        layer_model_config = self._build_encoder_layer_config()
        dependencies = self._control_factory_dependencies()
        gate_factory = self._gate(dependencies).build_encoder_factory()
        halting_factory = self._halting(dependencies).build_encoder_factory()
        memory_factory = self._memory(dependencies).build_encoder_factory()
        halting_config = None
        if halting_factory is not None:
            halting_config = halting_factory.build_halting_config()
        stack_residual_connection_option = self._stack_residual_connection_option()
        stack_residual_model_flag = self._stack_residual_model_flag()
        stack_residual_block_size = self._stack_residual_block_size()
        stack_residual_rms_norm_epsilon = self._stack_residual_rms_norm_epsilon()
        residual_config = build_residual_config(
            stack_residual_connection_option,
            stack_residual_model_flag,
            self.encoder_stack_options.residual_stack_options,
            residual_block_size=stack_residual_block_size,
            residual_rms_norm_epsilon=stack_residual_rms_norm_epsilon,
        )
        gate_config = None
        if gate_factory is not None:
            gate_config = gate_factory.build_gate_config()
        sequence_expansion_config = self._sequence_expansion_config()
        layer_config = TransformerEncoderBlockLayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=residual_config,
            dropout_probability=0.0,
            gate_config=gate_config,
            halting_config=None,
            layer_model_config=layer_model_config,
            sequence_expansion_config=sequence_expansion_config,
        )
        last_layer_bias_option = self._stack_last_layer_bias_option()
        apply_output_postprocessing_flag = (
            self._stack_apply_output_postprocessing_flag()
        )
        shared_gate_config = self._shared_gate_config()
        shared_memory_config = None
        if memory_factory is not None:
            shared_memory_config = memory_factory.build_memory_config()
        stack_config = LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
            shared_gate_config=shared_gate_config,
            shared_halting_config=halting_config,
            shared_memory_config=shared_memory_config,
            layer_config=layer_config,
        )
        recurrent_factory = self._recurrent(dependencies).build_encoder_factory()
        if recurrent_factory is None:
            return stack_config
        encoder_config = recurrent_factory.build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )
        return encoder_config

    def _build_encoder_layer_config(self) -> TransformerEncoderLayerConfig:
        options = self.encoder_options
        attention_config = self._build_attention_config()
        feed_forward_config = self._build_feed_forward_config()
        residual_config = AdditiveResidualConfig()
        return TransformerEncoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            dropout_probability=options.dropout_probability,
            residual_config=residual_config,
            attention_config=attention_config,
            feed_forward_config=feed_forward_config,
        )

    def _build_attention_config(
        self,
    ) -> SelfAttentionConfig | MixtureOfAttentionHeadsConfig:
        encoder_options = self.encoder_options
        attention_options = self.attention_options
        source_memory_flag = attention_options.source_memory_flag
        projection_model_config = self._build_attention_projection_stack_config()
        if self.expert_config_factory is not None:
            attention_config = self.expert_config_factory.build_attention_config(
                batch_size=self.batch_size,
                hidden_dim=self.hidden_dim,
                sequence_length=self.sequence_length,
                projection_model_config=projection_model_config,
            )
            if attention_config is not None:
                attention_config.source_memory_flag = source_memory_flag
                attention_config.source_memory_config = (
                    _source_memory.build_source_memory_config(
                        attention_options, attention_config.experts_config
                    )
                )
                return attention_config
        source_memory_config = _source_memory.build_source_memory_config(
            attention_options, projection_model_config
        )
        return _transformer_stack.separate_self_attention(
            batch_size=self.batch_size,
            num_heads=attention_options.num_heads,
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length,
            dropout_probability=encoder_options.dropout_probability,
            causal_attention_mask_flag=False,
            add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
            source_memory_flag=source_memory_flag,
            source_memory_config=source_memory_config,
            projection_model_config=projection_model_config,
        )

    def _build_feed_forward_config(self) -> FeedForwardConfig:
        feed_forward_stack_config = self._build_feed_forward_stack_config()
        return _transformer_stack.feed_forward_config(
            feed_forward_stack_config,
            self.feed_forward_options.soft_moe_config,
            self.hidden_dim,
        )

    def _build_attention_projection_stack_config(self):
        stack_config = self._build_attention_projection_base_stack_config()
        self._apply_attention_projection_controls(stack_config)
        control_factory_dependencies = self._control_factory_dependencies()
        recurrent_factory = self._recurrent(
            control_factory_dependencies
        ).build_attention_projection_factory()
        if recurrent_factory is None:
            return stack_config
        attention_projection_stack_config = recurrent_factory.build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )
        return attention_projection_stack_config

    def _build_attention_projection_base_stack_config(self) -> LayerStackConfig:
        options = self._effective_attention_projection_stack_options()
        factory = self.attention_projection_linear_layer_config_factory
        attention_projection_base_stack_config = factory.build_submodule_stack_config(
            options
        )
        return attention_projection_base_stack_config

    def _apply_attention_projection_controls(
        self,
        stack_config: LayerStackConfig,
    ) -> None:
        dependencies = self._control_factory_dependencies()
        gate_factory = self._gate(dependencies).build_attention_projection_factory()
        halting_factory = self._halting(
            dependencies
        ).build_attention_projection_factory()
        memory_factory = self._memory(dependencies).build_attention_projection_factory()
        _transformer_stack.apply_enabled_controls(
            stack_config,
            self.attention_projection_layer_controller_options,
            self.attention_projection_dynamic_memory_options,
            gate_factory,
            halting_factory,
            memory_factory,
        )

    def _build_feed_forward_stack_config(self):
        stack_config = self._build_feed_forward_base_stack_config()
        self._apply_feed_forward_controls(stack_config)
        control_factory_dependencies = self._control_factory_dependencies()
        recurrent_factory = self._recurrent(
            control_factory_dependencies
        ).build_feed_forward_factory()
        if recurrent_factory is None:
            return stack_config
        feed_forward_stack_config = recurrent_factory.build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )
        return feed_forward_stack_config

    def _build_feed_forward_base_stack_config(self):
        options = self._effective_feed_forward_stack_options()
        if self.expert_config_factory is not None:
            feed_forward_base_stack_config = (
                self.expert_config_factory.build_feed_forward_base_stack_config(options)
            )
            return feed_forward_base_stack_config
        factory = self.feed_forward_linear_layer_config_factory
        feed_forward_base_stack_config = factory.build_submodule_stack_config(options)
        return feed_forward_base_stack_config

    def _apply_feed_forward_controls(
        self, stack_config: LayerStackConfig | MixtureOfExpertsModelConfig
    ) -> None:
        target_stack = self._feed_forward_control_target_stack(stack_config)
        dependencies = self._control_factory_dependencies()
        gate_factory = self._gate(dependencies).build_feed_forward_factory()
        halting_factory = self._halting(dependencies).build_feed_forward_factory()
        memory_factory = self._memory(dependencies).build_feed_forward_factory()
        _transformer_stack.apply_enabled_controls(
            target_stack,
            self.feed_forward_layer_controller_options,
            self.feed_forward_dynamic_memory_options,
            gate_factory,
            halting_factory,
            memory_factory,
        )

    def _feed_forward_control_target_stack(
        self, stack_config: ConfigBase
    ) -> LayerStackConfig:
        return _transformer_stack.feed_forward_control_stack(stack_config)

    def _stack_residual_connection_option(self) -> type[ResidualConfig] | None:
        return self.encoder_stack_options.residual_connection_option

    def _stack_residual_model_flag(self) -> bool:
        return self.encoder_stack_options.residual_model_flag

    def _stack_residual_block_size(self) -> int | None:
        return self.encoder_stack_options.residual_block_size

    def _stack_residual_rms_norm_epsilon(self) -> float | None:
        return self.encoder_stack_options.residual_rms_norm_epsilon

    def _stack_last_layer_bias_option(self) -> LastLayerBiasOptions:
        return self.encoder_stack_options.last_layer_bias_option

    def _stack_apply_output_postprocessing_flag(self) -> bool:
        return self.encoder_stack_options.apply_output_postprocessing_flag

    def _shared_gate_config(self):
        if self.encoder_layer_controller_options is None:
            return None
        return self.encoder_layer_controller_options.shared_gate_config

    def _effective_attention_projection_stack_options(self) -> _Stack:
        return self.attention_projection_stack_options

    def _effective_feed_forward_stack_options(self) -> _Stack:
        return self.feed_forward_stack_options
