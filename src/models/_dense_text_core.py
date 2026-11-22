from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final, Protocol, cast

from emperor.attention import SelfAttentionConfig
from emperor.config import ConfigBase
from emperor.experts import SoftMoEConfig
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
from emperor.transformer import (
    FeedForwardConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerEncoderBlockLayerConfig,
)
from models import _attention_source_memory as _source_memory
from models import _sequence_expansion, _transformer_stack
from models._linear_stack import LayerStackSettings, LinearStackFactory
from models._residual_stack import ResidualStackSettings, build_residual_config
from models._stack_controllers import (
    GateConstructor,
    HaltingConstructor,
    MemoryConstructor,
    RecurrentConstructor,
)


class TextStageSettings(Protocol):
    num_layers: Final[int]
    dropout_probability: Final[float]


class _Attention(Protocol):
    num_heads: Final[int]
    source_memory_flag: Final[bool]
    add_key_value_bias_flag: Final[bool]


class _FeedForward(Protocol):
    soft_moe_config: Final[SoftMoEConfig | None]


class _MainStack(Protocol):
    residual_connection_option: Final[type[ResidualConfig] | None]
    residual_model_flag: Final[bool]
    residual_block_size: Final[int | None]
    residual_rms_norm_epsilon: Final[float | None]
    residual_stack_options: Final[ResidualStackSettings | None]
    last_layer_bias_option: Final[LastLayerBiasOptions]
    apply_output_postprocessing_flag: Final[bool]


class _LayerControls(Protocol):
    shared_gate_config: Final[GateConfig | None]


class _Gate(Protocol):
    def build_gate_config(self) -> GateConfig | None: ...


class _Halting(Protocol):
    def build_halting_config(self) -> HaltingConfig | None: ...


class _Memory(Protocol):
    def build_memory_config(self) -> DynamicMemoryConfig | None: ...


class _Recurrence(Protocol):
    def build_config(
        self, block_config: LayerStackConfig, *, input_dim: int, output_dim: int
    ) -> LayerStackConfig | RecurrentLayerConfig: ...


class DenseTextCore(ABC):
    batch_size: int
    hidden_dim: int
    sequence_length: int
    stack_options: _MainStack
    attention_options: _Attention
    feed_forward_options: _FeedForward
    linear_layer_config_factory: LinearStackFactory
    submodule_stack_options: LayerStackSettings
    layer_controller_options: _LayerControls
    dynamic_memory_options: object
    recurrent_controller_options: object
    attention_projection_stack_options: LayerStackSettings
    attention_projection_layer_controller_options: _LayerControls
    attention_projection_dynamic_memory_options: object
    attention_projection_recurrent_controller_options: object
    feed_forward_stack_options: LayerStackSettings
    feed_forward_layer_controller_options: _LayerControls
    feed_forward_dynamic_memory_options: object
    feed_forward_recurrent_controller_options: object
    _gate: GateConstructor[_Gate]
    _halting: HaltingConstructor[_Halting]
    _memory: MemoryConstructor[_Memory]
    _recurrent: RecurrentConstructor[_Recurrence]
    _block: (
        type[TransformerDecoderBlockLayerConfig]
        | type[TransformerEncoderBlockLayerConfig]
    )

    @property
    @abstractmethod
    def _stage_options(self) -> TextStageSettings: ...

    @abstractmethod
    def _causal_attention(self, options: TextStageSettings) -> bool: ...

    @abstractmethod
    def _build_stage_layer(self) -> ConfigBase: ...

    def _build_stage_stack(self) -> LayerStackConfig | RecurrentLayerConfig:
        stage_layer_config = self._build_stage_layer()
        gate_factory = self.__main_gate_factory()
        halting_factory = self.__main_halting_factory()
        memory_factory = self.__main_memory_factory()
        halting_config = halting_factory.build_halting_config()
        residual_config = build_residual_config(
            self.stack_options.residual_connection_option,
            self.stack_options.residual_model_flag,
            self.stack_options.residual_stack_options,
            residual_block_size=self.stack_options.residual_block_size,
            residual_rms_norm_epsilon=self.stack_options.residual_rms_norm_epsilon,
        )
        gate_config = gate_factory.build_gate_config()
        sequence_expansion_config = _sequence_expansion.build_sequence_expansion_config(
            cast(_sequence_expansion.SequenceExpansionOptions, self._stage_options),
            self.hidden_dim,
        )
        layer_config = self._block(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=residual_config,
            dropout_probability=0.0,
            gate_config=gate_config,
            halting_config=None,
            layer_model_config=stage_layer_config,
            sequence_expansion_config=sequence_expansion_config,
        )
        shared_memory_config = memory_factory.build_memory_config()
        stack_config = LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self._stage_options.num_layers,
            last_layer_bias_option=self.stack_options.last_layer_bias_option,
            apply_output_postprocessing_flag=(
                self.stack_options.apply_output_postprocessing_flag
            ),
            shared_gate_config=self.layer_controller_options.shared_gate_config,
            shared_halting_config=halting_config,
            shared_memory_config=shared_memory_config,
            layer_config=layer_config,
        )
        stage_config = self.__main_recurrent_factory().build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )
        return stage_config

    def _build_attention_config(self) -> SelfAttentionConfig:
        stage_options = self._stage_options
        attention_options = self.attention_options
        source_memory_flag = attention_options.source_memory_flag
        source_memory_config = _source_memory.build_source_memory_config(
            attention_options
        )
        base_stack = self.linear_layer_config_factory.build_submodule_stack_config(
            self.attention_projection_stack_options
        )
        projection_model_config = self.__apply_projection_controls(
            base_stack,
            self.attention_projection_stack_options,
            self.attention_projection_layer_controller_options,
            self.attention_projection_recurrent_controller_options,
            self.attention_projection_dynamic_memory_options,
        )
        return _transformer_stack.separate_self_attention(
            batch_size=self.batch_size,
            num_heads=attention_options.num_heads,
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length,
            dropout_probability=stage_options.dropout_probability,
            causal_attention_mask_flag=self._causal_attention(stage_options),
            add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
            source_memory_flag=source_memory_flag,
            source_memory_config=source_memory_config,
            projection_model_config=projection_model_config,
        )

    def _build_feed_forward_config(self) -> FeedForwardConfig:
        base_stack = self.linear_layer_config_factory.build_submodule_stack_config(
            self.feed_forward_stack_options
        )
        stack_config = self.__apply_projection_controls(
            base_stack,
            self.feed_forward_stack_options,
            self.feed_forward_layer_controller_options,
            self.feed_forward_recurrent_controller_options,
            self.feed_forward_dynamic_memory_options,
        )
        return _transformer_stack.feed_forward_config(
            stack_config, self.feed_forward_options.soft_moe_config, self.hidden_dim
        )

    def __apply_projection_controls(
        self,
        stack_config: LayerStackConfig,
        stack: LayerStackSettings,
        layer_controls: _LayerControls,
        recurrence: object,
        memory: object,
    ) -> LayerStackConfig | RecurrentLayerConfig:
        def gate_factory() -> _Gate:
            return self._gate(
                layer_controller_options=layer_controls,
                recurrent_controller_options=recurrence,
                submodule_stack_options=stack,
                recurrent_stack_inherits_gate_stack=False,
            )

        def halting_factory() -> _Halting:
            return self._halting(
                layer_controller_options=layer_controls,
                recurrent_controller_options=recurrence,
                submodule_stack_options=stack,
                output_dim=self.hidden_dim,
                halting_stack_defaults=stack,
                recurrent_stack_inherits_halting_stack=False,
            )

        gate = gate_factory()
        halting = halting_factory()
        memory_factory = self._memory(
            hidden_dim=self.hidden_dim,
            stack_options=self.stack_options,
            dynamic_memory_options=memory,
            submodule_stack_options=stack,
        )
        layer_config = cast(LayerConfig, stack_config.layer_config)
        layer_config.gate_config = gate.build_gate_config()
        layer_config.halting_config = halting.build_halting_config()
        stack_config.shared_gate_config = layer_controls.shared_gate_config
        stack_config.shared_memory_config = memory_factory.build_memory_config()
        recurrent_gate = gate_factory()
        recurrent_halting = halting_factory()
        recurrent = self._recurrent(
            recurrent_controller_options=recurrence,
            gate_config_factory=recurrent_gate,
            halting_config_factory=recurrent_halting,
        )
        return recurrent.build_config(
            stack_config, input_dim=self.hidden_dim, output_dim=self.hidden_dim
        )

    def __main_gate_factory(self) -> _Gate:
        return self._gate(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
        )

    def __main_halting_factory(self) -> _Halting:
        return self._halting(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
            output_dim=self.hidden_dim,
        )

    def __main_memory_factory(self) -> _Memory:
        return self._memory(
            hidden_dim=self.hidden_dim,
            stack_options=self.stack_options,
            dynamic_memory_options=self.dynamic_memory_options,
            submodule_stack_options=self.submodule_stack_options,
        )

    def __main_recurrent_factory(self) -> _Recurrence:
        gate_config_factory = self.__main_gate_factory()
        halting_config_factory = self.__main_halting_factory()
        return self._recurrent(
            recurrent_controller_options=self.recurrent_controller_options,
            gate_config_factory=gate_config_factory,
            halting_config_factory=halting_config_factory,
        )
