from dataclasses import dataclass
from typing import Protocol

from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
)
from emperor.config import ConfigBase
from emperor.experts import MixtureOfExpertsModelConfig
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.transformer import (
    FeedForwardConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerConfig,
)
from models import _attention_source_memory as _source_memory
from models import _sequence_expansion, _transformer_stack
from models.gpt.expert_linear._control_factory_dependencies import (
    GptControlFactoryDependencies,
)
from models.gpt.expert_linear._decoder_control_config_factory import (
    GptGateConfigFactory,
    GptHaltingConfigFactory,
    GptMemoryConfigFactory,
    GptRecurrentConfigFactory,
)
from models.gpt.expert_linear._linear_layer_config_factory import (
    LinearLayerConfigFactory as GptLinearLayerConfigFactory,
)
from models.gpt.expert_linear.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
)

from ._residual import build_residual_config


class _GptExpertConfigFactory(Protocol):
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
        feed_forward_stack_options: SubmoduleStackOptions,
    ) -> MixtureOfExpertsModelConfig: ...


@dataclass(frozen=True)
class CoreConfigDependencies:
    batch_size: int
    sequence_length: int
    decoder_options: TransformerDecoderOptions
    attention_options: TransformerAttentionOptions
    feed_forward_options: TransformerFeedForwardOptions
    attention_projection_stack_options: SubmoduleStackOptions
    attention_projection_layer_controller_options: LayerControllerOptions | None
    attention_projection_dynamic_memory_options: DynamicMemoryOptions | None
    attention_projection_recurrent_controller_options: RecurrentControllerOptions | None
    feed_forward_stack_options: SubmoduleStackOptions
    feed_forward_layer_controller_options: LayerControllerOptions | None
    feed_forward_dynamic_memory_options: DynamicMemoryOptions | None
    feed_forward_recurrent_controller_options: RecurrentControllerOptions | None
    stack_options: MainLayerStackOptions
    submodule_stack_options: SubmoduleStackOptions | None
    layer_controller_options: LayerControllerOptions | None
    dynamic_memory_options: DynamicMemoryOptions | None
    recurrent_controller_options: RecurrentControllerOptions | None
    linear_layer_config_factory: GptLinearLayerConfigFactory
    attention_projection_linear_layer_config_factory: (
        GptLinearLayerConfigFactory | None
    ) = None
    feed_forward_linear_layer_config_factory: GptLinearLayerConfigFactory | None = None
    expert_config_factory: _GptExpertConfigFactory | None = None


class GptCoreConfigFactory:
    def __init__(self, dependencies: CoreConfigDependencies) -> None:
        self.batch_size = dependencies.batch_size
        self.sequence_length = dependencies.sequence_length
        self.decoder_options = dependencies.decoder_options
        self.hidden_dim = dependencies.decoder_options.hidden_dim
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
        self.decoder_stack_options = dependencies.stack_options
        self.decoder_submodule_stack_options = dependencies.submodule_stack_options
        self.decoder_layer_controller_options = dependencies.layer_controller_options
        self.decoder_dynamic_memory_options = dependencies.dynamic_memory_options
        self.decoder_recurrent_controller_options = (
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

    def build_decoder_config(self):
        decoder_config = self._build_decoder_config()
        return decoder_config

    def _build_decoder_config(self):
        options = self.decoder_options
        layer_model_config = self._build_decoder_layer_config()
        dependencies = self._control_factory_dependencies()
        gate_factory = GptGateConfigFactory(dependencies).build_decoder_factory()
        halting_factory = GptHaltingConfigFactory(dependencies).build_decoder_factory()
        memory_factory = GptMemoryConfigFactory(dependencies).build_decoder_factory()
        halting_config = None
        if halting_factory is not None:
            halting_config = halting_factory.build_halting_config()
        stack_residual_connection_option = self._stack_residual_connection_option()
        stack_residual_model_flag = self._stack_residual_model_flag()
        stack_residual_block_size = self._stack_residual_block_size()
        stack_residual_rms_norm_epsilon = self._stack_residual_rms_norm_epsilon()
        residual_stack_options = None
        if self.decoder_stack_options is not None:
            residual_stack_options = self.decoder_stack_options.residual_stack_options
        residual_config = build_residual_config(
            stack_residual_connection_option,
            stack_residual_model_flag,
            (residual_stack_options),
            residual_block_size=stack_residual_block_size,
            residual_rms_norm_epsilon=stack_residual_rms_norm_epsilon,
        )
        gate_config = None
        if gate_factory is not None:
            gate_config = gate_factory.build_gate_config()
        sequence_expansion_config = _sequence_expansion.build_sequence_expansion_config(
            self.decoder_options, self.hidden_dim
        )
        layer_config = TransformerDecoderBlockLayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=residual_config,
            dropout_probability=0.0,
            gate_config=(gate_config),
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
            shared_memory_config=(shared_memory_config),
            layer_config=layer_config,
        )
        recurrent_factory = GptRecurrentConfigFactory(
            dependencies
        ).build_decoder_factory()
        if recurrent_factory is None:
            return stack_config
        decoder_config = recurrent_factory.build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )
        return decoder_config

    def _build_decoder_layer_config(self) -> TransformerDecoderLayerConfig:
        options = self.decoder_options
        attention_config = self._build_attention_config()
        feed_forward_config = self._build_feed_forward_config()
        residual_config = AdditiveResidualConfig()
        return TransformerDecoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            dropout_probability=options.dropout_probability,
            residual_config=residual_config,
            self_attention_config=attention_config,
            cross_attention_config=None,
            feed_forward_config=feed_forward_config,
        )

    def _build_attention_config(
        self,
    ) -> SelfAttentionConfig | MixtureOfAttentionHeadsConfig:
        decoder_options = self.decoder_options
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
                    _source_memory.build_source_memory_config(attention_options)
                )
                return attention_config
        source_memory_config = _source_memory.build_source_memory_config(
            attention_options
        )
        return _transformer_stack.separate_self_attention(
            batch_size=self.batch_size,
            num_heads=attention_options.num_heads,
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length,
            dropout_probability=decoder_options.dropout_probability,
            causal_attention_mask_flag=True,
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
        recurrent_factory = GptRecurrentConfigFactory(
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
        gate_factory = GptGateConfigFactory(
            dependencies
        ).build_attention_projection_factory()
        halting_factory = GptHaltingConfigFactory(
            dependencies
        ).build_attention_projection_factory()
        memory_factory = GptMemoryConfigFactory(
            dependencies
        ).build_attention_projection_factory()
        controller_options = self.attention_projection_layer_controller_options
        _transformer_stack.apply_enabled_controls(
            stack_config,
            controller_options,
            self.attention_projection_dynamic_memory_options,
            gate_factory,
            halting_factory,
            memory_factory,
        )

    def _build_feed_forward_stack_config(self):
        stack_config = self._build_feed_forward_base_stack_config()
        self._apply_feed_forward_controls(stack_config)
        control_factory_dependencies = self._control_factory_dependencies()
        recurrent_factory = GptRecurrentConfigFactory(
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

    def _apply_feed_forward_controls(self, stack_config) -> None:
        target_stack = self._feed_forward_control_target_stack(stack_config)
        dependencies = self._control_factory_dependencies()
        gate_factory = GptGateConfigFactory(dependencies).build_feed_forward_factory()
        halting_factory = GptHaltingConfigFactory(
            dependencies
        ).build_feed_forward_factory()
        memory_factory = GptMemoryConfigFactory(
            dependencies
        ).build_feed_forward_factory()
        controller_options = self.feed_forward_layer_controller_options
        _transformer_stack.apply_enabled_controls(
            target_stack,
            controller_options,
            self.feed_forward_dynamic_memory_options,
            gate_factory,
            halting_factory,
            memory_factory,
        )

    def _feed_forward_control_target_stack(
        self, stack_config: ConfigBase
    ) -> LayerStackConfig:
        return _transformer_stack.feed_forward_control_stack(stack_config)

    def _stack_residual_connection_option(self) -> type[ResidualConfig]:
        return self.decoder_stack_options.residual_connection_option

    def _stack_residual_model_flag(self) -> bool:
        return self.decoder_stack_options.residual_model_flag

    def _stack_residual_block_size(self) -> int | None:
        return self.decoder_stack_options.residual_block_size

    def _stack_residual_rms_norm_epsilon(self) -> float | None:
        return self.decoder_stack_options.residual_rms_norm_epsilon

    def _stack_last_layer_bias_option(self) -> LastLayerBiasOptions:
        return self.decoder_stack_options.last_layer_bias_option

    def _stack_apply_output_postprocessing_flag(self) -> bool:
        return self.decoder_stack_options.apply_output_postprocessing_flag

    def _shared_gate_config(self):
        if self.decoder_layer_controller_options is None:
            return None
        return self.decoder_layer_controller_options.shared_gate_config

    def _effective_attention_projection_stack_options(self) -> SubmoduleStackOptions:
        return self.attention_projection_stack_options

    def _effective_feed_forward_stack_options(self) -> SubmoduleStackOptions:
        return self.feed_forward_stack_options

    def _control_factory_dependencies(self) -> GptControlFactoryDependencies:
        return GptControlFactoryDependencies(
            hidden_dim=self.hidden_dim,
            decoder_stack_options=self.decoder_stack_options,
            decoder_submodule_stack_options=self.decoder_submodule_stack_options,
            attention_projection_stack_options=(
                self._effective_attention_projection_stack_options()
            ),
            feed_forward_stack_options=self._effective_feed_forward_stack_options(),
            decoder_layer_controller_options=self.decoder_layer_controller_options,
            attention_projection_layer_controller_options=(
                self.attention_projection_layer_controller_options
            ),
            feed_forward_layer_controller_options=(
                self.feed_forward_layer_controller_options
            ),
            decoder_dynamic_memory_options=self.decoder_dynamic_memory_options,
            attention_projection_dynamic_memory_options=(
                self.attention_projection_dynamic_memory_options
            ),
            feed_forward_dynamic_memory_options=(
                self.feed_forward_dynamic_memory_options
            ),
            decoder_recurrent_controller_options=(
                self.decoder_recurrent_controller_options
            ),
            attention_projection_recurrent_controller_options=(
                self.attention_projection_recurrent_controller_options
            ),
            feed_forward_recurrent_controller_options=(
                self.feed_forward_recurrent_controller_options
            ),
        )
