from dataclasses import dataclass

from emperor.attention import (
    SelfAttentionConfig,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
)
from emperor.transformer import (
    FeedForwardConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from models import _attention_source_memory as _source_memory
from models import _sequence_expansion, _transformer_stack
from models.bert.linear_adaptive._control_factory_dependencies import (
    BertControlFactoryDependencies,
)
from models.bert.linear_adaptive._encoder_control_config_factory import (
    BertGateConfigFactory,
    BertHaltingConfigFactory,
    BertMemoryConfigFactory,
    BertRecurrentConfigFactory,
)
from models.bert.linear_adaptive._linear_layer_config_factory import (
    LinearLayerConfigFactory,
)
from models.bert.linear_adaptive.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
)

from ._residual import build_residual_config


@dataclass(frozen=True)
class CoreConfigDependencies:
    batch_size: int
    sequence_length: int
    encoder_options: TransformerEncoderOptions
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
    linear_layer_config_factory: LinearLayerConfigFactory
    attention_projection_linear_layer_config_factory: (
        LinearLayerConfigFactory | None
    ) = None
    feed_forward_linear_layer_config_factory: LinearLayerConfigFactory | None = None


class BertCoreConfigFactory:
    def __init__(self, dependencies: CoreConfigDependencies) -> None:
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

    def build_encoder_config(self):
        dependencies = self._control_factory_dependencies()
        gate_factory = BertGateConfigFactory(dependencies).build_encoder_factory()
        halting_factory = BertHaltingConfigFactory(dependencies).build_encoder_factory()
        memory_factory = BertMemoryConfigFactory(dependencies).build_encoder_factory()
        halting_config = None
        if halting_factory is not None:
            halting_config = halting_factory.build_halting_config()
        residual_config = build_residual_config(
            self.encoder_stack_options.residual_connection_option,
            self.encoder_stack_options.residual_model_flag,
            self.encoder_stack_options.residual_stack_options,
            residual_block_size=self.encoder_stack_options.residual_block_size,
            residual_rms_norm_epsilon=self.encoder_stack_options.residual_rms_norm_epsilon,
        )
        gate_config = None
        if gate_factory is not None:
            gate_config = gate_factory.build_gate_config()
        layer_model_config = self._build_encoder_layer_config()
        sequence_expansion_config = _sequence_expansion.build_sequence_expansion_config(
            self.encoder_options, self.hidden_dim
        )
        layer_config = TransformerEncoderBlockLayerConfig(
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
        shared_memory_config = None
        if memory_factory is not None:
            shared_memory_config = memory_factory.build_memory_config()
        shared_gate_config = None
        if self.encoder_layer_controller_options is not None:
            shared_gate_config = (
                self.encoder_layer_controller_options.shared_gate_config
            )
        stack_config = LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self.encoder_options.num_layers,
            last_layer_bias_option=(self.encoder_stack_options.last_layer_bias_option),
            apply_output_postprocessing_flag=(
                self.encoder_stack_options.apply_output_postprocessing_flag
            ),
            shared_gate_config=shared_gate_config,
            shared_halting_config=halting_config,
            shared_memory_config=(shared_memory_config),
            layer_config=layer_config,
        )
        recurrent_factory = BertRecurrentConfigFactory(
            dependencies
        ).build_encoder_factory()
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
        residual_config = AdditiveResidualConfig()
        attention_config = self._build_attention_config()
        feed_forward_config = self._build_feed_forward_config()
        return TransformerEncoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            dropout_probability=options.dropout_probability,
            residual_config=residual_config,
            attention_config=attention_config,
            feed_forward_config=feed_forward_config,
        )

    def _build_attention_config(self) -> SelfAttentionConfig:
        encoder_options = self.encoder_options
        attention_options = self.attention_options
        source_memory_flag = attention_options.source_memory_flag
        source_memory_config = _source_memory.build_source_memory_config(
            attention_options
        )
        projection_model_config = self._build_attention_projection_stack_config()
        return _transformer_stack.separate_self_attention(
            batch_size=self.batch_size,
            num_heads=attention_options.num_heads,
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length,
            dropout_probability=encoder_options.dropout_probability,
            causal_attention_mask_flag=encoder_options.causal_attention_mask_flag,
            add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
            source_memory_flag=source_memory_flag,
            source_memory_config=source_memory_config,
            projection_model_config=projection_model_config,
        )

    def _build_feed_forward_config(self) -> FeedForwardConfig:
        stack_config = self._build_feed_forward_stack_config()
        return _transformer_stack.feed_forward_config(
            stack_config, self.feed_forward_options.soft_moe_config, self.hidden_dim
        )

    def _build_attention_projection_stack_config(self):
        options = self.attention_projection_stack_options
        factory = self.attention_projection_linear_layer_config_factory
        stack_config = factory.build_submodule_stack_config(options)
        self._apply_attention_projection_controls(stack_config)
        control_factory_dependencies = self._control_factory_dependencies()
        recurrent_factory = BertRecurrentConfigFactory(
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

    def _apply_attention_projection_controls(
        self,
        stack_config: LayerStackConfig,
    ) -> None:
        dependencies = self._control_factory_dependencies()
        gate_factory = BertGateConfigFactory(
            dependencies
        ).build_attention_projection_factory()
        halting_factory = BertHaltingConfigFactory(
            dependencies
        ).build_attention_projection_factory()
        memory_factory = BertMemoryConfigFactory(
            dependencies
        ).build_attention_projection_factory()
        controller_options = self.attention_projection_layer_controller_options
        _transformer_stack.apply_selected_controls(
            stack_config,
            controller_options,
            gate_factory,
            halting_factory,
            memory_factory,
        )

    def _build_feed_forward_stack_config(self):
        options = self.feed_forward_stack_options
        factory = self.feed_forward_linear_layer_config_factory
        stack_config = factory.build_submodule_stack_config(options)
        self._apply_feed_forward_controls(stack_config)
        control_factory_dependencies = self._control_factory_dependencies()
        recurrent_factory = BertRecurrentConfigFactory(
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

    def _apply_feed_forward_controls(self, stack_config: LayerStackConfig) -> None:
        dependencies = self._control_factory_dependencies()
        gate_factory = BertGateConfigFactory(dependencies).build_feed_forward_factory()
        halting_factory = BertHaltingConfigFactory(
            dependencies
        ).build_feed_forward_factory()
        memory_factory = BertMemoryConfigFactory(
            dependencies
        ).build_feed_forward_factory()
        controller_options = self.feed_forward_layer_controller_options
        _transformer_stack.apply_selected_controls(
            stack_config,
            controller_options,
            gate_factory,
            halting_factory,
            memory_factory,
        )

    def _control_factory_dependencies(self) -> BertControlFactoryDependencies:
        return BertControlFactoryDependencies(
            hidden_dim=self.hidden_dim,
            encoder_stack_options=self.encoder_stack_options,
            encoder_submodule_stack_options=self.encoder_submodule_stack_options,
            attention_projection_stack_options=(
                self.attention_projection_stack_options
            ),
            feed_forward_stack_options=self.feed_forward_stack_options,
            encoder_layer_controller_options=self.encoder_layer_controller_options,
            attention_projection_layer_controller_options=(
                self.attention_projection_layer_controller_options
            ),
            feed_forward_layer_controller_options=(
                self.feed_forward_layer_controller_options
            ),
            encoder_dynamic_memory_options=self.encoder_dynamic_memory_options,
            attention_projection_dynamic_memory_options=(
                self.attention_projection_dynamic_memory_options
            ),
            feed_forward_dynamic_memory_options=(
                self.feed_forward_dynamic_memory_options
            ),
            encoder_recurrent_controller_options=(
                self.encoder_recurrent_controller_options
            ),
            attention_projection_recurrent_controller_options=(
                self.attention_projection_recurrent_controller_options
            ),
            feed_forward_recurrent_controller_options=(
                self.feed_forward_recurrent_controller_options
            ),
        )
