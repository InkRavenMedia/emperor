from typing import TYPE_CHECKING

from emperor.attention import (
    SelfAttentionConfig,
)
from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.transformer import (
    FeedForwardConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerConfig,
)
from models import _attention_source_memory as _source_memory
from models import _sequence_expansion, _transformer_stack
from models._model_config import sequence_model_config
from models._positional_embedding import build_position_config
from models.gpt.expert_linear_adaptive._boundary_config_factory import (
    BoundaryConfigDependencies,
    BoundaryConfigFactory,
)
from models.gpt.expert_linear_adaptive._control_support import (
    ExpertsGateConfigFactory as GptGateConfigFactory,
)
from models.gpt.expert_linear_adaptive._control_support import (
    ExpertsHaltingConfigFactory as GptHaltingConfigFactory,
)
from models.gpt.expert_linear_adaptive._control_support import (
    ExpertsMemoryConfigFactory as GptMemoryConfigFactory,
)
from models.gpt.expert_linear_adaptive._control_support import (
    ExpertsRecurrentConfigFactory as GptRecurrentConfigFactory,
)
from models.gpt.expert_linear_adaptive.runtime_options import (
    ConstructionOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
)

from . import config
from ._residual import build_residual_config

if TYPE_CHECKING:
    from emperor.config import ConfigBase, ModelConfig


class GptBackendConfigBuilder:
    def __init__(
        self,
        options: ConstructionOptions,
        *,
        experiment_config_type: type["ConfigBase"],
    ) -> None:
        self.batch_size = options.batch_size
        self.learning_rate = options.learning_rate
        self.input_dim = options.input_dim
        self.output_dim = options.output_dim
        self.sequence_length = options.sequence_length
        self.embedding_options = options.embedding_options
        self.lm_head_options = options.lm_head_options
        self.decoder_options = options.decoder_options
        self.hidden_dim = self.decoder_options.hidden_dim
        self.stack_num_layers = self.decoder_options.num_layers
        self.stack_activation = self.decoder_options.activation
        self.stack_dropout_probability = self.decoder_options.dropout_probability
        self.layer_norm_position = self.decoder_options.layer_norm_position
        self.normalization = self.decoder_options.normalization
        self.positional_embedding_options = options.positional_embedding_options
        self.positional_embedding_option = self.positional_embedding_options.option
        self.positional_embedding_padding_idx = (
            self.positional_embedding_options.padding_idx
        )
        self.positional_embedding_auto_expand_flag = (
            self.positional_embedding_options.auto_expand_flag
        )
        self.attention_options = options.attention_options
        self.attn_num_heads = self.attention_options.num_heads
        self.attn_num_layers = self.attention_options.num_layers
        self.attn_bias_flag = self.attention_options.bias_flag
        self.attn_add_key_value_bias_flag = (
            self.attention_options.add_key_value_bias_flag
        )
        self.attention_projection_stack_options = (
            options.attention_projection_stack_options
        )
        self.attention_projection_layer_controller_options = (
            options.attention_projection_layer_controller_options
        )
        self.attention_projection_dynamic_memory_options = (
            options.attention_projection_dynamic_memory_options
        )
        self.attention_projection_recurrent_controller_options = (
            options.attention_projection_recurrent_controller_options
        )
        self.feed_forward_options = options.feed_forward_options
        self.ff_num_layers = self.feed_forward_options.num_layers
        self.ff_bias_flag = self.feed_forward_options.bias_flag
        self.feed_forward_stack_options = options.feed_forward_stack_options
        self.feed_forward_layer_controller_options = (
            options.feed_forward_layer_controller_options
        )
        self.feed_forward_dynamic_memory_options = (
            options.feed_forward_dynamic_memory_options
        )
        self.feed_forward_recurrent_controller_options = (
            options.feed_forward_recurrent_controller_options
        )
        self.submodule_stack_options = options.submodule_stack_options
        self.layer_controller_options = options.layer_controller_options
        self.dynamic_memory_options = options.dynamic_memory_options
        self.recurrent_controller_options = options.recurrent_controller_options
        self.experiment_config_type = experiment_config_type

    def build(self) -> "ModelConfig":
        positional_embedding_config = self._build_positional_embedding_config()
        resolved_decoder_config = self._build_decoder_config()
        decoder_config = _sequence_expansion.with_expansion_attention_capacity(
            self.runtime.configure_stack(resolved_decoder_config), self.sequence_length
        )
        boundary_config_dependencies = BoundaryConfigDependencies(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            embedding_options=self.embedding_options,
            lm_head_options=self.lm_head_options,
        )
        boundary_config = BoundaryConfigFactory(
            boundary_config_dependencies
        ).build_boundary_config()
        decoder_output_normalization = config.DECODER_OUTPUT_NORMALIZATION
        if self.decoder_options is not None:
            decoder_output_normalization = self.decoder_options.output_normalization
        experiment_config = self.experiment_config_type(
            positional_embedding_config=positional_embedding_config,
            boundary_config=boundary_config,
            decoder_config=decoder_config,
            decoder_output_normalization=decoder_output_normalization,
        )
        return sequence_model_config(self, experiment_config)

    def _build_positional_embedding_config(
        self,
    ) -> AbsolutePositionalEmbeddingConfig:
        options = self.positional_embedding_options
        return build_position_config(
            options.option,
            num_embeddings=self.sequence_length,
            embedding_dim=self.hidden_dim,
            padding_idx=options.padding_idx,
            auto_expand_flag=options.auto_expand_flag,
        )

    def _build_decoder_config(self) -> "ConfigBase":
        layer_model_config = self._build_decoder_layer_config()
        gate_factory = self._gate_config_factory()
        halting_factory = self._halting_config_factory()
        memory_factory = self._memory_config_factory()
        halting_config = halting_factory.build_halting_config()
        gate_config = gate_factory.build_gate_config()
        sequence_expansion_config = _sequence_expansion.build_sequence_expansion_config(
            self.decoder_options, self.hidden_dim
        )
        layer_config = TransformerDecoderBlockLayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=gate_config,
            halting_config=None,
            layer_model_config=layer_model_config,
            sequence_expansion_config=sequence_expansion_config,
        )
        shared_memory_config = memory_factory.build_memory_config()
        stack_config = LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self.decoder_options.num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=True,
            shared_gate_config=self.layer_controller_options.shared_gate_config,
            shared_halting_config=halting_config,
            shared_memory_config=shared_memory_config,
            # The wrapped TransformerDecoderLayer owns its own norm, residual, and
            # dropout, so the generic Layer pipeline is neutralized to a pass-through.
            layer_config=layer_config,
        )
        recurrent_factory = GptRecurrentConfigFactory(
            recurrent_controller_options=self.recurrent_controller_options,
            gate_config_factory=gate_factory,
            halting_config_factory=halting_factory,
        )
        decoder_config = recurrent_factory.build_config(stack_config)
        if isinstance(decoder_config, RecurrentLayerConfig):
            decoder_config.input_dim = self.hidden_dim
            decoder_config.output_dim = self.hidden_dim
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

    def _build_attention_config(self) -> SelfAttentionConfig:
        decoder_options = self.decoder_options
        attention_options = self.attention_options
        source_memory_flag = attention_options.source_memory_flag
        projection_model_config = self._build_attention_projection_stack_config()
        source_memory_config = _source_memory.build_source_memory_config(
            attention_options, projection_model_config
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

    def _build_attention_projection_stack_config(self) -> "ConfigBase":
        stack_config = self._build_attention_projection_base_stack_config()
        self._apply_attention_projection_controls(stack_config)
        recurrent_factory = self._attention_projection_recurrent_config_factory()
        if recurrent_factory is None:
            return stack_config
        recurrent_config = recurrent_factory.build_config(stack_config)
        if isinstance(recurrent_config, RecurrentLayerConfig):
            recurrent_config.input_dim = self.hidden_dim
            recurrent_config.output_dim = self.hidden_dim
        return recurrent_config

    def _build_attention_projection_base_stack_config(self) -> LayerStackConfig:
        options = self._effective_attention_projection_stack_options()
        attention_projection_base_stack_config = self._build_linear_stack_config(
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            bias_flag=options.bias_flag,
            activation=options.activation,
            residual_connection_option=options.residual_connection_option,
            residual_block_size=options.residual_block_size,
            residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
            residual_model_flag=options.residual_model_flag,
            residual_stack_options=options.residual_stack_options,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            dropout_probability=options.dropout_probability,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
        )
        return attention_projection_base_stack_config

    def _apply_attention_projection_controls(
        self,
        stack_config: LayerStackConfig,
    ) -> None:
        gate_factory = self._attention_projection_gate_config_factory()
        halting_factory = self._attention_projection_halting_config_factory()
        memory_factory = self._attention_projection_memory_config_factory()
        _transformer_stack.apply_enabled_controls(
            stack_config,
            self.attention_projection_layer_controller_options,
            self.attention_projection_dynamic_memory_options,
            gate_factory,
            halting_factory,
            memory_factory,
        )

    def _build_feed_forward_stack_config(self) -> "ConfigBase":
        stack_config = self._build_feed_forward_base_stack_config()
        self._apply_feed_forward_controls(stack_config)
        recurrent_factory = self._feed_forward_recurrent_config_factory()
        if recurrent_factory is None:
            return stack_config
        recurrent_config = recurrent_factory.build_config(stack_config)
        if isinstance(recurrent_config, RecurrentLayerConfig):
            recurrent_config.input_dim = self.hidden_dim
            recurrent_config.output_dim = self.hidden_dim
        return recurrent_config

    def _build_feed_forward_base_stack_config(self) -> "ConfigBase":
        options = self._effective_feed_forward_stack_options()
        feed_forward_base_stack_config = self._build_linear_stack_config(
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            bias_flag=options.bias_flag,
            activation=options.activation,
            residual_connection_option=options.residual_connection_option,
            residual_block_size=options.residual_block_size,
            residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
            residual_model_flag=options.residual_model_flag,
            residual_stack_options=options.residual_stack_options,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            dropout_probability=options.dropout_probability,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
        )
        return feed_forward_base_stack_config

    def _apply_feed_forward_controls(self, stack_config: "ConfigBase") -> None:
        target_stack = self._feed_forward_control_target_stack(stack_config)
        gate_factory = self._feed_forward_gate_config_factory()
        halting_factory = self._feed_forward_halting_config_factory()
        memory_factory = self._feed_forward_memory_config_factory()
        _transformer_stack.apply_enabled_controls(
            target_stack,
            self.feed_forward_layer_controller_options,
            self.feed_forward_dynamic_memory_options,
            gate_factory,
            halting_factory,
            memory_factory,
        )

    def _feed_forward_control_target_stack(
        self,
        stack_config: "ConfigBase",
    ) -> LayerStackConfig:
        return _transformer_stack.feed_forward_control_stack(stack_config)

    def _build_linear_layer_config(self, *, bias_flag: bool) -> LinearLayerConfig:
        return LinearLayerConfig(
            bias_flag=bias_flag,
        )

    def _build_linear_stack_config(
        self,
        *,
        num_layers: int,
        bias_flag: bool,
        layer_norm_position: LayerNormPositionOptions,
        normalization: NormalizationOptions = NormalizationOptions.RMS_NORM,
        dropout_probability: float,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        activation: ActivationOptions | None = None,
        residual_connection_option: type[ResidualConfig] | None = None,
        residual_model_flag: bool = False,
        residual_block_size: int | None = None,
        residual_rms_norm_epsilon: float | None = None,
        residual_stack_options=None,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag: bool = True,
    ) -> LayerStackConfig:
        layer_model_config = self._build_linear_layer_config(bias_flag=bias_flag)
        residual_config = build_residual_config(
            residual_connection_option,
            residual_model_flag,
            residual_stack_options,
            residual_block_size=residual_block_size,
            residual_rms_norm_epsilon=residual_rms_norm_epsilon,
        )
        if activation is None:
            activation = self.decoder_options.activation
        layer_config = LayerConfig(
            activation=activation,
            layer_norm_position=layer_norm_position,
            normalization=normalization,
            residual_config=residual_config,
            dropout_probability=dropout_probability,
            gate_config=None,
            halting_config=None,
            layer_model_config=layer_model_config,
        )
        if hidden_dim is None:
            hidden_dim = self.hidden_dim
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
            layer_config=layer_config,
        )

    def _gate_config_factory(self) -> GptGateConfigFactory:
        return GptGateConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
        )

    def _halting_config_factory(self) -> GptHaltingConfigFactory:
        return GptHaltingConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
            output_dim=self.hidden_dim,
        )

    def _memory_config_factory(self) -> GptMemoryConfigFactory:
        stack_options = ExpertsStackOptions(
            hidden_dim=self.hidden_dim,
            bias_flag=True,
            layer_norm_position=self.decoder_options.layer_norm_position,
            normalization=self.decoder_options.normalization,
            num_layers=self.decoder_options.num_layers,
            activation=self.decoder_options.activation,
            residual_connection_option=None,
            residual_block_size=None,
            residual_rms_norm_epsilon=None,
            residual_model_flag=False,
            residual_stack_options=self.submodule_stack_options.residual_stack_options,
            dropout_probability=self.decoder_options.dropout_probability,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=True,
        )
        return GptMemoryConfigFactory(
            stack_options=stack_options,
            dynamic_memory_options=self.dynamic_memory_options,
            submodule_stack_options=self.submodule_stack_options,
        )

    def _effective_attention_projection_stack_options(
        self,
    ) -> ExpertsSubmoduleStackOptions:
        if self.attention_projection_stack_options is not None:
            return self.attention_projection_stack_options
        return ExpertsSubmoduleStackOptions(
            hidden_dim=self.hidden_dim,
            num_layers=self.attention_options.num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=True,
            activation=self.decoder_options.activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_connection_option=None,
            residual_block_size=None,
            residual_rms_norm_epsilon=None,
            residual_model_flag=False,
            residual_stack_options=self.submodule_stack_options.residual_stack_options,
            dropout_probability=0.0,
            bias_flag=self.attention_options.bias_flag,
        )

    def _effective_feed_forward_stack_options(self) -> ExpertsSubmoduleStackOptions:
        if self.feed_forward_stack_options is not None:
            return self.feed_forward_stack_options
        return ExpertsSubmoduleStackOptions(
            hidden_dim=self.hidden_dim,
            num_layers=self.feed_forward_options.num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=True,
            activation=self.decoder_options.activation,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            normalization=NormalizationOptions.RMS_NORM,
            residual_connection_option=None,
            residual_block_size=None,
            residual_rms_norm_epsilon=None,
            residual_model_flag=False,
            residual_stack_options=self.submodule_stack_options.residual_stack_options,
            dropout_probability=self.decoder_options.dropout_probability,
            bias_flag=self.feed_forward_options.bias_flag,
        )

    def _attention_projection_gate_config_factory(
        self,
    ) -> GptGateConfigFactory | None:
        if (
            self.attention_projection_layer_controller_options is None
            or self.attention_projection_recurrent_controller_options is None
        ):
            return None
        return GptGateConfigFactory(
            layer_controller_options=(
                self.attention_projection_layer_controller_options
            ),
            recurrent_controller_options=(
                self.attention_projection_recurrent_controller_options
            ),
            submodule_stack_options=(
                self._effective_attention_projection_stack_options()
            ),
            recurrent_stack_inherits_gate_stack=False,
        )

    def _attention_projection_halting_config_factory(
        self,
    ) -> GptHaltingConfigFactory | None:
        if (
            self.attention_projection_layer_controller_options is None
            or self.attention_projection_recurrent_controller_options is None
        ):
            return None
        attention_stack_options = self._effective_attention_projection_stack_options()
        return GptHaltingConfigFactory(
            layer_controller_options=(
                self.attention_projection_layer_controller_options
            ),
            recurrent_controller_options=(
                self.attention_projection_recurrent_controller_options
            ),
            submodule_stack_options=attention_stack_options,
            output_dim=attention_stack_options.hidden_dim,
            halting_stack_defaults=attention_stack_options,
            recurrent_stack_inherits_halting_stack=False,
        )

    def _attention_projection_memory_config_factory(
        self,
    ) -> GptMemoryConfigFactory | None:
        if self.attention_projection_dynamic_memory_options is None:
            return None
        attention_stack_options = self._effective_attention_projection_stack_options()
        return GptMemoryConfigFactory(
            stack_options=attention_stack_options,
            dynamic_memory_options=(self.attention_projection_dynamic_memory_options),
            submodule_stack_options=attention_stack_options,
        )

    def _attention_projection_recurrent_config_factory(
        self,
    ) -> GptRecurrentConfigFactory | None:
        gate_factory = self._attention_projection_gate_config_factory()
        halting_factory = self._attention_projection_halting_config_factory()
        if (
            self.attention_projection_recurrent_controller_options is None
            or gate_factory is None
            or halting_factory is None
        ):
            return None
        return GptRecurrentConfigFactory(
            recurrent_controller_options=(
                self.attention_projection_recurrent_controller_options
            ),
            gate_config_factory=gate_factory,
            halting_config_factory=halting_factory,
        )

    def _feed_forward_gate_config_factory(self) -> GptGateConfigFactory | None:
        if (
            self.feed_forward_layer_controller_options is None
            or self.feed_forward_recurrent_controller_options is None
        ):
            return None
        return GptGateConfigFactory(
            layer_controller_options=self.feed_forward_layer_controller_options,
            recurrent_controller_options=(
                self.feed_forward_recurrent_controller_options
            ),
            submodule_stack_options=self._effective_feed_forward_stack_options(),
            recurrent_stack_inherits_gate_stack=False,
        )

    def _feed_forward_halting_config_factory(
        self,
    ) -> GptHaltingConfigFactory | None:
        if (
            self.feed_forward_layer_controller_options is None
            or self.feed_forward_recurrent_controller_options is None
        ):
            return None
        feed_forward_stack_options = self._effective_feed_forward_stack_options()
        return GptHaltingConfigFactory(
            layer_controller_options=self.feed_forward_layer_controller_options,
            recurrent_controller_options=(
                self.feed_forward_recurrent_controller_options
            ),
            submodule_stack_options=feed_forward_stack_options,
            output_dim=feed_forward_stack_options.hidden_dim,
            halting_stack_defaults=feed_forward_stack_options,
            recurrent_stack_inherits_halting_stack=False,
        )

    def _feed_forward_memory_config_factory(self) -> GptMemoryConfigFactory | None:
        if self.feed_forward_dynamic_memory_options is None:
            return None
        feed_forward_stack_options = self._effective_feed_forward_stack_options()
        return GptMemoryConfigFactory(
            stack_options=feed_forward_stack_options,
            dynamic_memory_options=self.feed_forward_dynamic_memory_options,
            submodule_stack_options=feed_forward_stack_options,
        )

    def _feed_forward_recurrent_config_factory(
        self,
    ) -> GptRecurrentConfigFactory | None:
        gate_factory = self._feed_forward_gate_config_factory()
        halting_factory = self._feed_forward_halting_config_factory()
        if (
            self.feed_forward_recurrent_controller_options is None
            or gate_factory is None
            or halting_factory is None
        ):
            return None
        return GptRecurrentConfigFactory(
            recurrent_controller_options=(
                self.feed_forward_recurrent_controller_options
            ),
            gate_config_factory=gate_factory,
            halting_config_factory=halting_factory,
        )
