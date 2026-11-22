from dataclasses import dataclass, replace
from typing import Literal

import models.gpt.linear.config as config
from emperor.layers import (
    AdditiveResidualConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.transformer import (
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerConfig,
)
from models._dense_text_core import DenseTextCore, TextStageSettings
from models.gpt.linear import _config_defaults as config_defaults
from models.gpt.linear._decoder_control_config_factory import (
    GateConfigFactory,
    HaltingConfigFactory,
    MemoryConfigFactory,
    RecurrentConfigFactory,
)
from models.gpt.linear._linear_layer_config_factory import LinearLayerConfigFactory
from models.gpt.linear.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class CoreConfigDependencies:
    batch_size: int
    sequence_length: int
    decoder_options: TransformerDecoderOptions | None
    attention_options: TransformerAttentionOptions | None
    feed_forward_options: TransformerFeedForwardOptions | None
    attention_projection_stack_options: SubmoduleStackOptions | None
    attention_projection_layer_controller_options: LayerControllerOptions | None
    attention_projection_dynamic_memory_options: DynamicMemoryOptions | None
    attention_projection_recurrent_controller_options: RecurrentControllerOptions | None
    feed_forward_stack_options: SubmoduleStackOptions | None
    feed_forward_layer_controller_options: LayerControllerOptions | None
    feed_forward_dynamic_memory_options: DynamicMemoryOptions | None
    feed_forward_recurrent_controller_options: RecurrentControllerOptions | None
    stack_options: MainLayerStackOptions | None
    submodule_stack_options: SubmoduleStackOptions | None
    layer_controller_options: LayerControllerOptions | None
    dynamic_memory_options: DynamicMemoryOptions | None
    recurrent_controller_options: RecurrentControllerOptions | None
    linear_layer_config_factory: LinearLayerConfigFactory


class CoreConfigFactory(DenseTextCore):
    _gate = staticmethod(GateConfigFactory)
    _halting = staticmethod(HaltingConfigFactory)
    _memory = staticmethod(MemoryConfigFactory)
    _recurrent = staticmethod(RecurrentConfigFactory)
    _block = TransformerDecoderBlockLayerConfig

    def __init__(self, dependencies: CoreConfigDependencies) -> None:
        self.batch_size = dependencies.batch_size
        self.sequence_length = dependencies.sequence_length
        self.linear_layer_config_factory = dependencies.linear_layer_config_factory
        self.decoder_options = (
            dependencies.decoder_options or config_defaults.gpt_decoder_options(config)
        )
        self.hidden_dim = self.decoder_options.hidden_dim
        self.attention_options = (
            dependencies.attention_options
            or config_defaults.gpt_attention_options(config)
        )
        self.feed_forward_options = (
            dependencies.feed_forward_options
            or config_defaults.gpt_feed_forward_options(config)
        )
        self.stack_options = (
            dependencies.stack_options
            or config_defaults.main_layer_stack_options(config)
        )
        self.submodule_stack_options = (
            dependencies.submodule_stack_options
            or config_defaults.submodule_stack_options(config, self.stack_options)
        )
        self.layer_controller_options = (
            dependencies.layer_controller_options
            or self.__layer_controller_options(
                role="main",
            )
        )
        self.dynamic_memory_options = (
            dependencies.dynamic_memory_options
            or self.__dynamic_memory_options(role="main")
        )
        self.recurrent_controller_options = (
            dependencies.recurrent_controller_options
            or self.__recurrent_controller_options(role="main")
        )
        self.attention_projection_stack_options = (
            dependencies.attention_projection_stack_options
            or config_defaults.attention_projection_stack_options(
                config,
                self.decoder_options,
                self.attention_options,
            )
        )
        self.attention_projection_layer_controller_options = (
            dependencies.attention_projection_layer_controller_options
            or self.__layer_controller_options(
                role="attention",
            )
        )
        self.attention_projection_dynamic_memory_options = (
            dependencies.attention_projection_dynamic_memory_options
            or self.__dynamic_memory_options(role="attention")
        )
        self.attention_projection_recurrent_controller_options = (
            dependencies.attention_projection_recurrent_controller_options
            or self.__recurrent_controller_options(role="attention")
        )
        self.feed_forward_stack_options = (
            dependencies.feed_forward_stack_options
            or config_defaults.feed_forward_stack_options(
                config,
                self.decoder_options,
                self.feed_forward_options,
            )
        )
        self.feed_forward_layer_controller_options = (
            dependencies.feed_forward_layer_controller_options
            or self.__layer_controller_options(
                role="feed_forward",
            )
        )
        self.feed_forward_dynamic_memory_options = (
            dependencies.feed_forward_dynamic_memory_options
            or self.__dynamic_memory_options(role="feed_forward")
        )
        self.feed_forward_recurrent_controller_options = (
            dependencies.feed_forward_recurrent_controller_options
            or self.__recurrent_controller_options(role="feed_forward")
        )

    @property
    def _stage_options(self) -> TransformerDecoderOptions:
        return self.decoder_options

    def _causal_attention(self, options: TextStageSettings) -> bool:
        return True

    def build_decoder_config(self) -> LayerStackConfig | RecurrentLayerConfig:
        return self._build_stage_stack()

    def _build_stage_layer(self) -> TransformerDecoderLayerConfig:
        options = self.decoder_options
        residual_config = AdditiveResidualConfig()
        self_attention_config = self._build_attention_config()
        feed_forward_config = self._build_feed_forward_config()
        return TransformerDecoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            dropout_probability=options.dropout_probability,
            residual_config=residual_config,
            self_attention_config=self_attention_config,
            cross_attention_config=None,
            feed_forward_config=feed_forward_config,
        )

    def __layer_controller_options(
        self,
        *,
        role: Literal["main", "attention", "feed_forward"],
    ) -> LayerControllerOptions:
        defaults = config_defaults.linears_layer_controller_options(
            config,
            role,
        )
        return replace(defaults, halting_option=LayerControllerOptions.halting_option)

    def __dynamic_memory_options(
        self,
        *,
        role: Literal["main", "attention", "feed_forward"],
    ) -> DynamicMemoryOptions:
        return config_defaults.linears_dynamic_memory_options(
            config,
            role,
        )

    def __recurrent_controller_options(
        self,
        *,
        role: Literal["main", "attention", "feed_forward"],
    ) -> RecurrentControllerOptions:
        defaults = config_defaults.linears_recurrent_controller_options(
            config,
            role,
        )
        return replace(
            defaults,
            recurrent_halting_option=(
                RecurrentControllerOptions.recurrent_halting_option
            ),
        )
