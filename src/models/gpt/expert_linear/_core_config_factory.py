from dataclasses import dataclass

import models.gpt.expert_linear.config as config
from models.gpt.expert_linear._config_defaults import (
    attention_projection_stack_options,
    feed_forward_stack_options,
    gpt_attention_options,
    gpt_decoder_options,
    gpt_feed_forward_options,
    main_layer_stack_options,
)
from models.gpt.expert_linear._gpt_core_config_factory import (
    CoreConfigDependencies as _CoreDependencies,
)
from models.gpt.expert_linear._gpt_core_config_factory import GptCoreConfigFactory
from models.gpt.expert_linear._gpt_core_config_factory import (
    _GptExpertConfigFactory as _ExpertConfigFactory,
)
from models.gpt.expert_linear._linear_layer_config_factory import (
    LinearLayerConfigFactory,
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
    expert_config_factory: _ExpertConfigFactory | None = None


class CoreConfigFactory:
    def __init__(self, dependencies: CoreConfigDependencies) -> None:
        self.dependencies = dependencies
        decoder_options = dependencies.decoder_options
        if decoder_options is None:
            decoder_options = gpt_decoder_options(config)
        self.decoder_options = decoder_options
        attention_options = dependencies.attention_options
        if attention_options is None:
            attention_options = gpt_attention_options(config)
        self.attention_options = attention_options
        feed_forward_options = dependencies.feed_forward_options
        if feed_forward_options is None:
            feed_forward_options = gpt_feed_forward_options(config)
        self.feed_forward_options = feed_forward_options
        stack_options = dependencies.stack_options
        if stack_options is None:
            stack_options = main_layer_stack_options(config)
        self.stack_options = stack_options
        self.attention_projection_stack_options = (
            dependencies.attention_projection_stack_options
            or attention_projection_stack_options(
                config,
                self.decoder_options,
                self.attention_options,
            )
        )
        self.feed_forward_stack_options = (
            dependencies.feed_forward_stack_options
            or feed_forward_stack_options(
                config,
                self.decoder_options,
                self.feed_forward_options,
            )
        )

    def build_decoder_config(self):
        core_dependencies = self.__core_dependencies()
        core_factory = GptCoreConfigFactory(core_dependencies)
        decoder_config = core_factory.build_decoder_config()
        return decoder_config

    def __core_dependencies(self) -> _CoreDependencies:
        dependencies = self.dependencies
        return _CoreDependencies(
            batch_size=dependencies.batch_size,
            sequence_length=dependencies.sequence_length,
            decoder_options=self.decoder_options,
            attention_options=self.attention_options,
            feed_forward_options=self.feed_forward_options,
            attention_projection_stack_options=(
                self.attention_projection_stack_options
            ),
            attention_projection_layer_controller_options=(
                dependencies.attention_projection_layer_controller_options
            ),
            attention_projection_dynamic_memory_options=(
                dependencies.attention_projection_dynamic_memory_options
            ),
            attention_projection_recurrent_controller_options=(
                dependencies.attention_projection_recurrent_controller_options
            ),
            feed_forward_stack_options=self.feed_forward_stack_options,
            feed_forward_layer_controller_options=(
                dependencies.feed_forward_layer_controller_options
            ),
            feed_forward_dynamic_memory_options=(
                dependencies.feed_forward_dynamic_memory_options
            ),
            feed_forward_recurrent_controller_options=(
                dependencies.feed_forward_recurrent_controller_options
            ),
            stack_options=self.stack_options,
            submodule_stack_options=dependencies.submodule_stack_options,
            layer_controller_options=dependencies.layer_controller_options,
            dynamic_memory_options=dependencies.dynamic_memory_options,
            recurrent_controller_options=dependencies.recurrent_controller_options,
            linear_layer_config_factory=dependencies.linear_layer_config_factory,
            expert_config_factory=dependencies.expert_config_factory,
        )
