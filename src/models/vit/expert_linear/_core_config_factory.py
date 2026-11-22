from dataclasses import dataclass

import models.vit.expert_linear.config as config
from models.vit._encoder_builder import ExpertFactory
from models.vit._encoder_defaults import ConfiguredVisionEncoder
from models.vit.expert_linear import _config_defaults as config_defaults
from models.vit.expert_linear._linear_layer_config_factory import (
    LinearLayerConfigFactory,
)
from models.vit.expert_linear._vit_core_config_factory import (
    CoreConfigDependencies as _CoreDependencies,
)
from models.vit.expert_linear._vit_core_config_factory import (
    VitCoreConfigFactory,
)
from models.vit.expert_linear.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class CoreConfigDependencies:
    batch_size: int
    sequence_length: int
    encoder_options: TransformerEncoderOptions | None
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
    expert_config_factory: ExpertFactory[SubmoduleStackOptions] | None = None


class CoreConfigFactory(
    ConfiguredVisionEncoder[CoreConfigDependencies, _CoreDependencies]
):
    _config = config
    _defaults = config_defaults
    _resolved_inputs = staticmethod(_CoreDependencies)
    _encoder = staticmethod(VitCoreConfigFactory)
