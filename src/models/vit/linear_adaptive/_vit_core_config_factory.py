from dataclasses import dataclass

from models.vit._encoder_builder import (
    EncoderInputs,
    ExpertFactory,
    VisionEncoderBuilder,
)
from models.vit.linear_adaptive._control_factory_dependencies import (
    VitControlFactoryDependencies,
)
from models.vit.linear_adaptive._encoder_control_config_factory import (
    VitGateConfigFactory,
    VitHaltingConfigFactory,
    VitMemoryConfigFactory,
    VitRecurrentConfigFactory,
)
from models.vit.linear_adaptive._linear_layer_config_factory import (
    LinearLayerConfigFactory as VitLinearLayerConfigFactory,
)
from models.vit.linear_adaptive.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
)

_VitExpertConfigFactory = ExpertFactory[SubmoduleStackOptions]


@dataclass(frozen=True)
class CoreConfigDependencies(
    EncoderInputs[
        TransformerEncoderOptions,
        SubmoduleStackOptions,
        MainLayerStackOptions,
        LayerControllerOptions,
        DynamicMemoryOptions,
        RecurrentControllerOptions,
    ]
):
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
    linear_layer_config_factory: VitLinearLayerConfigFactory
    attention_projection_linear_layer_config_factory: (
        VitLinearLayerConfigFactory | None
    ) = None
    feed_forward_linear_layer_config_factory: VitLinearLayerConfigFactory | None = None
    expert_config_factory: _VitExpertConfigFactory | None = None


class VitCoreConfigFactory(
    VisionEncoderBuilder[
        VitControlFactoryDependencies,
        SubmoduleStackOptions,
        TransformerEncoderOptions,
        MainLayerStackOptions,
        LayerControllerOptions,
        DynamicMemoryOptions,
        RecurrentControllerOptions,
    ]
):
    _control_inputs = staticmethod(VitControlFactoryDependencies.from_encoder)
    _gate = staticmethod(VitGateConfigFactory)
    _halting = staticmethod(VitHaltingConfigFactory)
    _memory = staticmethod(VitMemoryConfigFactory)
    _recurrent = staticmethod(VitRecurrentConfigFactory)
