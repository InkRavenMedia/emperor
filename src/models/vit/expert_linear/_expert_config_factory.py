from dataclasses import dataclass

import models.vit.expert_linear.config as config
from models.vit._expert_defaults import ExpertDefaults
from models.vit.expert_linear import _config_defaults as config_defaults
from models.vit.expert_linear._vit_expert_config_factory import (
    VitExpertConfigDependencies as _ExpertDependencies,
)
from models.vit.expert_linear._vit_expert_config_factory import VitExpertConfigFactory
from models.vit.expert_linear.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsSubmoduleStackOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class ExpertConfigDependencies:
    hidden_dim: int
    encoder_options: TransformerEncoderOptions | None
    attention_options: TransformerAttentionOptions | None
    feed_forward_options: TransformerFeedForwardOptions | None
    mixture_options: ExpertsMixtureOptions | None
    expert_stack_options: ExpertsSubmoduleStackOptions | None
    sampler_options: ExpertsSamplerOptions | None
    router_options: ExpertsRouterOptions | None
    router_stack_options: ExpertsSubmoduleStackOptions | None
    expert_layer_controller_options: ExpertsLayerControllerOptions | None
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_attention_use_kv_expert_models_flag: bool


class ExpertConfigFactory(
    ExpertDefaults[ExpertConfigDependencies, _ExpertDependencies, SubmoduleStackOptions]
):
    _config = config
    _defaults = config_defaults
    _resolved_inputs = staticmethod(_ExpertDependencies)
    _builder = staticmethod(VitExpertConfigFactory)
