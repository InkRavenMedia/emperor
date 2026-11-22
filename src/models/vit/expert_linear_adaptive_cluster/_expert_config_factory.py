from dataclasses import dataclass, field

import models.vit.expert_linear_adaptive_cluster.config as config
from emperor.augmentations.adaptive_parameters import GroupingConfig
from models.vit._expert_defaults import AdaptiveExpertDefaults
from models.vit.expert_linear_adaptive_cluster import (
    _config_defaults as config_defaults,
)
from models.vit.expert_linear_adaptive_cluster._vit_expert_config_factory import (
    VitExpertAdaptiveConfigDependencies as _ExpertAdaptiveDependencies,
)
from models.vit.expert_linear_adaptive_cluster._vit_expert_config_factory import (
    VitExpertAdaptiveConfigFactory,
)
from models.vit.expert_linear_adaptive_cluster.runtime_options import (
    AdaptiveGeneratorStackOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsSubmoduleStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class ExpertAdaptiveConfigDependencies:
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    router_grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    hidden_dim: int
    encoder_options: TransformerEncoderOptions | None
    attention_options: TransformerAttentionOptions | None
    feed_forward_options: TransformerFeedForwardOptions | None
    mixture_options: ExpertsMixtureOptions | None
    mixture_submodule_stack_options: ExpertsSubmoduleStackOptions | None
    mixture_layer_controller_options: ExpertsLayerControllerOptions | None
    mixture_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    mixture_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_stack_options: ExpertsSubmoduleStackOptions | None
    sampler_options: ExpertsSamplerOptions | None
    router_options: ExpertsRouterOptions | None
    router_stack_options: ExpertsSubmoduleStackOptions | None
    router_layer_controller_options: ExpertsLayerControllerOptions | None
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_layer_controller_options: ExpertsLayerControllerOptions | None
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions | None
    expert_attention_use_kv_expert_models_flag: bool


class ExpertAdaptiveConfigFactory(
    AdaptiveExpertDefaults[
        ExpertAdaptiveConfigDependencies,
        _ExpertAdaptiveDependencies,
        SubmoduleStackOptions,
    ]
):
    _config = config
    _defaults = config_defaults
    _resolved_inputs = staticmethod(_ExpertAdaptiveDependencies)
    _builder = staticmethod(VitExpertAdaptiveConfigFactory)
