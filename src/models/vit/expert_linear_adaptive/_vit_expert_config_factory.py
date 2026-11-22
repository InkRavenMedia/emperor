from dataclasses import dataclass, field

from emperor.augmentations.adaptive_parameters import GroupingConfig
from models._attention_experts import AdaptiveAttentionExpertBuilder
from models.vit.expert_linear_adaptive._expert_control_config_factory import (
    ControlConfigDependencies as ExpertAdaptiveControlConfigDependencies,
)
from models.vit.expert_linear_adaptive._expert_control_config_factory import (
    ControlConfigFactory as ExpertAdaptiveControlConfigFactory,
)
from models.vit.expert_linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class VitExpertConfigDependencies:
    hidden_dim: int
    encoder_options: TransformerEncoderOptions
    attention_options: TransformerAttentionOptions
    feed_forward_options: TransformerFeedForwardOptions
    mixture_options: ExpertsMixtureOptions
    expert_stack_options: ExpertsSubmoduleStackOptions
    sampler_options: ExpertsSamplerOptions
    router_options: ExpertsRouterOptions
    router_stack_options: ExpertsSubmoduleStackOptions
    expert_layer_controller_options: ExpertsLayerControllerOptions
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions
    expert_attention_use_kv_expert_models_flag: bool


@dataclass(frozen=True)
class VitExpertAdaptiveConfigDependencies(VitExpertConfigDependencies):
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    router_grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    mixture_submodule_stack_options: ExpertsSubmoduleStackOptions
    mixture_layer_controller_options: ExpertsLayerControllerOptions
    mixture_dynamic_memory_options: ExpertsDynamicMemoryOptions
    mixture_recurrent_controller_options: ExpertsRecurrentControllerOptions
    router_layer_controller_options: ExpertsLayerControllerOptions
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions


class VitExpertAdaptiveConfigFactory(
    AdaptiveAttentionExpertBuilder[
        VitExpertAdaptiveConfigDependencies,
        ExpertsStackOptions,
        ExpertAdaptiveControlConfigDependencies,
    ]
):
    _stack = staticmethod(ExpertsStackOptions)
    _control_inputs = staticmethod(ExpertAdaptiveControlConfigDependencies)
    _control = staticmethod(ExpertAdaptiveControlConfigFactory)

    @property
    def _block_options(self) -> TransformerEncoderOptions:
        return self.dependencies.encoder_options
