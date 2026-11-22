from dataclasses import dataclass
from typing import cast

from models._attention_experts import ExpertBlockSettings, RoutedAttentionExpertBuilder
from models.bert.expert_linear._expert_control_config_factory import (
    ControlConfigDependencies as ExpertLinearControlConfigDependencies,
)
from models.bert.expert_linear._expert_control_config_factory import (
    ControlConfigFactory as ExpertLinearControlConfigFactory,
)
from models.bert.expert_linear.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class BertExpertConfigDependencies:
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


class BertExpertConfigFactory(
    RoutedAttentionExpertBuilder[
        BertExpertConfigDependencies,
        ExpertsStackOptions,
        ExpertLinearControlConfigDependencies,
    ]
):
    _stack = staticmethod(ExpertsStackOptions)
    _control_inputs = staticmethod(ExpertLinearControlConfigDependencies)
    _control = staticmethod(ExpertLinearControlConfigFactory)

    @property
    def _block_options(self) -> TransformerEncoderOptions:
        return self.dependencies.encoder_options

    def _causal_attention(self, options: ExpertBlockSettings) -> bool:
        return cast(TransformerEncoderOptions, options).causal_attention_mask_flag
