from dataclasses import dataclass

from models._attention_experts import ExpertBlockSettings, RoutedAttentionExpertBuilder
from models.gpt.expert_linear._expert_control_config_factory import (
    ControlConfigDependencies as ExpertLinearControlConfigDependencies,
)
from models.gpt.expert_linear._expert_control_config_factory import (
    ControlConfigFactory as ExpertLinearControlConfigFactory,
)
from models.gpt.expert_linear.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class GptExpertConfigDependencies:
    hidden_dim: int
    decoder_options: TransformerDecoderOptions
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


class GptExpertConfigFactory(
    RoutedAttentionExpertBuilder[
        GptExpertConfigDependencies,
        ExpertsStackOptions,
        ExpertLinearControlConfigDependencies,
    ]
):
    _stack = staticmethod(ExpertsStackOptions)
    _control_inputs = staticmethod(ExpertLinearControlConfigDependencies)
    _control = staticmethod(ExpertLinearControlConfigFactory)

    @property
    def _block_options(self) -> TransformerDecoderOptions:
        return self.dependencies.decoder_options

    def _causal_attention(self, options: ExpertBlockSettings) -> bool:
        return True
