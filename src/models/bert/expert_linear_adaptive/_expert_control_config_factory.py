from dataclasses import dataclass, field
from types import EllipsisType

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    GroupingConfig,
)
from emperor.layers import (
    LayerStackConfig,
)
from models._configured_experts import AdaptiveRoutedExpertFactory
from models.bert.expert_linear_adaptive._adaptive_generator_stack_config_factory import (
    AdaptiveGeneratorStackConfigFactory,
)
from models.bert.expert_linear_adaptive._adaptive_linear_template import (
    AdaptiveLinearLayerTemplateConfig,
)
from models.bert.expert_linear_adaptive._control_support import (
    ExpertsGateConfigFactory,
    ExpertsHaltingConfigFactory,
    ExpertsMemoryConfigFactory,
    ExpertsRecurrentConfigFactory,
)
from models.bert.expert_linear_adaptive._router_controller_config import (
    RouterControllerModelConfig,
)
from models.bert.expert_linear_adaptive.runtime_options import (
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
)


@dataclass(frozen=True)
class ControlConfigDependencies:
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    router_grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    stack_options: ExpertsStackOptions
    submodule_stack_options: ExpertsSubmoduleStackOptions
    mixture_options: ExpertsMixtureOptions
    expert_stack_options: ExpertsSubmoduleStackOptions
    sampler_options: ExpertsSamplerOptions
    router_options: ExpertsRouterOptions
    router_stack_options: ExpertsSubmoduleStackOptions
    router_layer_controller_options: ExpertsLayerControllerOptions
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions
    layer_controller_options: ExpertsLayerControllerOptions
    dynamic_memory_options: ExpertsDynamicMemoryOptions
    recurrent_controller_options: ExpertsRecurrentControllerOptions
    expert_layer_controller_options: ExpertsLayerControllerOptions
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions
    hidden_dim: int
    output_dim: int


class ControlConfigFactory(AdaptiveRoutedExpertFactory[ControlConfigDependencies]):
    _generator = staticmethod(AdaptiveGeneratorStackConfigFactory)
    _router_config = RouterControllerModelConfig

    _gate_type = ExpertsGateConfigFactory
    _halting_type = ExpertsHaltingConfigFactory
    _memory_type = ExpertsMemoryConfigFactory
    _recurrent_type = ExpertsRecurrentConfigFactory

    def build_hidden_adaptive_linear_layer_config(
        self,
        bias_flag: bool,
    ) -> AdaptiveLinearLayerConfig:
        return AdaptiveLinearLayerTemplateConfig(
            bias_flag=bias_flag,
            adaptive_augmentation_config=self.hidden_adaptive_augmentation_config,
        )

    def build_router_adaptive_linear_layer_config(
        self,
        bias_flag: bool | EllipsisType = ...,
    ) -> AdaptiveLinearLayerConfig:
        if bias_flag is ...:
            bias_flag = self.router_stack_options.bias_flag
        return AdaptiveLinearLayerTemplateConfig(
            bias_flag=bias_flag,
            adaptive_augmentation_config=self.router_adaptive_augmentation_config,
        )

    def _build_expert_stack(self) -> LayerStackConfig:
        expert_stack_options = self.expert_stack_options
        layer_model_config = self.build_hidden_adaptive_linear_layer_config(
            expert_stack_options.bias_flag
        )
        return self._assemble_expert_stack(expert_stack_options, layer_model_config)
