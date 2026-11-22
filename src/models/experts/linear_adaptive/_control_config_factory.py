from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
)
from emperor.layers import (
    LayerStackConfig,
)
from models._adaptive_augmentation import AdaptiveAugmentationFactory
from models._controller_stack import build_controller_stack_with_model
from models._expert_router import RoutedExpertSettings, RoutedExpertStackFactory
from models.experts.linear_adaptive._adaptive_generator_stack_config_factory import (
    AdaptiveGeneratorStackConfigFactory,
)
from models.experts.linear_adaptive._adaptive_linear_template import (
    AdaptiveLinearLayerTemplateConfig,
)
from models.experts.linear_adaptive._control_support import (
    ExpertsGateConfigFactory,
    ExpertsHaltingConfigFactory,
    ExpertsMemoryConfigFactory,
    ExpertsRecurrentConfigFactory,
)
from models.experts.linear_adaptive._router_controller_config import (
    RouterControllerModelConfig,
)
from models.experts.linear_adaptive.runtime_options import (
    RuntimeOptions,
)


class ControlConfigFactory(RoutedExpertStackFactory):
    _router_config = RouterControllerModelConfig
    _build_standalone_stack = staticmethod(build_controller_stack_with_model)

    _gate_type = ExpertsGateConfigFactory
    _halting_type = ExpertsHaltingConfigFactory
    _memory_type = ExpertsMemoryConfigFactory
    _recurrent_type = ExpertsRecurrentConfigFactory

    def __init__(self, runtime: RuntimeOptions) -> None:
        self.runtime = runtime
        self.adaptive_generator_stack_config_factory = (
            AdaptiveGeneratorStackConfigFactory(
                runtime.adaptive_generator_stack_options
            )
        )
        augmentation = AdaptiveAugmentationFactory(
            self.adaptive_generator_stack_config_factory,
            default_weight_generation=lambda: None,
        )
        self.hidden_adaptive_augmentation_config = augmentation.build(
            grouping=runtime.grouping_config,
            weight=runtime.hidden_adaptive_weight_options,
            bias=runtime.hidden_adaptive_bias_options,
            diagonal=runtime.hidden_adaptive_diagonal_options,
            mask=runtime.hidden_adaptive_mask_options,
        )
        self.router_adaptive_augmentation_config = augmentation.build(
            grouping=runtime.router_grouping_config,
            weight=runtime.router_adaptive_weight_options,
            bias=runtime.router_adaptive_bias_options,
            diagonal=runtime.router_adaptive_diagonal_options,
            mask=runtime.router_adaptive_mask_options,
        )
        self.hidden_dim = runtime.stack_options.hidden_dim
        self._initialize_controllers()

    @property
    def _expert_settings(self) -> RoutedExpertSettings:
        return self.runtime

    def build_hidden_adaptive_linear_layer_config(self) -> AdaptiveLinearLayerConfig:
        return AdaptiveLinearLayerTemplateConfig(
            bias_flag=self.runtime.expert_stack_options.bias_flag,
            adaptive_augmentation_config=self.hidden_adaptive_augmentation_config,
        )

    def build_router_adaptive_linear_layer_config(self) -> AdaptiveLinearLayerConfig:
        return AdaptiveLinearLayerTemplateConfig(
            bias_flag=self.runtime.router_stack_options.bias_flag,
            adaptive_augmentation_config=self.router_adaptive_augmentation_config,
        )

    def _build_expert_stack(self) -> LayerStackConfig:
        expert_stack_options = self.runtime.expert_stack_options
        layer_model_config = self.build_hidden_adaptive_linear_layer_config()
        return self._assemble_expert_stack(expert_stack_options, layer_model_config)
