from dataclasses import dataclass, field

import models.bert.linear_adaptive._config_defaults as config_defaults
import models.bert.linear_adaptive.config as config
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
    GroupingConfig,
)
from emperor.layers import (
    ActivationOptions,
)
from models._adaptive_attention_backend import (
    AdaptiveBackendStackFactory,
    AttentionAugmentationFactory,
)
from models.bert.linear_adaptive._adaptive_hidden_model_config_factory import (
    HiddenModelConfigDependencies,
    HiddenModelConfigFactory,
)
from models.bert.linear_adaptive._adaptive_linear_template import (
    AdaptiveLinearLayerTemplateConfig,
)
from models.bert.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    TransformerEncoderOptions,
)


@dataclass(frozen=True)
class LinearLayerConfigDependencies:
    encoder_options: TransformerEncoderOptions | None
    adaptive_augmentation_config: AdaptiveParameterAugmentationConfig | None = None


class LinearLayerConfigFactory(AdaptiveBackendStackFactory):
    layer_template_type = AdaptiveLinearLayerTemplateConfig
    enable_bias_for_adaptive_bias = True

    def __init__(self, dependencies: LinearLayerConfigDependencies) -> None:
        encoder_options = dependencies.encoder_options
        if encoder_options is None:
            encoder_options = config_defaults.bert_encoder_options(config)
        self.encoder_options = encoder_options
        self.hidden_dim = self.encoder_options.hidden_dim
        self.adaptive_augmentation_config = dependencies.adaptive_augmentation_config

    @property
    def _default_activation(self) -> ActivationOptions:
        return self.encoder_options.activation


@dataclass(frozen=True)
class AdaptiveAugmentationDependencies:
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    hidden_dim: int
    output_dim: int
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None


class AdaptiveAugmentationConfigFactory(
    AttentionAugmentationFactory[
        AdaptiveGeneratorStackOptions, HiddenModelConfigDependencies
    ]
):
    hidden_options_type = HiddenModelConfigDependencies
    hidden_factory_type = HiddenModelConfigFactory
    use_defaults_for_falsey_options = True

    def __init__(self, dependencies: AdaptiveAugmentationDependencies) -> None:
        super().__init__(dependencies, config, config_defaults)
