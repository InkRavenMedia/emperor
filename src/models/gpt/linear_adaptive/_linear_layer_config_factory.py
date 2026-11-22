from dataclasses import dataclass, field

import models.gpt.linear_adaptive._config_defaults as config_defaults
import models.gpt.linear_adaptive.config as config
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
from models.gpt.linear_adaptive._adaptive_hidden_model_config_factory import (
    HiddenModelConfigDependencies,
    HiddenModelConfigFactory,
)
from models.gpt.linear_adaptive._adaptive_linear_template import (
    AdaptiveLinearLayerTemplateConfig,
)
from models.gpt.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    TransformerDecoderOptions,
)


@dataclass(frozen=True)
class LinearLayerConfigDependencies:
    decoder_options: TransformerDecoderOptions | None
    adaptive_augmentation_config: AdaptiveParameterAugmentationConfig | None = None


class LinearLayerConfigFactory(AdaptiveBackendStackFactory):
    layer_template_type = AdaptiveLinearLayerTemplateConfig
    enable_bias_for_adaptive_bias = True

    def __init__(self, dependencies: LinearLayerConfigDependencies) -> None:
        decoder_options = dependencies.decoder_options
        if decoder_options is None:
            decoder_options = config_defaults.gpt_decoder_options(config)
        self.decoder_options = decoder_options
        self.hidden_dim = self.decoder_options.hidden_dim
        self.adaptive_augmentation_config = dependencies.adaptive_augmentation_config

    @property
    def _default_activation(self) -> ActivationOptions:
        return self.decoder_options.activation


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
