"""Package-owned layer template; runtime adaptive configurations stay explicit."""

from copy import deepcopy
from dataclasses import fields

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
)


class AdaptiveLinearLayerTemplateConfig(AdaptiveLinearLayerConfig):
    """Construct a concrete configuration when the owning layer supplies dimensions."""

    def build(self, overrides: AdaptiveLinearLayerConfig | None = None):
        config = deepcopy(self)
        if overrides is not None:
            config.update(deepcopy(overrides))
        augmentation = config.adaptive_augmentation_config
        if augmentation is not None:
            for variant in (augmentation.weight_config, augmentation.bias_config):
                if not isinstance(
                    variant, (MatrixWeightsMixtureConfig, MatrixBiasMixtureConfig)
                ):
                    continue
                sampler = variant.sampler_config
                if sampler is None or sampler.router_config is None:
                    continue
                router = sampler.router_config
                if router.input_dim is None:
                    router.input_dim = config.input_dim
                if router.model_config is None:
                    router.model_config = deepcopy(
                        variant.model_config
                        if variant.model_config is not None
                        else augmentation.model_config
                    )
        concrete_config = AdaptiveLinearLayerConfig(
            **{field.name: getattr(config, field.name) for field in fields(config)}
        )
        return concrete_config.build()
