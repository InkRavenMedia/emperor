from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from typing import TYPE_CHECKING

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    MatrixBiasMixtureConfig,
)
from emperor.config import ConfigBase

if TYPE_CHECKING:
    from emperor.nn import Module


class AdaptiveLinearLayerTemplateConfig(AdaptiveLinearLayerConfig):
    """Construct a concrete configuration when the owning layer supplies dimensions."""

    def build(self, overrides: ConfigBase | None = None) -> Module:
        concrete = self.__resolve_for_layer(overrides)
        adaptive_layer = concrete.build()
        return adaptive_layer

    def __resolve_for_layer(
        self, overrides: ConfigBase | None
    ) -> AdaptiveLinearLayerConfig:
        template = deepcopy(self)
        if overrides is not None:
            template.update(deepcopy(overrides))
        template.__resolve_mixture_routers()
        return AdaptiveLinearLayerConfig(
            **{field.name: getattr(template, field.name) for field in fields(template)}
        )

    def __resolve_mixture_routers(self) -> None:
        augmentation = self.adaptive_augmentation_config
        if augmentation is None:
            return
        if isinstance(augmentation.bias_config, MatrixBiasMixtureConfig):
            self.__inherit_router_defaults(augmentation.bias_config)

    def __inherit_router_defaults(self, mixture: MatrixBiasMixtureConfig) -> None:
        sampler = mixture.sampler_config
        if sampler is None or sampler.router_config is None:
            return
        router = sampler.router_config
        if router.input_dim is None:
            router.input_dim = self.input_dim
        if router.model_config is not None:
            return
        generator = mixture.model_config
        if generator is None and self.adaptive_augmentation_config is not None:
            generator = self.adaptive_augmentation_config.model_config
        router.model_config = deepcopy(generator)
