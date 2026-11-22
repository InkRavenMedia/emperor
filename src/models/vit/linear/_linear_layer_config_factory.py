from dataclasses import dataclass

import models.vit.linear.config as config
from emperor.layers import (
    ActivationOptions,
)
from models._linear_stack import LinearStackFactory
from models.vit.linear._config_defaults import vit_encoder_options
from models.vit.linear.runtime_options import (
    TransformerEncoderOptions,
)


@dataclass(frozen=True)
class LinearLayerConfigDependencies:
    encoder_options: TransformerEncoderOptions | None


class LinearLayerConfigFactory(LinearStackFactory):
    def __init__(self, dependencies: LinearLayerConfigDependencies) -> None:
        encoder_options = dependencies.encoder_options
        if encoder_options is None:
            encoder_options = vit_encoder_options(config)
        self.encoder_options = encoder_options
        self.hidden_dim = self.encoder_options.hidden_dim

    @property
    def _default_activation(self) -> ActivationOptions:
        return self.encoder_options.activation
