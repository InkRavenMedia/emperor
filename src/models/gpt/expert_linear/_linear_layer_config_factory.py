from dataclasses import dataclass

import models.gpt.expert_linear.config as config
from emperor.layers import (
    ActivationOptions,
)
from models._linear_stack import LinearStackFactory
from models.gpt.expert_linear._config_defaults import gpt_decoder_options
from models.gpt.expert_linear.runtime_options import (
    TransformerDecoderOptions,
)


@dataclass(frozen=True)
class LinearLayerConfigDependencies:
    decoder_options: TransformerDecoderOptions | None


class LinearLayerConfigFactory(LinearStackFactory):
    def __init__(self, dependencies: LinearLayerConfigDependencies) -> None:
        decoder_options = dependencies.decoder_options
        if decoder_options is None:
            decoder_options = gpt_decoder_options(config)
        self.decoder_options = decoder_options
        self.hidden_dim = self.decoder_options.hidden_dim

    @property
    def _default_activation(self) -> ActivationOptions:
        return self.decoder_options.activation
