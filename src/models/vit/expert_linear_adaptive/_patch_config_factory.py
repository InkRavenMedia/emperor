from dataclasses import dataclass

import models.vit.expert_linear_adaptive.config as config
from models.vit._patch_projection import ConfiguredPatchProjection
from models.vit.expert_linear_adaptive import _config_defaults as config_defaults
from models.vit.expert_linear_adaptive._linear_layer_config_factory import (
    LinearLayerConfigFactory,
)
from models.vit.expert_linear_adaptive.runtime_options import (
    TransformerEncoderOptions,
    VitPatchOptions,
)


@dataclass(frozen=True)
class PatchConfigDependencies:
    hidden_dim: int
    patch_options: VitPatchOptions | None
    encoder_options: TransformerEncoderOptions | None
    linear_layer_config_factory: LinearLayerConfigFactory


class PatchConfigFactory(ConfiguredPatchProjection[PatchConfigDependencies]):
    _config = config
    _defaults = config_defaults
