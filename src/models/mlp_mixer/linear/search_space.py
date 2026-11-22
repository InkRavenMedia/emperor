from typing import Annotated

from emperor.layers import LayerNormPositionOptions, NormalizationOptions
from model_runtime.packages.configuration_metadata import Section

from .config import *  # noqa: F401,F403

_optimization = Section(("Optimization",))
SEARCH_SPACE_LEARNING_RATE: Annotated[list, _optimization] = [1e-4, 1e-3, 1e-2]

_model = Section(("Model",))
SEARCH_SPACE_HIDDEN_DIM: Annotated[list, _model] = [16, 32, 64, 128]

# Image patches
SEARCH_SPACE_IMAGE_PATCH_SIZE: Annotated[list, _model] = [1, 2, 4]

# Mixer blocks
SEARCH_SPACE_STACK_NUM_LAYERS: Annotated[list, _model] = [1, 2, 4, 8]
SEARCH_SPACE_LAYER_NORM_POSITION: Annotated[list, _model] = [
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
]
SEARCH_SPACE_NORMALIZATION: Annotated[list, _model] = list(NormalizationOptions)

# Token mixer
SEARCH_SPACE_TOKEN_MIXER_STACK_HIDDEN_DIM: Annotated[list, _model] = [32, 64, 128]

# Channel mixer
SEARCH_SPACE_CHANNEL_MIXER_STACK_HIDDEN_DIM: Annotated[list, _model] = [64, 128, 256]

# Embedding and output normalization
SEARCH_SPACE_ENCODER_OUTPUT_NORMALIZATION: Annotated[list, _model] = list(
    NormalizationOptions
)
