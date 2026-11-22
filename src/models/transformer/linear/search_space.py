from emperor.layers import LayerNormPositionOptions, NormalizationOptions

SEARCH_SPACE_LEARNING_RATE: list[float] = [0.5, 1.0, 2.0]
SEARCH_SPACE_MODEL_DIM: list[int] = [64, 128, 256]
SEARCH_SPACE_ENCODER_NUM_LAYERS: list[int] = [2, 3, 4]
SEARCH_SPACE_DECODER_NUM_LAYERS: list[int] = [2, 3, 4]
SEARCH_SPACE_ATTN_NUM_HEADS: list[int] = [2, 4, 8]
SEARCH_SPACE_FF_STACK_HIDDEN_DIM: list[int] = [256, 512, 1024]
SEARCH_SPACE_ENCODER_LAYER_NORM_POSITION: list[LayerNormPositionOptions] = [
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
]
SEARCH_SPACE_ENCODER_NORMALIZATION: list[NormalizationOptions] = list(
    NormalizationOptions
)
SEARCH_SPACE_DECODER_LAYER_NORM_POSITION: list[LayerNormPositionOptions] = (
    SEARCH_SPACE_ENCODER_LAYER_NORM_POSITION
)
SEARCH_SPACE_DECODER_NORMALIZATION: list[NormalizationOptions] = list(
    NormalizationOptions
)

# Embedding and output normalization
SEARCH_SPACE_ENCODER_OUTPUT_NORMALIZATION: list = list(NormalizationOptions)
SEARCH_SPACE_DECODER_OUTPUT_NORMALIZATION: list = list(NormalizationOptions)
