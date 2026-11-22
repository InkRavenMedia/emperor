from emperor.layers import NormalizationOptions

SEARCH_SPACE_LEARNING_RATE: list[float] = [0.5, 1.0, 2.0]
SEARCH_SPACE_MODEL_DIM: list[int] = [64, 128, 256]
SEARCH_SPACE_ENCODER_NUM_LAYERS: list[int] = [2, 3, 4]
SEARCH_SPACE_DECODER_NUM_LAYERS: list[int] = [2, 3, 4]
SEARCH_SPACE_ATTN_NUM_HEADS: list[int] = [2, 4, 8]
SEARCH_SPACE_FF_STACK_HIDDEN_DIM: list[int] = [256, 512, 1024]
SEARCH_SPACE_NUM_EXPERTS: list[int] = [4, 8]
SEARCH_SPACE_TOP_K: list[int] = [1, 2]

# Embedding and output normalization
SEARCH_SPACE_ENCODER_OUTPUT_NORMALIZATION: list = list(NormalizationOptions)
SEARCH_SPACE_DECODER_OUTPUT_NORMALIZATION: list = list(NormalizationOptions)
