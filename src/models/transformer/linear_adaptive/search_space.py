from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    AntiDynamicDiagonalConfig,
    AxisMaskConfig,
    BroadcastSumDynamicWeightConfig,
    CombinedDynamicDiagonalConfig,
    DiagonalAxisMaskConfig,
    DiagonallyModulatedLowRankDynamicWeightConfig,
    DualModelDynamicWeightConfig,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GeneratorDynamicBiasConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    MultiplicativeDynamicBiasConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    SigmoidGatedDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    StandardDynamicDiagonalConfig,
    TanhGatedDynamicBiasConfig,
    TopSliceAxisMaskConfig,
    WeightedBankDynamicBiasConfig,
    WeightInformedScoreAxisMaskConfig,
)
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
SEARCH_SPACE_PROJECTION_ADAPTIVE_WEIGHT_OPTION: list[
    type[DynamicWeightConfig] | None
] = [
    None,
    SingleModelDynamicWeightConfig,
    BroadcastSumDynamicWeightConfig,
    DualModelDynamicWeightConfig,
    DiagonallyModulatedLowRankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
]
SEARCH_SPACE_FEED_FORWARD_ADAPTIVE_WEIGHT_OPTION: list[
    type[DynamicWeightConfig] | None
] = [
    None,
    BroadcastSumDynamicWeightConfig,
    DualModelDynamicWeightConfig,
    DiagonallyModulatedLowRankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
]
SEARCH_SPACE_PROJECTION_ADAPTIVE_BIAS_OPTION: list[type[DynamicBiasConfig] | None] = [
    None,
    AffineTransformDynamicBiasConfig,
    AdditiveDynamicBiasConfig,
    GeneratorDynamicBiasConfig,
    MultiplicativeDynamicBiasConfig,
    SigmoidGatedDynamicBiasConfig,
    TanhGatedDynamicBiasConfig,
    WeightedBankDynamicBiasConfig,
]
SEARCH_SPACE_FEED_FORWARD_ADAPTIVE_BIAS_OPTION: list[type[DynamicBiasConfig] | None] = (
    SEARCH_SPACE_PROJECTION_ADAPTIVE_BIAS_OPTION
)
SEARCH_SPACE_PROJECTION_ADAPTIVE_DIAGONAL_OPTION: list[
    type[DynamicDiagonalConfig] | None
] = [
    None,
    StandardDynamicDiagonalConfig,
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
]
SEARCH_SPACE_FEED_FORWARD_ADAPTIVE_DIAGONAL_OPTION: list[
    type[DynamicDiagonalConfig] | None
] = SEARCH_SPACE_PROJECTION_ADAPTIVE_DIAGONAL_OPTION
SEARCH_SPACE_PROJECTION_ADAPTIVE_ROW_MASK_OPTION: list[type[AxisMaskConfig] | None] = [
    None,
    DiagonalAxisMaskConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    TopSliceAxisMaskConfig,
    WeightInformedScoreAxisMaskConfig,
]
SEARCH_SPACE_FEED_FORWARD_ADAPTIVE_ROW_MASK_OPTION: list[
    type[AxisMaskConfig] | None
] = SEARCH_SPACE_PROJECTION_ADAPTIVE_ROW_MASK_OPTION

# Embedding and output normalization
SEARCH_SPACE_ENCODER_OUTPUT_NORMALIZATION: list = list(NormalizationOptions)
SEARCH_SPACE_DECODER_OUTPUT_NORMALIZATION: list = list(NormalizationOptions)
