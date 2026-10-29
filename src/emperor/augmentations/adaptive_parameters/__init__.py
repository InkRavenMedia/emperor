"""Public Interface for input-adaptive parameter configuration."""

from emperor.augmentations.adaptive_parameters._biases.config import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    DynamicBiasConfig,
    GeneratorDynamicBiasConfig,
    MatrixBiasMixtureConfig,
    MultiplicativeDynamicBiasConfig,
    SigmoidGatedDynamicBiasConfig,
    TanhGatedDynamicBiasConfig,
    WeightedBankDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters._biases.variants.matrix_mixture import (
    MatrixBiasMixture,
)
from emperor.augmentations.adaptive_parameters._config import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters._decay.options import (
    WeightDecayScheduleOptions,
)
from emperor.augmentations.adaptive_parameters._decay.policy import DecayPolicy
from emperor.augmentations.adaptive_parameters._diagonals.config import (
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
    DynamicDiagonalConfig,
    StandardDynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters._grouping.config import (
    AttentionGroupingConfig,
    GroupingConfig,
    MeanGroupingConfig,
    MeanStdGroupingConfig,
    RMSGroupingConfig,
    SumGroupingConfig,
)
from emperor.augmentations.adaptive_parameters._grouping.options import (
    SummaryNormalizationOptions,
)
from emperor.augmentations.adaptive_parameters._masks.config import (
    AxisMaskConfig,
    DiagonalAxisMaskConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    TopSliceAxisMaskConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters._options import (
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    BankExpansionFactorOptions,
    LowRankFactorSourceOptions,
    MaskDimensionOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._weights.config import (
    DiagonallyModulatedLowRankDynamicWeightConfig,
    DualModelDynamicWeightConfig,
    DynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    MatrixWeightsMixtureConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters._weights.normalization import (
    WeightNormalizationPolicy,
)
from emperor.augmentations.adaptive_parameters._weights.variants.matrix_mixture import (
    MatrixWeightsMixture,
)
from emperor.augmentations.adaptive_parameters.monitoring import (
    AdaptiveParameterMonitorCallback,
    WeightBankUtilizationMonitorCallback,
)

__all__ = (
    "DepthMappingHandlerConfig",
    "DepthMappingLayerStack",
    "DecayPolicy",
    "WeightNormalizationPolicy",
    "MatrixWeightsMixtureConfig",
    "MatrixBiasMixtureConfig",
    "MatrixWeightsMixture",
    "MatrixBiasMixture",
    "DiagonallyModulatedLowRankDynamicWeightConfig",
    "LowRankFactorSourceOptions",
    "AttentionGroupingConfig",
    "GroupingConfig",
    "MeanGroupingConfig",
    "MeanStdGroupingConfig",
    "RMSGroupingConfig",
    "SumGroupingConfig",
    "SummaryNormalizationOptions",
    "AdaptiveParameterAugmentationConfig",
    "AdaptiveParameterInputOrderOptions",
    "AdaptiveLinearLayerConfig",
    "DynamicWeightConfig",
    "SingleModelDynamicWeightConfig",
    "DualModelDynamicWeightConfig",
    "LowRankDynamicWeightConfig",
    "HypernetworkDynamicWeightConfig",
    "LayeredWeightedBankDynamicWeightConfig",
    "SoftWeightedBankDynamicWeightConfig",
    "DynamicBiasConfig",
    "AffineTransformDynamicBiasConfig",
    "AdditiveDynamicBiasConfig",
    "MultiplicativeDynamicBiasConfig",
    "SigmoidGatedDynamicBiasConfig",
    "TanhGatedDynamicBiasConfig",
    "GeneratorDynamicBiasConfig",
    "WeightedBankDynamicBiasConfig",
    "DynamicDiagonalConfig",
    "StandardDynamicDiagonalConfig",
    "AntiDynamicDiagonalConfig",
    "CombinedDynamicDiagonalConfig",
    "AxisMaskConfig",
    "WeightInformedScoreAxisMaskConfig",
    "PerAxisScoreMaskConfig",
    "TopSliceAxisMaskConfig",
    "OuterProductMaskConfig",
    "DiagonalAxisMaskConfig",
    "BankExpansionFactorOptions",
    "MaskDimensionOptions",
    "WeightDecayScheduleOptions",
    "WeightNormalizationOptions",
    "WeightNormalizationPositionOptions",
    "AdaptiveParameterGroupingScopeOptions",
    "AdaptiveParameterMonitorCallback",
    "WeightBankUtilizationMonitorCallback",
)
