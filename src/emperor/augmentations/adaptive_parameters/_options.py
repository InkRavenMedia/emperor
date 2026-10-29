from enum import Enum

from emperor.augmentations.adaptive_parameters._decay.options import (
    WeightDecayScheduleOptions as WeightDecayScheduleOptions,
)


class LowRankFactorSourceOptions(Enum):
    GENERATED = 1
    SHARED_PARAMETER = 2


class AdaptiveParameterGroupingScopeOptions(Enum):
    ROWS = 1
    SEQUENCE = 2


class AdaptiveParameterInputOrderOptions(Enum):
    BATCH_FIRST = 0
    SEQUENCE_FIRST = 1


class WeightNormalizationOptions(Enum):
    DISABLED = 0
    CLAMP = 1
    L2_SCALE = 2
    SOFT_CLAMP = 3
    RMS = 4
    SIGMOID_SCALE = 5


class BankExpansionFactorOptions(Enum):
    DISABLED = 0
    FACTOR_OF_ONE = 1
    FACTOR_OF_TWO = 2
    FACTOR_OF_THREE = 3
    FACTOR_OF_FOUR = 4


class MaskDimensionOptions(Enum):
    ROW = 0
    COLUMN = 1


class WeightNormalizationPositionOptions(Enum):
    DISABLED = 0
    BEFORE_OUTER_PRODUCT = 1
    AFTER_OUTER_PRODUCT = 2
