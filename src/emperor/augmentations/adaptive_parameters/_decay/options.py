from enum import Enum


class WeightDecayScheduleOptions(Enum):
    DISABLED = 0
    EXPONENTIAL = 1
    LINEAR = 2
    MULTIPLICATIVE = 3
