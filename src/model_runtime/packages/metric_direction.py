"""Directions accepted by package-declared early stopping and checkpoint monitors."""

from enum import StrEnum


class MetricDirection(StrEnum):
    MIN = "min"
    MAX = "max"
