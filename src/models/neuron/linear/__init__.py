from __future__ import annotations

from model_runtime.packages import (
    InspectionConstructionLimits,
    InspectionFieldProductLimit,
    ModelIdentity,
    ModelPackage,
)
from model_runtime.packages.settings_validation import BuildSettingsValidation
from models._package_adapter import ModelPackageAdapter

SETTINGS_VALIDATION = BuildSettingsValidation(
    "models.neuron.linear", allow_integer_floats=False, allow_full_dropout=False
)

_IDENTITY = ModelIdentity("neuron", "linear")


class _ModelPackageAdapter(ModelPackageAdapter):
    package_name = __name__
    identity = _IDENTITY


_INSPECTION_LIMITS = InspectionConstructionLimits(
    field_maximums={"CLUSTER_BEAM_WIDTH": 64},
    field_product_limits=(
        InspectionFieldProductLimit(
            label="initial neuron count",
            factors=(
                (
                    "CLUSTER_INITIAL_X_AXIS_TOTAL_NEURONS",
                    "CLUSTER_X_AXIS_TOTAL_NEURONS",
                ),
                (
                    "CLUSTER_INITIAL_Y_AXIS_TOTAL_NEURONS",
                    "CLUSTER_Y_AXIS_TOTAL_NEURONS",
                ),
                (
                    "CLUSTER_INITIAL_Z_AXIS_TOTAL_NEURONS",
                    "CLUSTER_Z_AXIS_TOTAL_NEURONS",
                ),
            ),
            maximum=4_096,
            repeats_dense_parameter_estimate=True,
        ),
        InspectionFieldProductLimit(
            label="neuron capacity",
            factors=(
                ("CLUSTER_X_AXIS_TOTAL_NEURONS",),
                ("CLUSTER_Y_AXIS_TOTAL_NEURONS",),
                ("CLUSTER_Z_AXIS_TOTAL_NEURONS",),
            ),
            maximum=4_096,
        ),
    ),
)

MODEL_PACKAGE = ModelPackage(
    _IDENTITY,
    _ModelPackageAdapter(),
    inspection_construction_limits=_INSPECTION_LIMITS,
)

__all__ = ["MODEL_PACKAGE"]
