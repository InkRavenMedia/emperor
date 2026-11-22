from __future__ import annotations

from model_runtime.packages import ModelIdentity, ModelPackage
from model_runtime.packages.settings_validation import BuildSettingsValidation
from models._package_adapter import ModelPackageAdapter

SETTINGS_VALIDATION = BuildSettingsValidation(
    "models.parametric.parametric_vector",
    allow_integer_floats=False,
    allow_full_dropout=False,
)

_IDENTITY = ModelIdentity("parametric", "parametric_vector")


class _ModelPackageAdapter(ModelPackageAdapter):
    package_name = __name__
    identity = _IDENTITY


MODEL_PACKAGE = ModelPackage(_IDENTITY, _ModelPackageAdapter())

__all__ = ["MODEL_PACKAGE"]
