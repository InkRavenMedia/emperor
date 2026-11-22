from __future__ import annotations

from model_runtime.packages import ModelIdentity, ModelPackage
from model_runtime.packages.settings_validation import BuildSettingsValidation
from models._package_adapter import ModelPackageAdapter

SETTINGS_VALIDATION = BuildSettingsValidation(
    "models.linears.linear_adaptive",
    allow_integer_floats=True,
    allow_full_dropout=True,
    invalid_value_format="runtime key {key!r} {requirement}",
)

_IDENTITY = ModelIdentity("linears", "linear_adaptive")


class _ModelPackageAdapter(ModelPackageAdapter):
    package_name = __name__
    identity = _IDENTITY


MODEL_PACKAGE = ModelPackage(_IDENTITY, _ModelPackageAdapter())

__all__ = ["MODEL_PACKAGE"]
