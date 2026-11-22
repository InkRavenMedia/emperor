from __future__ import annotations

from model_runtime.packages import ModelIdentity, ModelPackage
from models._package_adapter import ModelPackageAdapter

_IDENTITY = ModelIdentity("bert", "expert_linear_adaptive")


class _ModelPackageAdapter(ModelPackageAdapter):
    package_name = __name__
    identity = _IDENTITY


MODEL_PACKAGE = ModelPackage(_IDENTITY, _ModelPackageAdapter())

__all__ = ["MODEL_PACKAGE"]
