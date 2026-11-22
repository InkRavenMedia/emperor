from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.packages import ModelIdentity, ModelPackage
from models._package_adapter import ModelPackageAdapter

_IDENTITY = ModelIdentity("linears", "linear")


class _ModelPackageAdapter(ModelPackageAdapter):
    package_name = __name__
    identity = _IDENTITY


def _checkpoint_config_interpreter(
    tensor_shapes: Mapping[str, tuple[int, ...]],
) -> Mapping[str, Any]:
    from .checkpoint_metadata import checkpoint_config_overrides

    return checkpoint_config_overrides(tensor_shapes)


MODEL_PACKAGE = ModelPackage(
    _IDENTITY,
    _ModelPackageAdapter(),
    _checkpoint_config_interpreter=_checkpoint_config_interpreter,
)

__all__ = ["MODEL_PACKAGE"]
