from __future__ import annotations

from model_runtime.packages import ModelIdentity, ModelPackage
from models._package_adapter import ModelPackageAdapter

_IDENTITY = ModelIdentity("gpt", "expert_linear")


class _ModelPackageAdapter(ModelPackageAdapter):
    package_name = __name__
    identity = _IDENTITY


def _checkpoint_config_interpreter(tensor_shapes):
    from .checkpoint_metadata import checkpoint_config_overrides

    return checkpoint_config_overrides(tensor_shapes)


MODEL_PACKAGE = ModelPackage(
    _IDENTITY,
    _ModelPackageAdapter(),
    _checkpoint_config_interpreter=_checkpoint_config_interpreter,
)

__all__ = ["MODEL_PACKAGE"]
