from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.packages import ModelIdentity, ModelPackage
from model_runtime.packages.settings_validation import BuildSettingsValidation
from models._package_adapter import ModelPackageAdapter

SETTINGS_VALIDATION = BuildSettingsValidation(
    "models.linears.linear", allow_integer_floats=False, allow_full_dropout=False
)

_IDENTITY = ModelIdentity("linears", "linear")


class _ModelPackageAdapter(ModelPackageAdapter):
    package_name = __name__
    identity = _IDENTITY

    def load_runtime_options_type(self) -> type:
        from .runtime_options import LinearBuildSettings

        return LinearBuildSettings

    def bind_runtime_defaults(self, values):
        from .runtime_defaults import build_settings_from_flat

        return build_settings_from_flat(values)


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
