from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from model_runtime.packages.runtime_values import validate_runtime_default_values
from models._recurrent_variants import variant_updates
from models._runtime_binding import bind_runtime_options

from . import config
from .runtime_options import RuntimeOptions


@bind_runtime_options(config, variant_updates)
def runtime_from_flat(values: Mapping[str, object] | None = None) -> RuntimeOptions:
    return RuntimeOptions(
        validate_runtime_default_values(
            values,
            package="models.parametric.parametric_generator",
            config_module=config,
        )
    )


DEFAULT_RUNTIME: Final = runtime_from_flat()

__all__ = ["DEFAULT_RUNTIME", "runtime_from_flat"]
