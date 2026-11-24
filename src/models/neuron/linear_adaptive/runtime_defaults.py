from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from model_runtime.packages.runtime_values import (
    runtime_default_mapping,
    validate_runtime_default_value_types,
)
from models._recurrent_variants import variant_updates
from models._runtime_binding import bind_runtime_options

from . import config
from ._runtime_defaults_resolver import resolve_neuron_runtime
from .runtime_options import RuntimeOptions


@bind_runtime_options(config, variant_updates)
def runtime_from_flat(values: Mapping[str, object] | None = None) -> RuntimeOptions:
    flat_values = runtime_default_mapping(
        values, package="models.neuron.linear_adaptive"
    )
    validate_runtime_default_value_types(
        flat_values,
        package="models.neuron.linear_adaptive",
        config_module=config,
    )
    return resolve_neuron_runtime(flat_values)


DEFAULT_RUNTIME: Final = runtime_from_flat()

__all__ = ["DEFAULT_RUNTIME", "runtime_from_flat"]
