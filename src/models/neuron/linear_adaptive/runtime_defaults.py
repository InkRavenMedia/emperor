from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from model_runtime.packages.runtime_values import (
    runtime_default_mapping,
    validate_runtime_default_value_types,
)

from . import config
from ._runtime_defaults_resolver import NeuronLinearAdaptiveRuntimeDefaultsResolver
from .runtime_options import RuntimeOptions


def runtime_from_flat(values: Mapping[str, object] | None = None) -> RuntimeOptions:
    flat_values = runtime_default_mapping(
        values, package="models.neuron.linear_adaptive"
    )
    validate_runtime_default_value_types(
        flat_values,
        package="models.neuron.linear_adaptive",
        config_module=config,
    )
    resolver = NeuronLinearAdaptiveRuntimeDefaultsResolver(flat_values)
    return RuntimeOptions(
        {
            "hidden_runtime": resolver.hidden_runtime,
            "cluster_capacity_options": resolver.cluster_capacity_options,
            "terminal_options": resolver.terminal_options,
            "terminal_router_options": resolver.terminal_router_options,
            "terminal_sampler_options": resolver.terminal_sampler_options,
            "cluster_halting_options": resolver.cluster_halting_options,
        }
    )


DEFAULT_RUNTIME: Final = runtime_from_flat()

__all__ = ["DEFAULT_RUNTIME", "runtime_from_flat"]
