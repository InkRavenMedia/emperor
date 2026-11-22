from __future__ import annotations

from collections.abc import Mapping

import models.neuron.linear.config as config
from models.neuron._runtime_resolver import NeuronDefaults, NeuronRuntimeResolver
from models.neuron.linear._hidden.runtime_defaults import runtime_from_flat
from models.neuron.linear.runtime_options import (
    ClusterRouteHaltingOptions,
    NeuronClusterCapacityOptions,
    NeuronSubmoduleStackOptions,
    NeuronTerminalOptions,
    NeuronTerminalRoutingTreeOptions,
    NeuronTerminalSamplerOptions,
    RuntimeOptions,
)


class _RuntimeResolver(NeuronRuntimeResolver[RuntimeOptions]):
    defaults = NeuronDefaults.from_config(config)
    capacity_type = NeuronClusterCapacityOptions
    terminal_type = NeuronTerminalOptions
    routing_tree_type = NeuronTerminalRoutingTreeOptions
    stack_type = NeuronSubmoduleStackOptions
    sampler_type = NeuronTerminalSamplerOptions
    halting_type = ClusterRouteHaltingOptions
    runtime_type = RuntimeOptions

    @staticmethod
    def _hidden_settings(values: dict[str, object]) -> tuple[object, object]:
        hidden_runtime = runtime_from_flat(values)
        residual_stack_options = hidden_runtime.stack.residual_stack_options
        return hidden_runtime, residual_stack_options


def resolve_neuron_runtime(values: Mapping[str, object]) -> RuntimeOptions:
    resolver = _RuntimeResolver(values)
    return resolver.resolve()
