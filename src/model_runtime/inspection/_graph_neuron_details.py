from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Any, Protocol, cast

from torch import Tensor

from model_runtime.inspection._graph_accounting import GraphModule
from model_runtime.inspection.capture_limits import InspectionCaptureLimits
from model_runtime.inspection.errors import InspectionError

_MISSING_ATTRIBUTE = object()


class _NeuronClusterRemainder(Protocol):
    y_axis_total_neurons: Any
    z_axis_total_neurons: Any


@dataclass(frozen=True, slots=True)
class _NeuronClusterObservation:
    owner: _NeuronClusterRemainder
    x_axis_total_neurons: Any
    cluster: Any


class _TerminalReachRemainder(Protocol):
    y_axis_position: Any
    z_axis_position: Any


@dataclass(frozen=True, slots=True)
class _TerminalReachObservation:
    owner: _TerminalReachRemainder
    neuron_connections: Any
    x_axis_position: Any


def _neuron_cluster_capability(
    module: GraphModule,
) -> _NeuronClusterObservation | None:
    x_axis_total_neurons = getattr(
        module,
        "x_axis_total_neurons",
        _MISSING_ATTRIBUTE,
    )
    if x_axis_total_neurons is _MISSING_ATTRIBUTE:
        return None
    cluster = getattr(module, "cluster", _MISSING_ATTRIBUTE)
    if cluster is _MISSING_ATTRIBUTE:
        return None
    return _NeuronClusterObservation(
        owner=cast(_NeuronClusterRemainder, module),
        x_axis_total_neurons=x_axis_total_neurons,
        cluster=cluster,
    )


def _coordinate_from_neuron_name(name: str) -> list[int] | None:
    parts = name.split("_")
    if len(parts) != 4 or parts[0] != "neuron":
        return None
    try:
        return [int(parts[1]), int(parts[2]), int(parts[3])]
    except ValueError:
        return None


class NeuronDetails:
    @staticmethod
    def _cluster_details(
        module: GraphModule,
        limits: InspectionCaptureLimits,
    ) -> dict[str, Any] | None:
        cluster_observation = _neuron_cluster_capability(module)
        if cluster_observation is None:
            return None
        cluster = cluster_observation.cluster
        total_coordinates = len(cluster)
        cluster_names = tuple(
            islice(
                cast(Iterable[str], cluster.keys()),
                limits.maximum_neuron_coordinates,
            )
        )
        coordinates = sorted(
            coordinate
            for coordinate in (
                _coordinate_from_neuron_name(name) for name in cluster_names
            )
            if coordinate is not None
        )
        coordinates_truncated = total_coordinates > len(cluster_names)
        return {
            "capacity": [
                cluster_observation.x_axis_total_neurons,
                cluster_observation.owner.y_axis_total_neurons,
                cluster_observation.owner.z_axis_total_neurons,
            ],
            "initial": [
                getattr(module, "initial_x_axis_total_neurons", None),
                getattr(module, "initial_y_axis_total_neurons", None),
                getattr(module, "initial_z_axis_total_neurons", None),
            ],
            "initialStart": [
                getattr(module, "initial_x_axis_start", 1),
                getattr(module, "initial_y_axis_start", 1),
                getattr(module, "initial_z_axis_start", 1),
            ],
            "instantiated": total_coordinates,
            "coordinates": coordinates,
            "coordinatesTotal": total_coordinates,
            "coordinatesInvalid": len(cluster_names) - len(coordinates),
            "coordinatesTruncated": coordinates_truncated,
            "coordinatesTruncationReason": (
                "maximum_neuron_coordinates" if coordinates_truncated else None
            ),
            "maxSteps": getattr(module, "max_steps", None),
            "growthThreshold": getattr(module, "growth_threshold", None),
        }

    @staticmethod
    def _terminal_reach_details(
        module: GraphModule,
        limits: InspectionCaptureLimits,
    ) -> dict[str, Any] | None:
        source: object = module
        connections = getattr(module, "neuron_connections", _MISSING_ATTRIBUTE)
        if connections is _MISSING_ATTRIBUTE:
            source = getattr(module, "terminal", module)
            connections = (
                None
                if source is module
                else getattr(source, "neuron_connections", None)
            )
        if connections is None:
            return None
        x_axis_position = getattr(source, "x_axis_position", _MISSING_ATTRIBUTE)
        if x_axis_position is _MISSING_ATTRIBUTE:
            return None
        valid_shape = isinstance(connections, Tensor) and connections.shape[1:] == (3,)
        if not valid_shape:
            raise InspectionError(
                "Terminal neuron_connections must be a Tensor shaped [connections, 3]."
            )
        terminal = _TerminalReachObservation(
            owner=cast(_TerminalReachRemainder, source),
            neuron_connections=connections,
            x_axis_position=x_axis_position,
        )
        observed_connections = terminal.neuron_connections
        total = int(observed_connections.shape[0])
        maximum = limits.maximum_terminal_connections
        selected_connections = observed_connections[:maximum]
        truncated = total > maximum
        return {
            "position": [
                terminal.x_axis_position,
                terminal.owner.y_axis_position,
                terminal.owner.z_axis_position,
            ],
            "connections": selected_connections.detach().cpu().tolist(),
            "total": total,
            "truncated": truncated,
            "truncationReason": ("maximum_terminal_connections" if truncated else None),
        }

    def append(
        self,
        module: GraphModule,
        limits: InspectionCaptureLimits,
        details: dict[str, Any],
    ) -> bool:
        cluster = self._cluster_details(module, limits)
        if cluster is not None:
            halting_model = getattr(module, "halting_model", _MISSING_ATTRIBUTE)
            if halting_model is not _MISSING_ATTRIBUTE:
                details["halting"] = halting_model is not None
            details["cluster"] = cluster

        terminal_reach = self._terminal_reach_details(
            module,
            limits,
        )
        if terminal_reach is not None:
            details["terminalReach"] = terminal_reach
        return cluster is not None
