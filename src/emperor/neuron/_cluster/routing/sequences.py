"""Cluster traversal whose unit of execution is a whole image sequence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.neuron._cluster.routing.sequence_halting import SequenceHalting
from emperor.neuron._cluster.routing.sequence_tokens import (
    SequenceTokens,
    read_sequence_history,
)
from emperor.neuron._cluster.routing.state import (
    NeuronClusterRouteState,
    _NeuronClusterForwardContext,
)
from emperor.neuron._options import NeuronSequenceRoutingOptions
from emperor.neuron._trace import NeuronClusterTrace, NeuronClusterTraceStep

if TYPE_CHECKING:
    from emperor.neuron._cluster.model import NeuronCluster
    from emperor.neuron._cluster.routing.delegate import ClusterRoutingDelegate
    from emperor.neuron._cluster.topology import ClusterTopologyDelegate
    from emperor.transformer import TransformerEncoderLayerState


@dataclass
class _Branch:
    rows: Tensor
    weights: Tensor
    tokens: SequenceTokens


class SequenceRouting:
    def __init__(
        self,
        owner: NeuronCluster,
        routing: ClusterRoutingDelegate,
        topology: ClusterTopologyDelegate,
    ):
        self.owner = owner
        self.routing = routing
        self.topology = topology
        self.history_enabled = (
            owner.cfg.neuron_config.sequence_residual_config is not None
        )

    def run(
        self,
        state: TransformerEncoderLayerState,
        return_trace: bool,
        context: _NeuronClusterForwardContext,
    ) -> tuple[TransformerEncoderLayerState, list[NeuronClusterTraceStep]]:
        if state.hidden.ndim != 3 or state.hidden.shape[-1] != self.owner.input_dim:
            raise ValueError(
                "Sequence routing requires [batch, tokens, cluster input_dim]."
            )
        if (
            self.owner.cfg.sequence_routing_mode
            is NeuronSequenceRoutingOptions.SEQUENTIAL
        ):
            return self._sequential(state, return_trace, context)
        return self._learned(state, return_trace, context)

    def _sequential(
        self,
        state: TransformerEncoderLayerState,
        return_trace: bool,
        context: _NeuronClusterForwardContext,
    ) -> tuple[TransformerEncoderLayerState, list[NeuronClusterTraceStep]]:
        history = [SequenceTokens.from_state(state)] if self.history_enabled else []
        steps = []
        names = sorted(self.owner.cluster, key=self.topology.parse_neuron_name)
        for name in names:
            context.called_neuron_names.add(name)
            neuron = self.owner.cluster[name]
            result = neuron(state)
            state = read_sequence_history(neuron, result, history)
            if self.history_enabled:
                history.append(SequenceTokens.from_state(state))
            if return_trace:
                coords = self.topology.parse_neuron_name(name)
                batch = state.hidden.size(0)
                coordinates = torch.tensor(coords, device=state.hidden.device).expand(
                    batch, 1, 3
                )
                valid = torch.ones(
                    (batch, 1), device=state.hidden.device, dtype=torch.bool
                )
                steps.append(
                    self._step(
                        state.hidden.new_ones((batch, 1)),
                        coordinates,
                        valid,
                        ~valid,
                        ~valid[:, 0],
                        valid[:, 0] & (name != names[-1]),
                    )
                )
        return state, steps

    def _learned(
        self,
        state: TransformerEncoderLayerState,
        return_trace: bool,
        context: _NeuronClusterForwardContext,
    ) -> tuple[TransformerEncoderLayerState, list[NeuronClusterTraceStep]]:
        tokens = SequenceTokens.from_state(state)
        history = [tokens] if self.history_enabled else []
        probabilities, coords, loss = self.routing.route_entry_input(tokens.summary())
        if state.loss is not None:
            loss = loss + state.loss
        active = torch.ones(
            tokens.hidden.size(0), device=tokens.hidden.device, dtype=torch.bool
        )
        escaped = torch.zeros_like(active)
        halting = SequenceHalting(self.owner.halting_model)
        steps = []
        for hop in range(self.owner.max_steps + 1):
            raw, valid, escapes, branch_loss = self._visit(
                tokens, probabilities, coords, active, history, context
            )
            if branch_loss is not None:
                loss = loss + branch_loss
            winner = probabilities.argmax(dim=1)
            rows = torch.arange(winner.size(0), device=winner.device)
            winning_valid = valid[rows, winner]
            escaped = escaped | (active & ~winning_valid)
            tokens = halting.update(raw, valid.any(dim=1))
            # A winning escape leaves with the actual mixed output of that round.
            tokens = replace(
                raw,
                hidden=torch.where(escaped[:, None, None], raw.hidden, tokens.hidden),
            )
            halted = halting.halt_mask(tokens)
            active = active & winning_valid & ~halted
            if self.history_enabled:
                history.append(raw)
            if return_trace:
                steps.append(
                    self._step(probabilities, coords, valid, escapes, halted, active)
                )
            if hop == self.owner.max_steps or not active.any():
                break
            route_state = NeuronClusterRouteState(
                hidden=tokens.summary(),
                positions=coords[rows, winner],
                active_mask=active,
                escaped_mask=escaped,
                final_mask=~active,
                halting_state=None,
                loss=loss,
            )
            routes = self.routing.collect_routes(route_state, active)
            probabilities, coords, loss = (
                routes.probabilities,
                routes.coordinates,
                routes.loss,
            )

        tokens, ponder = halting.finish(tokens, raw, escaped)
        if ponder is not None:
            loss = loss + ponder
        return replace(tokens.state(), loss=loss), steps

    def _visit(
        self,
        tokens: SequenceTokens,
        probabilities: Tensor,
        coords: Tensor,
        active: Tensor,
        history: list[SequenceTokens],
        context: _NeuronClusterForwardContext,
    ) -> tuple[SequenceTokens, Tensor, Tensor, Tensor | None]:
        groups, valid, escapes = self.routing.group_valid_process_calls(coords, active)
        branches = []
        loss = None
        for name, selections in groups.items():
            rows, slots = torch.tensor(selections, device=coords.device).unbind(dim=1)
            neuron = self.owner.cluster[name]
            context.called_neuron_names.add(name)
            incoming = tokens.rows(rows)
            result = neuron(incoming.state())
            if self.owner.growth_warmup_steps is not None:
                result = replace(
                    result,
                    hidden=self.routing.blend_warming_up_neuron_output(
                        neuron,
                        incoming.pad_to(result.hidden.size(1)).hidden,
                        result.hidden,
                    ),
                )
            result = read_sequence_history(neuron, result, history, rows)
            branches.append(
                _Branch(
                    rows, probabilities[rows, slots], SequenceTokens.from_state(result)
                )
            )
            if result.loss is not None:
                loss = result.loss if loss is None else loss + result.loss

        # Escaping branches relay their input; finished images keep their last result.
        width = max(
            [
                tokens.hidden.size(1),
                *(branch.tokens.hidden.size(1) for branch in branches),
            ]
        )
        source = tokens.pad_to(width)
        relay = (probabilities * escapes).sum(dim=1)
        relay = torch.where(active, relay, torch.ones_like(relay))
        hidden = source.hidden * relay[:, None, None]
        lengths = (~source.padding).sum(dim=1)
        for branch in branches:
            output = branch.tokens.pad_to(width)
            weighted = output.hidden * branch.weights[:, None, None]
            dtype = torch.promote_types(hidden.dtype, weighted.dtype)
            hidden = hidden.to(dtype).index_add(0, branch.rows, weighted.to(dtype))
            lengths = lengths.scatter_reduce(
                0, branch.rows, (~output.padding).sum(dim=1), reduce="amax"
            )
        padding = (
            source.padding
            if source.attention_mask is not None
            else torch.arange(width, device=hidden.device) >= lengths[:, None]
        )
        return replace(source, hidden=hidden, padding=padding), valid, escapes, loss

    @staticmethod
    def _step(
        probabilities: Tensor,
        coords: Tensor,
        valid: Tensor,
        escapes: Tensor,
        halted: Tensor,
        active: Tensor,
    ) -> NeuronClusterTraceStep:
        return NeuronClusterTraceStep(
            probabilities.detach(),
            coords.detach(),
            valid.detach(),
            escapes.detach(),
            probabilities.argmax(dim=1).detach(),
            halted.detach(),
            active.detach(),
        )

    def trace(
        self, input_shape: tuple[int, ...], steps: list[NeuronClusterTraceStep]
    ) -> NeuronClusterTrace:
        entry, *following = steps
        entry_coordinates = self.owner.entry_coordinates
        if (
            self.owner.cfg.sequence_routing_mode
            is NeuronSequenceRoutingOptions.SEQUENTIAL
        ):
            entry_coordinates = entry_coordinates[:1]
        return NeuronClusterTrace(
            tuple(input_shape),
            entry_coordinates.detach(),
            entry.probabilities,
            entry.selected_coordinates,
            entry.valid_mask,
            entry.escape_mask,
            entry.chosen_branch_indices,
            entry.halt_mask,
            entry.active_mask,
            following,
        )
