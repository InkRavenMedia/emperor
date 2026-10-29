"""Construction, input dispatch, and execution of independent expert networks."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from emperor.experts._config import MixtureOfExpertsConfig
from emperor.experts._routing.capacity import ExpertCapacityHandler
from emperor.experts._routing.weighting import ExpertWeightingHandler
from emperor.experts._validation.experts import ExpertsValidator
from emperor.layers import Layer, LayerStackConfig, RecurrentCompositionConfig


@dataclass
class ExpertInputData:
    expert_index: int
    expert_samples: Tensor
    dropped_samples: Tensor
    expert_routing_positions: Tensor | None
    dropped_routing_positions: Tensor | None
    probabilities: Tensor | None
    dropped_probabilities: Tensor | None = None


class Experts(nn.ModuleList):
    """Own experts under numeric child names, preserving their checkpoint keys."""

    VALIDATOR = ExpertsValidator

    def __init__(self, cfg: MixtureOfExpertsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.num_experts: int = self.cfg.num_experts
        self.top_k: int = self.cfg.top_k
        self.capacity_handler = ExpertCapacityHandler(self.cfg)
        self.expert_weighting_handler = ExpertWeightingHandler(self.cfg)
        self.expert_model_config: LayerStackConfig | RecurrentCompositionConfig = (
            self.cfg.expert_model_config
        )
        self.__initialize_experts()

    def __initialize_experts(self) -> None:
        for _ in range(self.num_experts):
            expert_dimension_overrides = LayerStackConfig(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
            )
            expert_model = self.expert_model_config.build(expert_dimension_overrides)
            self.append(expert_model)

    def __getitem__(self, index: int | slice) -> nn.Module:
        # ModuleList slicing must select existing modules without constructing experts.
        if isinstance(index, slice):
            expert_models = list(self)
            selected_experts = expert_models[index]
            return nn.ModuleList(selected_experts)
        return super().__getitem__(index)

    def forward(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        """Dispatch a token batch, execute its experts, and return route metadata."""
        expert_input_batches = self.__split_tokens_per_expert(
            input_batch, probabilities, indices
        )
        return self.__compute_expert_batches(expert_input_batches, probabilities)

    def forward_assignments(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        """Execute inputs that already contain one row per routing assignment."""
        expert_input_batches = self.__build_assignment_expert_inputs(
            input_batch, probabilities, indices
        )
        return self.__compute_expert_batches(expert_input_batches, probabilities)

    def forward_soft_tokens(
        self,
        input_batch: Tensor,
        dispatch_weights: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Pool a token batch into expert slots and execute every expert's slots."""
        input_slots = self.__dispatch_tokens_to_slots(
            input_batch, dispatch_weights, skip_mask
        )
        return self.__compute_slot_experts(input_slots)

    def __run_expert(
        self, expert_index: int, expert_samples: Tensor
    ) -> tuple[Tensor, Tensor]:
        expert_model = self[expert_index]
        expert_state = Layer.run_model_from_hidden(expert_model, expert_samples)
        self.VALIDATOR.validate_output_rows(self, expert_samples, expert_state.hidden)
        expert_loss = self.__resolve_expert_loss(expert_state.loss, expert_samples)
        return expert_state.hidden, expert_loss

    def __compute_slot_experts(self, input_slots: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, _, slots_per_expert, _ = input_slots.shape
        expert_slot_outputs = []
        total_expert_loss = input_slots.new_zeros(())
        for expert_index in range(self.num_experts):
            expert_samples = self.__flatten_expert_slots(input_slots, expert_index)
            expert_output, expert_loss = self.__run_expert(expert_index, expert_samples)
            expert_slot_output_shape = (batch_size, slots_per_expert, self.output_dim)
            expert_slot_output = expert_output.reshape(expert_slot_output_shape)
            expert_slot_outputs.append(expert_slot_output)
            total_expert_loss = total_expert_loss + expert_loss
        output_slots = torch.stack(expert_slot_outputs, dim=1)
        return output_slots, total_expert_loss

    def __flatten_expert_slots(self, input_slots: Tensor, expert_index: int) -> Tensor:
        selected_expert_slots = input_slots[:, expert_index]
        return selected_expert_slots.reshape(-1, self.input_dim)

    def __resolve_expert_loss(
        self, expert_loss: Tensor | None, expert_samples: Tensor
    ) -> Tensor:
        if expert_loss is None:
            return expert_samples.new_zeros(())
        return expert_loss

    def __split_tokens_per_expert(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> list[ExpertInputData]:
        if self.num_experts == self.top_k:
            return self.__build_dense_expert_inputs(input_batch, probabilities)
        return self.__build_routed_expert_inputs(input_batch, probabilities, indices)

    def __build_dense_expert_inputs(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None,
    ) -> list[ExpertInputData]:
        no_dropped_samples = torch.tensor(
            [], dtype=input_batch.dtype, device=input_batch.device
        )
        expert_input_batches = []
        for expert_index in range(self.num_experts):
            expert_probabilities = (
                self.expert_weighting_handler.maybe_get_expert_probabilities(
                    None, probabilities, expert_index
                )
            )
            expert_input = ExpertInputData(
                expert_index=expert_index,
                expert_samples=input_batch,
                dropped_samples=no_dropped_samples,
                expert_routing_positions=None,
                dropped_routing_positions=None,
                probabilities=expert_probabilities,
            )
            expert_input_batches.append(expert_input)
        return expert_input_batches

    def __build_routed_expert_inputs(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None,
        indices: Tensor,
    ) -> list[ExpertInputData]:
        expert_input_batches = []
        for expert_index in range(self.num_experts):
            retained_token_indices, dropped_token_indices = (
                self.__get_expert_token_indices(indices, expert_index)
            )
            if self.__should_skip_expert_without_retained_tokens(
                retained_token_indices
            ):
                continue
            expert_routing_positions, dropped_routing_positions = (
                self.__get_expert_routing_positions(indices, expert_index)
            )
            expert_samples, dropped_samples = (
                self.capacity_handler.select_expert_and_dropped_samples(
                    input_batch, retained_token_indices, dropped_token_indices
                )
            )
            expert_probabilities = (
                self.expert_weighting_handler.maybe_get_expert_probabilities(
                    expert_routing_positions, probabilities, expert_index
                )
            )
            dropped_probabilities = (
                self.expert_weighting_handler.maybe_get_expert_probabilities(
                    dropped_routing_positions,
                    probabilities,
                    expert_index,
                )
            )
            expert_input = ExpertInputData(
                expert_index=expert_index,
                expert_samples=expert_samples,
                dropped_samples=dropped_samples,
                expert_routing_positions=expert_routing_positions,
                dropped_routing_positions=dropped_routing_positions,
                probabilities=expert_probabilities,
                dropped_probabilities=dropped_probabilities,
            )
            expert_input_batches.append(expert_input)
        return expert_input_batches

    def __get_expert_token_indices(
        self,
        indices: Tensor,
        expert_index: int,
    ) -> tuple[Tensor, Tensor]:
        batch_size = indices.size(0)
        expert_assignment_mask = indices == expert_index
        if self.__has_expert_choice_dimension(indices):
            samples_routed_to_expert = expert_assignment_mask.sum(dim=-1)
        else:
            samples_routed_to_expert = expert_assignment_mask
        routed_token_index_coordinates = samples_routed_to_expert.nonzero()
        token_indices_for_expert = routed_token_index_coordinates.flatten()
        return self.capacity_handler.maybe_apply_capacity_limit_token_indices(
            token_indices_for_expert,
            batch_size,
            training=self.training,
        )

    def __has_expert_choice_dimension(self, indices: Tensor) -> bool:
        return indices.dim() > 1

    def __should_skip_expert_without_retained_tokens(
        self,
        retained_token_indices: Tensor,
    ) -> bool:
        return retained_token_indices.numel() == 0

    def __get_expert_routing_positions(
        self,
        indices: Tensor,
        expert_index: int,
    ) -> tuple[Tensor, Tensor]:
        batch_size = indices.size(0)
        expert_assignment_mask = indices == expert_index
        flattened_expert_assignment_mask = expert_assignment_mask.flatten()
        expert_routing_position_coordinates = flattened_expert_assignment_mask.nonzero()
        routing_positions_for_expert = expert_routing_position_coordinates.squeeze(
            dim=-1
        )
        return self.capacity_handler.maybe_apply_capacity_limit_routing_positions(
            routing_positions_for_expert, batch_size
        )

    def __compute_expert_batches(
        self,
        expert_input_batches: list[ExpertInputData],
        full_probabilities: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        outputs_grouped_by_expert = []
        routing_positions_by_expert = []
        total_loss = expert_input_batches[0].expert_samples.new_zeros(())

        for expert_input in expert_input_batches:
            expert_output, expert_loss = self.__compute_expert_output(expert_input)
            self.__append_expert_output(
                outputs_grouped_by_expert,
                expert_output,
                expert_input,
            )
            self.__append_routing_positions(routing_positions_by_expert, expert_input)
            total_loss = total_loss + expert_loss

        expert_outputs = torch.cat(outputs_grouped_by_expert, dim=0)
        expert_outputs = self.__restore_dense_sample_order(expert_outputs)
        routing_positions, routing_probabilities = self.__collect_routing_positions(
            routing_positions_by_expert, full_probabilities
        )

        return (
            expert_outputs,
            routing_positions,
            routing_probabilities,
            total_loss,
        )

    def __restore_dense_sample_order(self, expert_outputs: Tensor) -> Tensor:
        if self.top_k != self.num_experts:
            return expert_outputs
        output_dim = expert_outputs.size(-1)
        expert_major_outputs = expert_outputs.reshape(self.num_experts, -1, output_dim)
        sample_major_outputs = expert_major_outputs.transpose(0, 1)
        return sample_major_outputs.reshape(-1, output_dim)

    def __compute_expert_output(
        self,
        expert_input: ExpertInputData,
    ) -> tuple[Tensor, Tensor]:
        expert_samples = self.expert_weighting_handler.maybe_apply_probabilities_before(
            expert_input.expert_samples, expert_input.probabilities
        )

        return self.__run_expert(expert_input.expert_index, expert_samples)

    def __append_expert_output(
        self,
        outputs_grouped_by_expert: list[Tensor],
        expert_output: Tensor,
        expert_input: ExpertInputData,
    ) -> None:
        outputs_for_expert_routes = expert_output
        if self.__should_append_dropped_token_fallbacks(expert_input):
            dropped_samples = (
                self.expert_weighting_handler.maybe_apply_probabilities_before(
                    expert_input.dropped_samples,
                    expert_input.dropped_probabilities,
                )
            )
            expert_outputs_and_dropped_token_fallbacks = [
                expert_output,
                dropped_samples,
            ]
            outputs_for_expert_routes = torch.cat(
                expert_outputs_and_dropped_token_fallbacks,
                dim=0,
            )
        outputs_grouped_by_expert.append(outputs_for_expert_routes)

    def __should_append_dropped_token_fallbacks(
        self,
        expert_input: ExpertInputData,
    ) -> bool:
        return expert_input.dropped_samples.numel() > 0

    def __append_routing_positions(
        self,
        routing_positions_by_expert: list[Tensor],
        expert_input: ExpertInputData,
    ) -> None:
        if self.top_k != self.num_experts:
            retained_and_dropped_routing_positions = [
                expert_input.expert_routing_positions,
                expert_input.dropped_routing_positions,
            ]
            routing_positions_for_expert = torch.cat(
                retained_and_dropped_routing_positions, dim=0
            )
            routing_positions_by_expert.append(routing_positions_for_expert)

    def __collect_routing_positions(
        self,
        routing_positions_by_expert: list[Tensor],
        full_probabilities: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.top_k == self.num_experts:
            return None, full_probabilities
        routing_positions = torch.cat(routing_positions_by_expert)
        return routing_positions, full_probabilities

    def __build_assignment_expert_inputs(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> list[ExpertInputData]:
        input_device = input_batch.device
        expert_input_batches = []
        no_dropped_samples = torch.zeros(
            0, dtype=input_batch.dtype, device=input_device
        )
        for expert_index in range(self.num_experts):
            expert_routing_positions, dropped_routing_positions = (
                self.__resolve_expert_routing_positions(
                    input_batch, indices, expert_index, input_device
                )
            )
            if self.__should_skip_expert_without_routed_samples(
                expert_routing_positions
            ):
                continue
            expert_samples = self.__select_expert_samples(
                input_batch, expert_routing_positions
            )
            expert_input = ExpertInputData(
                expert_index=expert_index,
                expert_samples=expert_samples,
                dropped_samples=no_dropped_samples,
                expert_routing_positions=expert_routing_positions,
                dropped_routing_positions=dropped_routing_positions,
                probabilities=None,
            )
            expert_input_batches.append(expert_input)
        return expert_input_batches

    def __resolve_expert_routing_positions(
        self,
        input_batch: Tensor,
        indices: Tensor | None,
        expert_index: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        if indices is None:
            return self.__get_dense_expert_routing_positions(
                input_batch, expert_index, device
            )
        return self.__get_expert_routing_positions(indices, expert_index)

    def __should_skip_expert_without_routed_samples(
        self,
        expert_routing_positions: Tensor | None,
    ) -> bool:
        return (
            expert_routing_positions is not None
            and expert_routing_positions.numel() == 0
        )

    def __get_dense_expert_routing_positions(
        self,
        input_batch: Tensor,
        expert_index: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        num_routed_samples = input_batch.size(0)
        expert_routing_positions = torch.arange(
            expert_index,
            num_routed_samples,
            self.num_experts,
            device=device,
        )
        dropped_routing_positions = torch.empty(0, dtype=torch.long, device=device)
        return expert_routing_positions, dropped_routing_positions

    def __select_expert_samples(
        self,
        input_batch: Tensor,
        expert_routing_positions: Tensor,
    ) -> Tensor:
        return input_batch[expert_routing_positions]

    def __dispatch_tokens_to_slots(
        self, inputs: Tensor, dispatch_weights: Tensor, skip_mask: Tensor | None
    ) -> Tensor:
        masked_token_features = self.__mask_inactive_token_features(inputs, skip_mask)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            dispatch_token_features = masked_token_features.to(dispatch_weights.dtype)
            input_slots = torch.einsum(
                "btes,btd->besd", dispatch_weights, dispatch_token_features
            )
        return input_slots.to(inputs.dtype)

    def __mask_inactive_token_features(
        self, token_features: Tensor, skip_mask: Tensor | None
    ) -> Tensor:
        if skip_mask is None:
            return token_features
        return token_features.masked_fill(~skip_mask, 0)
