from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.experts._config import MixtureOfExpertsConfig
from emperor.experts._layers.experts import Experts
from emperor.experts._options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.experts._routing.capacity import ExpertCapacityHandler
from emperor.experts._routing.weighting import ExpertWeightingHandler
from emperor.experts._validation.mixture import MixtureOfExpertsValidator
from emperor.layers import LayerStackConfig, RecurrentCompositionConfig
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.sampler import SamplerConfig, SamplerModel


class MixtureOfExperts(Module):
    VALIDATOR = MixtureOfExpertsValidator

    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig",
        overrides: "MixtureOfExpertsConfig | None" = None,
    ):
        self.VALIDATOR.validate_config_type(cfg)
        self.VALIDATOR.validate_overrides_type(overrides)
        super().__init__()
        self.cfg: MixtureOfExpertsConfig = self._override_config(cfg, overrides)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.expert_model_config: LayerStackConfig | RecurrentCompositionConfig = (
            self.cfg.expert_model_config
        )
        self.top_k: int = self.cfg.top_k
        self.num_experts: int = self.cfg.num_experts
        self.capacity_factor: float = self.cfg.capacity_factor
        self.dropped_token_behavior: DroppedTokenOptions = (
            self.cfg.dropped_token_behavior or DroppedTokenOptions.ZEROS
        )
        self.compute_expert_mixture_flag: bool = self.cfg.compute_expert_mixture_flag
        self.weighted_parameters_flag: bool = self.cfg.weighted_parameters_flag
        self.routing_initialization_mode: RoutingInitializationMode = (
            self.cfg.routing_initialization_mode
        )
        self.weighting_position_option: ExpertWeightingPositionOptions = (
            self.cfg.weighting_position_option
        )
        self.sampler_config: SamplerConfig = self.cfg.sampler_config

        self.VALIDATOR.validate(self)
        self.sampler = self.__maybe_create_sampler()
        self.expert_modules: Experts = Experts(self.cfg)
        self.capacity_handler: ExpertCapacityHandler = (
            self.expert_modules.capacity_handler
        )
        self.expert_weighting_handler: ExpertWeightingHandler = (
            self.expert_modules.expert_weighting_handler
        )

    def __maybe_create_sampler(
        self,
    ) -> "SamplerModel | None":
        if self.routing_initialization_mode != RoutingInitializationMode.LAYER:
            return None
        self.VALIDATOR.validate_sampler_config_exists(self)
        self.VALIDATOR.validate_router_config_exists(self)
        return self.sampler_config.build_with_router_input_dim(self.input_dim)


    def get_top_k(self) -> int:
        return self.top_k

    def forward(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        self.VALIDATOR.validate_forward_inputs(
            self, input_batch, probabilities, indices, skip_mask
        )
        probabilities, indices, skip_mask, sampler_loss = (
            self._maybe_compute_expert_indices(
                input_batch, probabilities, indices, skip_mask
            )
        )
        self.VALIDATOR.validate_grouping_skip_mask(self, skip_mask)
        expert_outputs, routing_positions, probabilities, expert_loss = (
            self.expert_modules(input_batch, probabilities, indices)
        )
        mixture_output = self.__compute_expert_mixture(
            expert_outputs, routing_positions, probabilities
        )
        total_loss = sampler_loss + expert_loss
        return mixture_output, skip_mask, total_loss



    def _maybe_compute_expert_indices(
        self,
        inputs: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor]:
        if self.__should_skip_layer_routing(probabilities, indices):
            return probabilities, indices, skip_mask, inputs.new_zeros(())
        self.VALIDATOR.validate_sampler_is_initialized(self)
        self.VALIDATOR.validate_external_probabilities_are_not_given(
            probabilities, indices
        )
        probabilities, indices, skip_mask, sampler_loss = (
            self.sampler.sample_probabilities_and_indices(inputs, skip_mask)
        )
        return probabilities, indices, skip_mask, sampler_loss

    def __should_skip_layer_routing(
        self,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> bool:
        layer_routing_is_enabled = (
            self.routing_initialization_mode == RoutingInitializationMode.LAYER
        )
        external_routing_is_provided = indices is not None or probabilities is not None
        return not layer_routing_is_enabled or external_routing_is_provided

    def __compute_expert_mixture(
        self,
        expert_outputs: Tensor,
        routing_positions: Tensor | None,
        routing_probabilities: Tensor | None = None,
    ) -> Tensor:
        output_dim = expert_outputs.size(-1)
        if self.__should_restore_routing_order(routing_positions):
            _, routing_position_sort_order = routing_positions.sort(dim=0)
            expert_outputs = expert_outputs[routing_position_sort_order]

        expert_outputs = self.expert_weighting_handler.maybe_apply_probabilities_after(
            expert_outputs, routing_probabilities
        )

        if self.__should_return_expert_outputs_without_reduction():
            return expert_outputs

        expert_outputs_grouped_by_sample = expert_outputs.view(
            -1, self.top_k, output_dim
        )
        combined_expert_output_per_sample = expert_outputs_grouped_by_sample.sum(dim=1)
        return combined_expert_output_per_sample

    def __should_restore_routing_order(
        self,
        routing_positions: Tensor | None,
    ) -> bool:
        return self.top_k != self.num_experts and routing_positions is not None

    def __should_return_expert_outputs_without_reduction(self) -> bool:
        return not self.compute_expert_mixture_flag or self.top_k == 1
