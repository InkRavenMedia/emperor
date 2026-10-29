from dataclasses import replace

from torch import Tensor

from emperor.experts._config import MixtureOfExpertsConfig
from emperor.experts._layers.mixture import MixtureOfExperts
from emperor.experts._options import (
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)


class MixtureOfExpertsReduce(MixtureOfExperts):
    def __init__(
        self,
        cfg: MixtureOfExpertsConfig,
        overrides: MixtureOfExpertsConfig | None = None,
    ):
        self.VALIDATOR.validate_config_type(cfg)
        self.VALIDATOR.validate_overrides_type(overrides)
        overrides = self.__update_overrides(overrides)
        super().__init__(cfg, overrides)

    def __update_overrides(
        self, overrides: MixtureOfExpertsConfig | None = None
    ) -> MixtureOfExpertsConfig:
        if overrides is None:
            return MixtureOfExpertsConfig(
                weighted_parameters_flag=True,
                compute_expert_mixture_flag=True,
                weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
                routing_initialization_mode=RoutingInitializationMode.DISABLED,
            )
        return replace(
            overrides,
            weighted_parameters_flag=True,
            compute_expert_mixture_flag=True,
            weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
            routing_initialization_mode=RoutingInitializationMode.DISABLED,
        )

    def forward(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        self.VALIDATOR.validate_reduce_forward_inputs(
            self, input_batch, probabilities, indices, skip_mask
        )

        expert_outputs, routing_positions, probabilities, expert_loss = (
            self.expert_modules.forward_assignments(input_batch, probabilities, indices)
        )
        output = self.__compute_expert_mixture(
            expert_outputs, routing_positions, probabilities
        )
        return output, skip_mask, expert_loss

    def __compute_expert_mixture(
        self,
        experts_output: Tensor,
        routing_positions: Tensor | None,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        output_dim = experts_output.size(-1)
        if self.__should_restore_routing_order(routing_positions):
            _, routing_position_sort_order = routing_positions.sort(dim=0)
            experts_output = experts_output[routing_position_sort_order]

        experts_output = self.expert_weighting_handler.maybe_apply_probabilities_after(
            experts_output, probabilities
        )

        if self.__should_return_expert_outputs_without_reduction():
            return experts_output

        experts_output = experts_output.view(-1, self.top_k, output_dim)
        return experts_output.sum(dim=1)

    def __should_restore_routing_order(self, routing_positions: Tensor | None) -> bool:
        return self.top_k != self.num_experts and routing_positions is not None

    def __should_return_expert_outputs_without_reduction(self) -> bool:
        return not self.compute_expert_mixture_flag or self.top_k == 1
