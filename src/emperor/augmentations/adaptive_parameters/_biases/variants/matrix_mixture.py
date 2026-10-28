from copy import deepcopy
from dataclasses import replace

from torch import Tensor

from emperor.augmentations.adaptive_parameters._biases.base import DynamicBiasAbstract
from emperor.augmentations.adaptive_parameters._biases.config import (
    MatrixBiasMixtureConfig,
)
from emperor.augmentations.adaptive_parameters._biases.validation import (
    MatrixBiasMixtureValidator,
)


class MatrixBiasMixture(DynamicBiasAbstract):
    VALIDATOR = MatrixBiasMixtureValidator

    def __init__(
        self,
        cfg: MatrixBiasMixtureConfig,
        overrides: MatrixBiasMixtureConfig | None = None,
    ):
        config = deepcopy(self._override_config(cfg, overrides))
        super().__init__(config)
        self.num_experts = self.cfg.num_experts
        self.top_k = self.cfg.top_k
        self.parameter_bank_shape = (self.num_experts, self.output_dim)
        self.parameter_bank = self._init_parameter_bank(self.parameter_bank_shape)
        self.sampler = self.__init_sampler()

    def __init_sampler(self):
        self.VALIDATOR.validate_sampler_config(self.cfg.sampler_config)
        return self.cfg.sampler_config.build()

    def get_auxiliary_loss(self) -> Tensor:
        return self.sampler.get_auxiliary_loss()

    @classmethod
    def validate_owner_config(cls, cfg, *, input_dim, output_dim) -> None:
        """Preflight the effective configuration without constructing parameters."""
        for name, value in (("input_dim", input_dim), ("output_dim", output_dim)):
            cls.VALIDATOR.validate_matching_value(name, getattr(cfg, name), value)
        owner_config = replace(cfg, input_dim=input_dim, output_dim=output_dim)
        cls.VALIDATOR.validate_initialization_config(owner_config)
        cls.VALIDATOR.validate_routing_config(owner_config)

    def forward(
        self,
        bias_params: Tensor,
        context: Tensor,
    ) -> Tensor:
        self.VALIDATOR.validate_forward_inputs(self, bias_params, context)
        self.VALIDATOR.validate_sampler_available(self.sampler)
        skip_mask = None
        if (
            self.cfg.sampler_config.threshold != 0.0
            or self.cfg.sampler_config.mutual_information_loss_weight != 0.0
        ):
            skip_mask = context.new_ones((context.shape[0], 1))
        probabilities, indices, skip_mask, _ = (
            self.sampler.sample_probabilities_and_indices(context, skip_mask)
        )
        if skip_mask is not None:
            mask_shape = (-1,) + (1,) * (probabilities.ndim - 1)
            probabilities = probabilities * skip_mask.reshape(mask_shape)
        self.VALIDATOR.validate_route(self, probabilities, indices)
        generated_parameters = self.__reduce_mixture(probabilities, indices)
        decayed_parameters = self._maybe_apply_bias_decay(bias_params)
        return decayed_parameters + generated_parameters

    def __reduce_mixture(self, probabilities: Tensor, indices: Tensor | None) -> Tensor:
        selected_parameters = self.__select_parameters(indices)
        weighted_parameters = self.__apply_probabilities(
            selected_parameters, probabilities
        )
        return weighted_parameters.sum(dim=1)

    def __select_parameters(self, indices: Tensor | None) -> Tensor:
        if indices is None:
            return self.parameter_bank
        expert_indices_per_context = indices.reshape(-1, self.top_k)
        return self.parameter_bank[expert_indices_per_context]

    def __apply_probabilities(
        self, selected_parameters: Tensor, probabilities: Tensor
    ) -> Tensor:
        probability_shape = (-1, self.top_k) + (1,) * (self.parameter_bank.ndim - 1)
        return selected_parameters * probabilities.reshape(probability_shape)
