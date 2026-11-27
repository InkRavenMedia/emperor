from copy import deepcopy
from dataclasses import replace

import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._weights.base import (
    DynamicWeightAbstract,
)
from emperor.augmentations.adaptive_parameters._weights.config import (
    MatrixWeightsMixtureConfig,
)
from emperor.augmentations.adaptive_parameters._weights.validation import (
    MatrixWeightsMixtureValidator,
)
from emperor.layers import LayerStack, LayerStackConfig
from emperor.nn import SoftSlotRouting


class MatrixWeightsMixture(DynamicWeightAbstract):
    VALIDATOR = MatrixWeightsMixtureValidator

    def __init__(
        self,
        cfg: MatrixWeightsMixtureConfig,
        overrides: MatrixWeightsMixtureConfig | None = None,
    ):
        config = deepcopy(self._override_config(cfg, overrides))
        super().__init__(config)
        self.num_experts = self.cfg.num_experts
        self.parameter_bank_shape = (self.num_experts, self.input_dim, self.output_dim)
        self.parameter_bank = self._init_parameter_bank(self.parameter_bank_shape)
        self.model = self.__init_router_model()
        self.routing = SoftSlotRouting()

    def __init_router_model(self) -> LayerStack:
        model_overrides = LayerStackConfig(
            input_dim=self.input_dim, output_dim=self.num_experts
        )
        return self.cfg.model_config.build(model_overrides)

    @classmethod
    def validate_owner_config(cls, cfg, *, input_dim, output_dim) -> None:
        for name, value in (("input_dim", input_dim), ("output_dim", output_dim)):
            cls.VALIDATOR.validate_matching_value(name, getattr(cfg, name), value)
        owner_config = replace(cfg, input_dim=input_dim, output_dim=output_dim)
        cls.VALIDATOR.validate_initialization_config(owner_config)

    def forward(
        self,
        weight_params: Tensor,
        context: Tensor,
    ) -> Tensor:
        self.VALIDATOR.validate_forward_inputs(self, weight_params, context)
        bank_logits = self.model.forward_hidden(context)
        dispatch_weights = self.routing.compute_dispatch_weights(
            bank_logits.unsqueeze(-1)
        )
        pooling_dtype = torch.promote_types(
            self.parameter_bank.dtype, dispatch_weights.dtype
        )
        bank_vectors = self.parameter_bank.flatten(1).to(pooling_dtype)
        pooled_bank_vectors = self.routing.pool_tokens_with_dispatch_weights(
            bank_vectors, dispatch_weights.to(pooling_dtype)
        )
        weight_update = pooled_bank_vectors.reshape(-1, self.input_dim, self.output_dim)
        update_dtype = torch.promote_types(self.parameter_bank.dtype, bank_logits.dtype)
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + weight_update.to(update_dtype)
