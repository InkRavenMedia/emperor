"""Normalize each depth's broadcast sum before combining weights."""

from torch import Tensor

from emperor.augmentations.adaptive_parameters._weights.base import (
    DynamicWeightAbstract,
)
from emperor.augmentations.adaptive_parameters._weights.config import (
    BroadcastSumDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)


class BroadcastSumDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: BroadcastSumDynamicWeightConfig,
        overrides: BroadcastSumDynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.input_model = self.__init_input_model()
        self.output_model = self.__init_output_model()

    def __init_input_model(self) -> DepthMappingLayerStack:
        overrides = DepthMappingHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        return self._init_model(overrides)

    def __init_output_model(self) -> DepthMappingLayerStack:
        overrides = DepthMappingHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self._init_model(overrides)

    def _compute_broadcast_sum(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        broadcast_sum = input_vectors.unsqueeze(-1) + output_vectors.unsqueeze(-2)
        return self._apply_normalization_transform(broadcast_sum)

    def forward(
        self,
        weight_params: Tensor,
        X: Tensor,
    ) -> Tensor:
        input_vectors = self.input_model(X)
        output_vectors = self.output_model(X)
        broadcast_sum = self._compute_broadcast_sum(input_vectors, output_vectors)
        dynamic_params = self._compute_dynamic_weights(broadcast_sum)
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + dynamic_params
