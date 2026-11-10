"""Weight normalization with optional learnable scale and clamp limit."""

from math import expm1, log

import torch
from torch import Tensor, nn

from emperor.augmentations.adaptive_parameters._options import (
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._weights.config import (
    WeightNormalizationConfig,
)
from emperor.augmentations.adaptive_parameters._weights.validation import (
    WeightNormalizationValidator,
)
from emperor.nn import Module


class WeightNormalizationPolicy(Module):
    VALIDATOR = WeightNormalizationValidator
    _MINIMUM_SCALE = 0.1

    def __init__(
        self,
        cfg: WeightNormalizationConfig,
        overrides: WeightNormalizationConfig | None = None,
    ):
        super().__init__()
        self.cfg: WeightNormalizationConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.generator_depth = self.cfg.generator_depth
        self.normalization_option = self.cfg.normalization_option
        self.normalization_position_option = self.cfg.normalization_position_option
        self.normalization_learnable_scale_flag = (
            self.cfg.normalization_learnable_scale_flag
        )
        self.register_parameter("raw_scale", self.__initialize_raw_scale())
        self.clamp_limit = nn.Parameter(
            torch.tensor(1.0), requires_grad=self.normalization_learnable_scale_flag
        )

    def __initialize_raw_scale(self) -> nn.Parameter | None:
        if (
            self.normalization_position_option
            is WeightNormalizationPositionOptions.AFTER_DEPTH_SUM
        ):
            return None
        initial_effective_scale = 1.0
        initial_softplus_output = initial_effective_scale - self._MINIMUM_SCALE
        exponentiated_initial_raw_scale = expm1(initial_softplus_output)
        initial_raw_scale_value = log(exponentiated_initial_raw_scale)
        initial_raw_scale_shape = (self.generator_depth,)
        initial_raw_scale_tensor = torch.full(
            initial_raw_scale_shape, initial_raw_scale_value
        )
        return nn.Parameter(
            initial_raw_scale_tensor,
            requires_grad=self.normalization_learnable_scale_flag,
        )

    @property
    def scale(self) -> Tensor | None:
        if self.raw_scale is None:
            return None
        return self.__calculate_scale()

    def normalize_before_outer_product(self, vectors: Tensor) -> Tensor:
        self.VALIDATOR.validate_normalization_position_option(
            self.normalization_position_option
        )
        if (
            self.normalization_position_option
            is WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
        ):
            return self(vectors)
        return vectors

    def normalize_after_outer_product(self, outer_product: Tensor) -> Tensor:
        self.VALIDATOR.validate_normalization_position_option(
            self.normalization_position_option
        )
        if (
            self.normalization_position_option
            is WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT
        ):
            return self(outer_product)
        return outer_product

    def normalize_after_depth_sum(self, summed_weights: Tensor) -> Tensor:
        self.VALIDATOR.validate_normalization_position_option(
            self.normalization_position_option
        )
        if (
            self.normalization_position_option
            is WeightNormalizationPositionOptions.AFTER_DEPTH_SUM
        ):
            return self(summed_weights)
        return summed_weights

    def forward(self, vectors: Tensor) -> Tensor:
        self.VALIDATOR.validate_normalization_option(self.normalization_option)
        self.VALIDATOR.validate_normalization_position_option(
            self.normalization_position_option
        )
        match self.normalization_option:
            case WeightNormalizationOptions.CLAMP:
                return self.__apply_symmetric_clamp(vectors)
            case WeightNormalizationOptions.L2_SCALE:
                normalized_vectors = self.__apply_stable_l2_normalization(vectors)
            case WeightNormalizationOptions.SOFT_CLAMP:
                return self.__apply_stable_soft_clamp(vectors)
            case WeightNormalizationOptions.RMS:
                normalized_vectors = self.__apply_stable_rms_normalization(vectors)
            case WeightNormalizationOptions.SIGMOID_SCALE:
                normalized_vectors = torch.sigmoid(vectors) * 2 - 1
            case WeightNormalizationOptions.SOFTSIGN:
                normalized_vectors = torch.nn.functional.softsign(vectors)
            case WeightNormalizationOptions.DISABLED:
                return vectors
        if (
            self.normalization_position_option
            is WeightNormalizationPositionOptions.AFTER_DEPTH_SUM
        ):
            return normalized_vectors
        return normalized_vectors * self.__scale_for_depths(vectors)

    def __calculate_scale(self) -> Tensor:
        return self._MINIMUM_SCALE + torch.nn.functional.softplus(self.raw_scale)

    def __scale_for_depths(self, vectors: Tensor) -> Tensor:
        depth_axis = 0 if vectors.ndim == 2 else 1
        self.VALIDATOR.validate_depth_axis(vectors, self.generator_depth, depth_axis)
        broadcast_shape = [1] * vectors.ndim
        broadcast_shape[depth_axis] = self.generator_depth
        effective_scales = self.__calculate_scale().to(dtype=vectors.dtype)
        return effective_scales.reshape(broadcast_shape)

    def __apply_symmetric_clamp(self, vectors: Tensor) -> Tensor:
        clamp_limit_magnitude = self.clamp_limit.abs()
        return torch.clamp(
            vectors,
            -clamp_limit_magnitude,
            clamp_limit_magnitude,
        )

    def __apply_stable_l2_normalization(self, vectors: Tensor) -> Tensor:
        (
            accumulator_vectors,
            maximum_magnitude,
            magnitude_scaled_vectors,
            contains_nonzero_value,
        ) = self.__scale_vectors_by_maximum_magnitude(vectors)
        squared_magnitude_scaled_vectors = magnitude_scaled_vectors.square()
        scaled_squared_l2_norm = squared_magnitude_scaled_vectors.sum(
            dim=-1,
            keepdim=True,
        )
        stable_scaled_squared_l2_norm = torch.where(
            contains_nonzero_value,
            scaled_squared_l2_norm,
            torch.ones_like(scaled_squared_l2_norm),
        )
        scaled_l2_norm = stable_scaled_squared_l2_norm.sqrt()
        l2_norm = maximum_magnitude * scaled_l2_norm

        finite_l2_norm = torch.isfinite(l2_norm)
        safe_l2_norm = torch.where(
            finite_l2_norm,
            l2_norm,
            torch.ones_like(l2_norm),
        )
        minimum_l2_norm = max(1e-12, torch.finfo(vectors.dtype).tiny)
        normalized_by_l2_norm = accumulator_vectors / safe_l2_norm.clamp_min(
            minimum_l2_norm
        )
        normalized_by_scaled_l2_norm = magnitude_scaled_vectors / scaled_l2_norm
        normalized_vectors = torch.where(
            finite_l2_norm, normalized_by_l2_norm, normalized_by_scaled_l2_norm
        )
        return normalized_vectors.to(dtype=vectors.dtype)

    @staticmethod
    def __scale_vectors_by_maximum_magnitude(
        vectors: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if vectors.dtype in (torch.float16, torch.bfloat16):
            accumulator_vectors = vectors.float()
        else:
            accumulator_vectors = vectors
        maximum_magnitude = accumulator_vectors.abs().amax(
            dim=-1,
            keepdim=True,
        )
        contains_nonzero_value = maximum_magnitude > 0
        safe_maximum_magnitude = torch.where(
            contains_nonzero_value,
            maximum_magnitude,
            torch.ones_like(maximum_magnitude),
        )
        magnitude_scaled_vectors = accumulator_vectors / safe_maximum_magnitude
        return (
            accumulator_vectors,
            maximum_magnitude,
            magnitude_scaled_vectors,
            contains_nonzero_value,
        )

    def __apply_stable_soft_clamp(self, vectors: Tensor) -> Tensor:
        clamp_limit_magnitude = self.clamp_limit.abs()
        minimum_safe_denominator = torch.finfo(vectors.dtype).eps
        safe_clamp_denominator = clamp_limit_magnitude.clamp_min(
            minimum_safe_denominator
        )
        scaled_vectors = vectors / safe_clamp_denominator
        return clamp_limit_magnitude * torch.tanh(scaled_vectors)

    def __apply_stable_rms_normalization(self, vectors: Tensor) -> Tensor:
        (
            accumulator_vectors,
            maximum_magnitude,
            magnitude_scaled_vectors,
            contains_nonzero_value,
        ) = self.__scale_vectors_by_maximum_magnitude(vectors)
        scaled_squared_mean = magnitude_scaled_vectors.square().mean(
            dim=-1,
            keepdim=True,
        )
        stable_scaled_squared_mean = torch.where(
            contains_nonzero_value,
            scaled_squared_mean,
            torch.ones_like(scaled_squared_mean),
        )
        scaled_root_mean_square = stable_scaled_squared_mean.sqrt()
        root_mean_square = maximum_magnitude * scaled_root_mean_square
        minimum_root_mean_square = max(1e-8, torch.finfo(vectors.dtype).tiny)
        normalized_vectors = accumulator_vectors / (
            root_mean_square + minimum_root_mean_square
        )
        return normalized_vectors.to(dtype=vectors.dtype)
