from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.augmentations.adaptive_parameters._decay.options import (
    WeightDecayScheduleOptions,
)
from emperor.augmentations.adaptive_parameters._options import (
    BankExpansionFactorOptions,
    LowRankFactorSourceOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.config import ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.layers import LayerStackConfig
    from emperor.sampler import SamplerConfig


@dataclass
class WeightNormalizationConfig(ConfigBase):
    generator_depth: int | None = optional_field(
        "Required positive generator depth; scales are per depth before the depth sum."
    )
    normalization_option: WeightNormalizationOptions | None = optional_field(
        "Required enabled weight normalization method; DISABLED is invalid."
    )
    normalization_position_option: WeightNormalizationPositionOptions | None = (
        optional_field("Required normalization position, including explicit DISABLED.")
    )
    normalization_learnable_scale_flag: bool | None = optional_field(
        "Required boolean controlling normalization scales and clamp limits; "
        "must be False for AFTER_DEPTH_SUM."
    )

    def _registry_owner(self) -> type:
        from .normalization import WeightNormalizationPolicy

        return WeightNormalizationPolicy


@dataclass
class DynamicWeightConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    generator_depth: int | None = optional_field(
        "Positive integer generator depth for dynamic weights."
    )
    decay_schedule: WeightDecayScheduleOptions | None = optional_field(
        "Base weight decay schedule."
    )
    decay_rate: float | None = optional_field("Decay rate for the selected schedule.")
    decay_warmup_batches: int | None = optional_field(
        "Warmup batches before weight decay starts."
    )
    model_config: "LayerStackConfig | None" = optional_field(
        "Internal generator network config."
    )
    normalization_config: WeightNormalizationConfig | None = optional_field(
        "Optional normalization policy config; None disables weight normalization."
    )

    def _registry_owner(self) -> type:
        raise ValueError(
            "DynamicWeightConfig is abstract and has no registered "
            "DynamicWeight class; instantiate a concrete leaf config instead."
        )


@dataclass
class SingleModelDynamicWeightConfig(DynamicWeightConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._weights.variants.single_model import (
            SingleModelDynamicWeight,
        )

        return SingleModelDynamicWeight


@dataclass
class DualModelDynamicWeightConfig(DynamicWeightConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._weights.variants.dual_model import (
            DualModelDynamicWeight,
        )

        return DualModelDynamicWeight


@dataclass
class BroadcastSumDynamicWeightConfig(DynamicWeightConfig):
    def _registry_owner(self) -> type:
        from .variants.broadcast_sum import BroadcastSumDynamicWeight

        return BroadcastSumDynamicWeight


@dataclass
class LowRankDynamicWeightConfig(DynamicWeightConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._weights.variants.low_rank import (
            LowRankDynamicWeight,
        )

        return LowRankDynamicWeight


@dataclass
class DiagonallyModulatedLowRankDynamicWeightConfig(LowRankDynamicWeightConfig):
    input_factor_source: LowRankFactorSourceOptions | None = optional_field(
        "Source of U rank components. None selects GENERATED."
    )
    output_factor_source: LowRankFactorSourceOptions | None = optional_field(
        "Source of V rank components. None selects GENERATED."
    )
    input_factor_model_config: "LayerStackConfig | None" = optional_field(
        "Independent U generator; None inherits model_config. Invalid for shared U."
    )
    output_factor_model_config: "LayerStackConfig | None" = optional_field(
        "Independent V generator; None inherits model_config. Invalid for shared V."
    )
    coefficient_model_config: "LayerStackConfig | None" = optional_field(
        "Rank coefficient trunk; None inherits model_config."
    )

    def _registry_owner(self) -> type:
        from .variants.diagonally_modulated_low_rank import (
            DiagonallyModulatedLowRankDynamicWeight,
        )

        return DiagonallyModulatedLowRankDynamicWeight


@dataclass
class HypernetworkDynamicWeightConfig(DynamicWeightConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._weights.variants.hypernetwork import (
            HypernetworkDynamicWeight,
        )

        return HypernetworkDynamicWeight


@dataclass
class LayeredWeightedBankDynamicWeightConfig(DynamicWeightConfig):
    bank_expansion_factor: BankExpansionFactorOptions | None = optional_field(
        "Weight bank expansion factor."
    )

    def _registry_owner(self) -> type:
        from .variants.layered_weighted_bank import (
            LayeredWeightedBankDynamicWeight,
        )

        return LayeredWeightedBankDynamicWeight


@dataclass
class SoftWeightedBankDynamicWeightConfig(DynamicWeightConfig):
    bank_expansion_factor: BankExpansionFactorOptions | None = optional_field(
        "Weight bank expansion factor."
    )

    def _registry_owner(self) -> type:
        from .variants.soft_weighted_bank import (
            SoftWeightedBankDynamicWeight,
        )

        return SoftWeightedBankDynamicWeight


@dataclass
class MatrixWeightsMixtureConfig(DynamicWeightConfig):
    num_experts: int | None = optional_field(
        "Number of complete parameter bank entries."
    )
    top_k: int | None = optional_field("Number of selected entries per context.")
    sampler_config: "SamplerConfig | None" = optional_field(
        "Required sampler and router; routing controls and loss weights must be explicit."
    )

    def _registry_owner(self) -> type:
        from .variants.matrix_mixture import MatrixWeightsMixture

        return MatrixWeightsMixture
