from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Protocol

from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from model_runtime.packages._runtime_default_fields import read_optional_stack

from ._control_options import StackOptions, inherit_if_unset


class AdaptiveSettings(Protocol):
    adaptive_generator_stack_activation: Final[ActivationOptions]
    adaptive_generator_stack_apply_output_postprocessing_flag: Final[bool]
    adaptive_generator_stack_bias_flag: Final[bool]
    adaptive_generator_stack_dropout_probability: Final[float]
    adaptive_generator_stack_hidden_dim: Final[int]
    adaptive_generator_stack_last_layer_bias_option: Final[LastLayerBiasOptions]
    adaptive_generator_stack_layer_norm_position: Final[LayerNormPositionOptions]
    adaptive_generator_stack_normalization: Final[NormalizationOptions]
    adaptive_generator_stack_num_layers: Final[int]
    adaptive_generator_stack_residual_block_size: Final[int | None]
    adaptive_generator_stack_residual_connection_option: Final[
        type[ResidualConfig] | None
    ]
    adaptive_generator_stack_residual_model_flag: Final[bool]
    adaptive_generator_stack_residual_rms_norm_epsilon: Final[float | None]
    bias_bank_expansion_factor: Final[BankExpansionFactorOptions]
    bias_decay_rate: Final[float]
    bias_decay_schedule: Final[WeightDecayScheduleOptions]
    bias_decay_warmup_batches: Final[int]
    bias_option: Final[type[DynamicBiasConfig] | None]
    bias_option_flag: Final[bool]
    diagonal_option: Final[type[DynamicDiagonalConfig] | None]
    diagonal_option_flag: Final[bool]
    generator_depth: Final[int]
    mask_dimension_option: Final[MaskDimensionOptions]
    mask_floor: Final[float]
    mask_option_flag: Final[bool]
    mask_surrogate_scale: Final[float]
    mask_threshold: Final[float]
    mask_transition_width: Final[float]
    row_mask_option: Final[type[AxisMaskConfig] | None]
    weight_bank_expansion_factor: Final[BankExpansionFactorOptions]
    weight_decay_rate: Final[float]
    weight_decay_schedule: Final[WeightDecayScheduleOptions]
    weight_decay_warmup_batches: Final[int]
    weight_normalization_learnable_scale_flag: Final[bool]
    weight_normalization_option: Final[WeightNormalizationOptions]
    weight_normalization_position_option: Final[WeightNormalizationPositionOptions]
    weight_option: Final[type[DynamicWeightConfig] | None]
    weight_option_flag: Final[bool]


@dataclass(frozen=True, slots=True)
class GeneratorStackSource:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    activation: ActivationOptions | None
    dropout_probability: float | None
    layer_norm_position: LayerNormPositionOptions | None
    normalization: NormalizationOptions | None = field(default=None, kw_only=True)
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    bias_flag: bool | None

    def resolve(self, defaults: StackOptions) -> StackOptions | None:
        if not self.independent_flag:
            return None
        return StackOptions(
            hidden_dim=inherit_if_unset(self.hidden_dim, defaults.hidden_dim),
            num_layers=inherit_if_unset(self.num_layers, defaults.num_layers),
            activation=inherit_if_unset(self.activation, defaults.activation),
            dropout_probability=inherit_if_unset(
                self.dropout_probability, defaults.dropout_probability
            ),
            layer_norm_position=inherit_if_unset(
                self.layer_norm_position, defaults.layer_norm_position
            ),
            normalization=inherit_if_unset(self.normalization, defaults.normalization),
            residual_connection_option=inherit_if_unset(
                self.residual_connection_option, defaults.residual_connection_option
            ),
            residual_block_size=self.residual_block_size,
            residual_rms_norm_epsilon=self.residual_rms_norm_epsilon,
            residual_model_flag=self.residual_model_flag,
            last_layer_bias_option=inherit_if_unset(
                self.last_layer_bias_option, defaults.last_layer_bias_option
            ),
            apply_output_postprocessing_flag=inherit_if_unset(
                self.apply_output_postprocessing_flag,
                defaults.apply_output_postprocessing_flag,
            ),
            bias_flag=inherit_if_unset(self.bias_flag, defaults.bias_flag),
        )


@dataclass(frozen=True, slots=True)
class WeightOptions:
    enabled: bool
    implementation: type[DynamicWeightConfig] | None
    generator_depth: int
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    normalization_option: WeightNormalizationOptions
    normalization_learnable_scale_flag: bool
    normalization_position_option: WeightNormalizationPositionOptions
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack: GeneratorStackSource


@dataclass(frozen=True, slots=True)
class BiasOptions:
    enabled: bool
    implementation: type[DynamicBiasConfig] | None
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack: GeneratorStackSource


@dataclass(frozen=True, slots=True)
class DiagonalOptions:
    enabled: bool
    implementation: type[DynamicDiagonalConfig] | None
    generator_stack: GeneratorStackSource


@dataclass(frozen=True, slots=True)
class MaskOptions:
    enabled: bool
    implementation: type[AxisMaskConfig] | None
    threshold: float
    surrogate_scale: float
    floor: float
    dimension_option: MaskDimensionOptions
    transition_width: float
    generator_stack: GeneratorStackSource


@dataclass(frozen=True, slots=True)
class AdaptiveOptions:
    generator_stack: StackOptions
    weight: WeightOptions
    bias: BiasOptions
    diagonal: DiagonalOptions
    mask: MaskOptions


def adaptive_options(runtime: AdaptiveSettings) -> AdaptiveOptions:
    return AdaptiveOptions(
        generator_stack=StackOptions(
            hidden_dim=runtime.adaptive_generator_stack_hidden_dim,
            num_layers=runtime.adaptive_generator_stack_num_layers,
            activation=runtime.adaptive_generator_stack_activation,
            dropout_probability=runtime.adaptive_generator_stack_dropout_probability,
            layer_norm_position=runtime.adaptive_generator_stack_layer_norm_position,
            normalization=runtime.adaptive_generator_stack_normalization,
            residual_connection_option=(
                runtime.adaptive_generator_stack_residual_connection_option
            ),
            residual_block_size=runtime.adaptive_generator_stack_residual_block_size,
            residual_rms_norm_epsilon=runtime.adaptive_generator_stack_residual_rms_norm_epsilon,
            residual_model_flag=(runtime.adaptive_generator_stack_residual_model_flag),
            last_layer_bias_option=(
                runtime.adaptive_generator_stack_last_layer_bias_option
            ),
            apply_output_postprocessing_flag=(
                runtime.adaptive_generator_stack_apply_output_postprocessing_flag
            ),
            bias_flag=runtime.adaptive_generator_stack_bias_flag,
        ),
        weight=WeightOptions(
            enabled=runtime.weight_option_flag,
            implementation=runtime.weight_option,
            generator_depth=runtime.generator_depth,
            decay_schedule=runtime.weight_decay_schedule,
            decay_rate=runtime.weight_decay_rate,
            decay_warmup_batches=runtime.weight_decay_warmup_batches,
            normalization_option=runtime.weight_normalization_option,
            normalization_learnable_scale_flag=runtime.weight_normalization_learnable_scale_flag,
            normalization_position_option=(
                runtime.weight_normalization_position_option
            ),
            bank_expansion_factor=runtime.weight_bank_expansion_factor,
            generator_stack=read_optional_stack(
                GeneratorStackSource, runtime, "weight_generator_stack"
            ),
        ),
        bias=BiasOptions(
            enabled=runtime.bias_option_flag,
            implementation=runtime.bias_option,
            decay_schedule=runtime.bias_decay_schedule,
            decay_rate=runtime.bias_decay_rate,
            decay_warmup_batches=runtime.bias_decay_warmup_batches,
            bank_expansion_factor=runtime.bias_bank_expansion_factor,
            generator_stack=read_optional_stack(
                GeneratorStackSource, runtime, "bias_generator_stack"
            ),
        ),
        diagonal=DiagonalOptions(
            enabled=runtime.diagonal_option_flag,
            implementation=runtime.diagonal_option,
            generator_stack=read_optional_stack(
                GeneratorStackSource, runtime, "diagonal_generator_stack"
            ),
        ),
        mask=MaskOptions(
            enabled=runtime.mask_option_flag,
            implementation=runtime.row_mask_option,
            threshold=runtime.mask_threshold,
            surrogate_scale=runtime.mask_surrogate_scale,
            floor=runtime.mask_floor,
            dimension_option=runtime.mask_dimension_option,
            transition_width=runtime.mask_transition_width,
            generator_stack=read_optional_stack(
                GeneratorStackSource, runtime, "mask_generator_stack"
            ),
        ),
    )


def weight_mixture_router_generator_stack_source(
    runtime: AdaptiveSettings,
) -> GeneratorStackSource:
    return read_optional_stack(
        GeneratorStackSource, runtime, "weight_mixture_router_generator_stack"
    )


def bias_mixture_router_generator_stack_source(
    runtime: AdaptiveSettings,
) -> GeneratorStackSource:
    return read_optional_stack(
        GeneratorStackSource, runtime, "bias_mixture_router_generator_stack"
    )


def weight_input_factor_generator_stack_source(
    runtime: AdaptiveSettings,
) -> GeneratorStackSource:
    return read_optional_stack(
        GeneratorStackSource, runtime, "weight_input_factor_generator_stack"
    )


def weight_output_factor_generator_stack_source(
    runtime: AdaptiveSettings,
) -> GeneratorStackSource:
    return read_optional_stack(
        GeneratorStackSource, runtime, "weight_output_factor_generator_stack"
    )


def weight_coefficient_generator_stack_source(
    runtime: AdaptiveSettings,
) -> GeneratorStackSource:
    return read_optional_stack(
        GeneratorStackSource, runtime, "weight_coefficient_generator_stack"
    )
