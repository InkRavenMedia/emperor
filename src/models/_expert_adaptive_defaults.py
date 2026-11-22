from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Final, Generic, Protocol, TypeVar, cast

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

_Result = TypeVar("_Result")


class _GeneratorSettings(Protocol):
    hidden_dim: Final[int]
    layer_norm_position: Final[LayerNormPositionOptions]
    normalization: Final[NormalizationOptions]
    num_layers: Final[int]
    activation: Final[ActivationOptions]
    residual_connection_option: Final[type[ResidualConfig] | None]
    residual_block_size: Final[int | None]
    residual_rms_norm_epsilon: Final[float | None]
    residual_model_flag: Final[bool]
    dropout_probability: Final[float]
    last_layer_bias_option: Final[LastLayerBiasOptions]
    apply_output_postprocessing_flag: Final[bool]
    residual_stack_options: Final[object]


class AdaptiveSettings(Protocol):
    weight_option_flag: Final[bool | None]
    generator_depth: Final[int]
    diagonal_option: Final[type[DynamicDiagonalConfig] | None]
    diagonal_option_flag: Final[bool | None]
    bias_option: Final[type[DynamicBiasConfig] | None]
    bias_option_flag: Final[bool | None]
    weight_option: Final[type[DynamicWeightConfig] | None]
    weight_normalization_option: Final[WeightNormalizationOptions]
    weight_normalization_learnable_scale_flag: Final[bool]
    weight_normalization_position_option: Final[WeightNormalizationPositionOptions]
    weight_decay_schedule: Final[WeightDecayScheduleOptions]
    weight_decay_rate: Final[float]
    weight_decay_warmup_batches: Final[int]
    weight_bank_expansion_factor: Final[BankExpansionFactorOptions]
    bias_decay_schedule: Final[WeightDecayScheduleOptions]
    bias_decay_rate: Final[float]
    bias_decay_warmup_batches: Final[int]
    bias_bank_expansion_factor: Final[BankExpansionFactorOptions]
    row_mask_option: Final[type[AxisMaskConfig] | None]
    mask_option_flag: Final[bool | None]
    mask_dimension_option: Final[MaskDimensionOptions]
    mask_threshold: Final[float]
    mask_surrogate_scale: Final[float]
    mask_floor: Final[float]
    mask_transition_width: Final[float]
    adaptive_generator_stack_num_layers: Final[int]
    adaptive_generator_stack_hidden_dim: Final[int]
    adaptive_generator_stack_activation: Final[ActivationOptions]
    adaptive_generator_stack_residual_connection_option: Final[
        type[ResidualConfig] | None
    ]
    adaptive_generator_stack_residual_model_flag: Final[bool]
    adaptive_generator_stack_residual_block_size: Final[int | None]
    adaptive_generator_stack_residual_rms_norm_epsilon: Final[float | None]
    adaptive_generator_stack_dropout_probability: Final[float]
    adaptive_generator_stack_layer_norm_position: Final[LayerNormPositionOptions]
    adaptive_generator_stack_normalization: Final[NormalizationOptions]
    adaptive_generator_stack_last_layer_bias_option: Final[LastLayerBiasOptions]
    adaptive_generator_stack_apply_output_postprocessing_flag: Final[bool]
    adaptive_generator_stack_bias_flag: Final[bool]
    input_layer_weight_option: Final[type[DynamicWeightConfig] | None]
    input_layer_generator_depth: Final[int]
    input_layer_weight_decay_schedule: Final[WeightDecayScheduleOptions]
    input_layer_weight_decay_rate: Final[float]
    input_layer_weight_decay_warmup_batches: Final[int]
    input_layer_weight_normalization_option: Final[WeightNormalizationOptions]
    input_layer_weight_normalization_learnable_scale_flag: Final[bool]
    input_layer_weight_normalization_position_option: Final[
        WeightNormalizationPositionOptions
    ]
    input_layer_weight_bank_expansion_factor: Final[BankExpansionFactorOptions]
    input_layer_bias_option: Final[type[DynamicBiasConfig] | None]
    input_layer_bias_decay_schedule: Final[WeightDecayScheduleOptions]
    input_layer_bias_decay_rate: Final[float]
    input_layer_bias_decay_warmup_batches: Final[int]
    input_layer_bias_bank_expansion_factor: Final[BankExpansionFactorOptions]
    input_layer_diagonal_option: Final[type[DynamicDiagonalConfig] | None]
    input_layer_row_mask_option: Final[type[AxisMaskConfig] | None]
    input_layer_mask_dimension_option: Final[MaskDimensionOptions]
    input_layer_mask_threshold: Final[float]
    input_layer_mask_surrogate_scale: Final[float]
    input_layer_mask_floor: Final[float]
    input_layer_mask_transition_width: Final[float]
    output_layer_weight_option: Final[type[DynamicWeightConfig] | None]
    output_layer_generator_depth: Final[int]
    output_layer_weight_decay_schedule: Final[WeightDecayScheduleOptions]
    output_layer_weight_decay_rate: Final[float]
    output_layer_weight_decay_warmup_batches: Final[int]
    output_layer_weight_normalization_option: Final[WeightNormalizationOptions]
    output_layer_weight_normalization_learnable_scale_flag: Final[bool]
    output_layer_weight_normalization_position_option: Final[
        WeightNormalizationPositionOptions
    ]
    output_layer_weight_bank_expansion_factor: Final[BankExpansionFactorOptions]
    output_layer_bias_option: Final[type[DynamicBiasConfig] | None]
    output_layer_bias_decay_schedule: Final[WeightDecayScheduleOptions]
    output_layer_bias_decay_rate: Final[float]
    output_layer_bias_decay_warmup_batches: Final[int]
    output_layer_bias_bank_expansion_factor: Final[BankExpansionFactorOptions]
    output_layer_diagonal_option: Final[type[DynamicDiagonalConfig] | None]
    output_layer_row_mask_option: Final[type[AxisMaskConfig] | None]
    output_layer_mask_dimension_option: Final[MaskDimensionOptions]
    output_layer_mask_threshold: Final[float]
    output_layer_mask_surrogate_scale: Final[float]
    output_layer_mask_floor: Final[float]
    output_layer_mask_transition_width: Final[float]
    router_weight_option_flag: Final[bool | None]
    router_weight_option: Final[type[DynamicWeightConfig] | None]
    router_generator_depth: Final[int]
    router_weight_normalization_option: Final[WeightNormalizationOptions]
    router_weight_normalization_learnable_scale_flag: Final[bool]
    router_weight_normalization_position_option: Final[
        WeightNormalizationPositionOptions
    ]
    router_weight_decay_schedule: Final[WeightDecayScheduleOptions]
    router_weight_decay_rate: Final[float]
    router_weight_decay_warmup_batches: Final[int]
    router_weight_bank_expansion_factor: Final[BankExpansionFactorOptions]
    router_bias_option_flag: Final[bool | None]
    router_bias_option: Final[type[DynamicBiasConfig] | None]
    router_bias_decay_schedule: Final[WeightDecayScheduleOptions]
    router_bias_decay_rate: Final[float]
    router_bias_decay_warmup_batches: Final[int]
    router_bias_bank_expansion_factor: Final[BankExpansionFactorOptions]
    router_diagonal_option_flag: Final[bool | None]
    router_diagonal_option: Final[type[DynamicDiagonalConfig] | None]
    router_mask_option_flag: Final[bool | None]
    router_row_mask_option: Final[type[AxisMaskConfig] | None]
    router_mask_dimension_option: Final[MaskDimensionOptions]
    router_mask_threshold: Final[float]
    router_mask_surrogate_scale: Final[float]
    router_mask_floor: Final[float]
    router_mask_transition_width: Final[float]
    adaptive_generator_stack_options: Final[_GeneratorSettings | None]
    hidden_adaptive_weight_options: Final[object]
    hidden_adaptive_bias_options: Final[object]
    hidden_adaptive_diagonal_options: Final[object]
    hidden_adaptive_mask_options: Final[object]
    input_boundary_options: Final[object]
    output_boundary_options: Final[object]
    router_adaptive_weight_options: Final[object]
    router_adaptive_bias_options: Final[object]
    router_adaptive_diagonal_options: Final[object]
    router_adaptive_mask_options: Final[object]


@dataclass(frozen=True, slots=True)
class _HiddenAdaptiveDefaults:
    weight: object
    bias: object
    diagonal: object
    mask: object


@dataclass(frozen=True, slots=True)
class _RouterAdaptiveDefaults:
    weight: object
    bias: object
    diagonal: object
    mask: object


@dataclass(frozen=True, slots=True)
class _BoundaryDefaults:
    input: object
    output: object


class ExpertAdaptiveDefaults(Generic[_Result]):
    generator_type: type[_GeneratorSettings]
    source_type: Callable[..., object]
    weight_type: Callable[..., object]
    bias_type: Callable[..., object]
    diagonal_type: Callable[..., object]
    mask_type: Callable[..., object]
    boundary_type: Callable[..., object]
    defaults_type: Callable[..., _Result]
    config: ModuleType
    inherit_foreign_generator_residual = False

    def _adaptive_option_flag(
        self, explicit_flag: bool | None, option: type | None, default_flag: bool
    ) -> bool:
        if explicit_flag is not None:
            return explicit_flag
        if option is not None:
            return True
        return default_flag

    def _adaptive_generator_defaults(self, values: AdaptiveSettings) -> object:
        provided = values.adaptive_generator_stack_options
        create_generator = cast(Callable[..., _GeneratorSettings], self.generator_type)
        if provided is None:
            return create_generator(
                hidden_dim=values.adaptive_generator_stack_hidden_dim,
                layer_norm_position=values.adaptive_generator_stack_layer_norm_position,
                normalization=values.adaptive_generator_stack_normalization,
                num_layers=values.adaptive_generator_stack_num_layers,
                activation=values.adaptive_generator_stack_activation,
                residual_connection_option=values.adaptive_generator_stack_residual_connection_option,
                residual_block_size=values.adaptive_generator_stack_residual_block_size,
                residual_rms_norm_epsilon=values.adaptive_generator_stack_residual_rms_norm_epsilon,
                residual_model_flag=values.adaptive_generator_stack_residual_model_flag,
                dropout_probability=values.adaptive_generator_stack_dropout_probability,
                last_layer_bias_option=values.adaptive_generator_stack_last_layer_bias_option,
                apply_output_postprocessing_flag=values.adaptive_generator_stack_apply_output_postprocessing_flag,
                bias_flag=values.adaptive_generator_stack_bias_flag,
            )
        if isinstance(provided, self.generator_type):
            return provided
        return create_generator(
            hidden_dim=provided.hidden_dim,
            layer_norm_position=provided.layer_norm_position,
            normalization=provided.normalization,
            num_layers=provided.num_layers,
            activation=provided.activation,
            residual_connection_option=provided.residual_connection_option,
            residual_block_size=provided.residual_block_size,
            residual_rms_norm_epsilon=provided.residual_rms_norm_epsilon,
            residual_model_flag=provided.residual_model_flag,
            dropout_probability=provided.dropout_probability,
            last_layer_bias_option=provided.last_layer_bias_option,
            apply_output_postprocessing_flag=provided.apply_output_postprocessing_flag,
            bias_flag=values.adaptive_generator_stack_bias_flag,
            residual_stack_options=self._foreign_generator_residual(provided),
        )

    def _hidden_adaptive_defaults(
        self, values: AdaptiveSettings
    ) -> _HiddenAdaptiveDefaults:
        hidden_adaptive_weight_options = values.hidden_adaptive_weight_options
        hidden_adaptive_bias_options = values.hidden_adaptive_bias_options
        hidden_adaptive_diagonal_options = values.hidden_adaptive_diagonal_options
        hidden_adaptive_mask_options = values.hidden_adaptive_mask_options
        hidden_adaptive_weight_options = (
            hidden_adaptive_weight_options
            or self.weight_type(
                generator_depth=values.generator_depth,
                option_flag=self._adaptive_option_flag(
                    values.weight_option_flag,
                    values.weight_option,
                    self.config.WEIGHT_OPTION_FLAG,
                ),
                option=values.weight_option,
                normalization_option=values.weight_normalization_option,
                normalization_learnable_scale_flag=values.weight_normalization_learnable_scale_flag,
                normalization_position_option=values.weight_normalization_position_option,
                decay_schedule=values.weight_decay_schedule,
                decay_rate=values.weight_decay_rate,
                decay_warmup_batches=values.weight_decay_warmup_batches,
                bank_expansion_factor=values.weight_bank_expansion_factor,
                generator_stack_source=read_optional_stack(
                    self.source_type, values, "weight_generator_stack"
                ),
            )
        )
        hidden_adaptive_bias_options = hidden_adaptive_bias_options or self.bias_type(
            option_flag=self._adaptive_option_flag(
                values.bias_option_flag,
                values.bias_option,
                self.config.BIAS_OPTION_FLAG,
            ),
            option=values.bias_option,
            decay_schedule=values.bias_decay_schedule,
            decay_rate=values.bias_decay_rate,
            decay_warmup_batches=values.bias_decay_warmup_batches,
            bank_expansion_factor=values.bias_bank_expansion_factor,
            generator_stack_source=read_optional_stack(
                self.source_type, values, "bias_generator_stack"
            ),
        )
        hidden_adaptive_diagonal_options = (
            hidden_adaptive_diagonal_options
            or self.diagonal_type(
                option_flag=self._adaptive_option_flag(
                    values.diagonal_option_flag,
                    values.diagonal_option,
                    self.config.DIAGONAL_OPTION_FLAG,
                ),
                option=values.diagonal_option,
                generator_stack_source=read_optional_stack(
                    self.source_type, values, "diagonal_generator_stack"
                ),
            )
        )
        hidden_adaptive_mask_options = hidden_adaptive_mask_options or self.mask_type(
            option_flag=self._adaptive_option_flag(
                values.mask_option_flag,
                values.row_mask_option,
                self.config.MASK_OPTION_FLAG,
            ),
            row_mask_option=values.row_mask_option,
            mask_dimension_option=values.mask_dimension_option,
            mask_threshold=values.mask_threshold,
            mask_surrogate_scale=values.mask_surrogate_scale,
            mask_floor=values.mask_floor,
            mask_transition_width=values.mask_transition_width,
            generator_stack_source=read_optional_stack(
                self.source_type, values, "mask_generator_stack"
            ),
        )
        return _HiddenAdaptiveDefaults(
            weight=hidden_adaptive_weight_options,
            bias=hidden_adaptive_bias_options,
            diagonal=hidden_adaptive_diagonal_options,
            mask=hidden_adaptive_mask_options,
        )

    def _boundary_defaults(self, values: AdaptiveSettings) -> _BoundaryDefaults:
        input_boundary_options = values.input_boundary_options
        output_boundary_options = values.output_boundary_options
        input_boundary_options = input_boundary_options or self.boundary_type(
            weight_option=values.input_layer_weight_option,
            generator_depth=values.input_layer_generator_depth,
            weight_decay_schedule=values.input_layer_weight_decay_schedule,
            weight_decay_rate=values.input_layer_weight_decay_rate,
            weight_decay_warmup_batches=values.input_layer_weight_decay_warmup_batches,
            weight_normalization_option=values.input_layer_weight_normalization_option,
            weight_normalization_learnable_scale_flag=values.input_layer_weight_normalization_learnable_scale_flag,
            weight_normalization_position_option=values.input_layer_weight_normalization_position_option,
            weight_bank_expansion_factor=values.input_layer_weight_bank_expansion_factor,
            bias_option=values.input_layer_bias_option,
            bias_decay_schedule=values.input_layer_bias_decay_schedule,
            bias_decay_rate=values.input_layer_bias_decay_rate,
            bias_decay_warmup_batches=values.input_layer_bias_decay_warmup_batches,
            bias_bank_expansion_factor=values.input_layer_bias_bank_expansion_factor,
            diagonal_option=values.input_layer_diagonal_option,
            row_mask_option=values.input_layer_row_mask_option,
            mask_dimension_option=values.input_layer_mask_dimension_option,
            mask_threshold=values.input_layer_mask_threshold,
            mask_surrogate_scale=values.input_layer_mask_surrogate_scale,
            mask_floor=values.input_layer_mask_floor,
            mask_transition_width=values.input_layer_mask_transition_width,
        )
        output_boundary_options = output_boundary_options or self.boundary_type(
            weight_option=values.output_layer_weight_option,
            generator_depth=values.output_layer_generator_depth,
            weight_decay_schedule=values.output_layer_weight_decay_schedule,
            weight_decay_rate=values.output_layer_weight_decay_rate,
            weight_decay_warmup_batches=values.output_layer_weight_decay_warmup_batches,
            weight_normalization_option=values.output_layer_weight_normalization_option,
            weight_normalization_learnable_scale_flag=values.output_layer_weight_normalization_learnable_scale_flag,
            weight_normalization_position_option=values.output_layer_weight_normalization_position_option,
            weight_bank_expansion_factor=values.output_layer_weight_bank_expansion_factor,
            bias_option=values.output_layer_bias_option,
            bias_decay_schedule=values.output_layer_bias_decay_schedule,
            bias_decay_rate=values.output_layer_bias_decay_rate,
            bias_decay_warmup_batches=values.output_layer_bias_decay_warmup_batches,
            bias_bank_expansion_factor=values.output_layer_bias_bank_expansion_factor,
            diagonal_option=values.output_layer_diagonal_option,
            row_mask_option=values.output_layer_row_mask_option,
            mask_dimension_option=values.output_layer_mask_dimension_option,
            mask_threshold=values.output_layer_mask_threshold,
            mask_surrogate_scale=values.output_layer_mask_surrogate_scale,
            mask_floor=values.output_layer_mask_floor,
            mask_transition_width=values.output_layer_mask_transition_width,
        )
        return _BoundaryDefaults(
            input=input_boundary_options, output=output_boundary_options
        )

    def _router_adaptive_defaults(
        self, values: AdaptiveSettings
    ) -> _RouterAdaptiveDefaults:
        router_adaptive_weight_options = values.router_adaptive_weight_options
        router_adaptive_bias_options = values.router_adaptive_bias_options
        router_adaptive_diagonal_options = values.router_adaptive_diagonal_options
        router_adaptive_mask_options = values.router_adaptive_mask_options
        router_adaptive_weight_options = (
            router_adaptive_weight_options
            or self.weight_type(
                generator_depth=values.router_generator_depth,
                option_flag=self._adaptive_option_flag(
                    values.router_weight_option_flag,
                    values.router_weight_option,
                    self.config.ROUTER_WEIGHT_OPTION_FLAG,
                ),
                option=values.router_weight_option,
                normalization_option=values.router_weight_normalization_option,
                normalization_learnable_scale_flag=values.router_weight_normalization_learnable_scale_flag,
                normalization_position_option=values.router_weight_normalization_position_option,
                decay_schedule=values.router_weight_decay_schedule,
                decay_rate=values.router_weight_decay_rate,
                decay_warmup_batches=values.router_weight_decay_warmup_batches,
                bank_expansion_factor=values.router_weight_bank_expansion_factor,
                generator_stack_source=read_optional_stack(
                    self.source_type, values, "router_weight_generator_stack"
                ),
            )
        )
        router_adaptive_bias_options = router_adaptive_bias_options or self.bias_type(
            option_flag=self._adaptive_option_flag(
                values.router_bias_option_flag,
                values.router_bias_option,
                self.config.ROUTER_BIAS_OPTION_FLAG,
            ),
            option=values.router_bias_option,
            decay_schedule=values.router_bias_decay_schedule,
            decay_rate=values.router_bias_decay_rate,
            decay_warmup_batches=values.router_bias_decay_warmup_batches,
            bank_expansion_factor=values.router_bias_bank_expansion_factor,
            generator_stack_source=read_optional_stack(
                self.source_type, values, "router_bias_generator_stack"
            ),
        )
        router_adaptive_diagonal_options = (
            router_adaptive_diagonal_options
            or self.diagonal_type(
                option_flag=self._adaptive_option_flag(
                    values.router_diagonal_option_flag,
                    values.router_diagonal_option,
                    self.config.ROUTER_DIAGONAL_OPTION_FLAG,
                ),
                option=values.router_diagonal_option,
                generator_stack_source=read_optional_stack(
                    self.source_type, values, "router_diagonal_generator_stack"
                ),
            )
        )
        router_adaptive_mask_options = router_adaptive_mask_options or self.mask_type(
            option_flag=self._adaptive_option_flag(
                values.router_mask_option_flag,
                values.router_row_mask_option,
                self.config.ROUTER_MASK_OPTION_FLAG,
            ),
            row_mask_option=values.router_row_mask_option,
            mask_dimension_option=values.router_mask_dimension_option,
            mask_threshold=values.router_mask_threshold,
            mask_surrogate_scale=values.router_mask_surrogate_scale,
            mask_floor=values.router_mask_floor,
            mask_transition_width=values.router_mask_transition_width,
            generator_stack_source=read_optional_stack(
                self.source_type, values, "router_mask_generator_stack"
            ),
        )
        return _RouterAdaptiveDefaults(
            weight=router_adaptive_weight_options,
            bias=router_adaptive_bias_options,
            diagonal=router_adaptive_diagonal_options,
            mask=router_adaptive_mask_options,
        )

    def resolve_adaptive_defaults(self, values: AdaptiveSettings) -> _Result:
        generator_stack = self._adaptive_generator_defaults(values)
        hidden = self._hidden_adaptive_defaults(values)
        boundary = self._boundary_defaults(values)
        router = self._router_adaptive_defaults(values)
        return self.defaults_type(
            adaptive_generator_stack_options=generator_stack,
            hidden_adaptive_weight_options=hidden.weight,
            hidden_adaptive_bias_options=hidden.bias,
            hidden_adaptive_diagonal_options=hidden.diagonal,
            hidden_adaptive_mask_options=hidden.mask,
            input_boundary_options=boundary.input,
            output_boundary_options=boundary.output,
            router_adaptive_weight_options=router.weight,
            router_adaptive_bias_options=router.bias,
            router_adaptive_diagonal_options=router.diagonal,
            router_adaptive_mask_options=router.mask,
        )

    def _foreign_generator_residual(self, generator: _GeneratorSettings) -> object:
        if self.inherit_foreign_generator_residual:
            return generator.residual_stack_options
        return None
