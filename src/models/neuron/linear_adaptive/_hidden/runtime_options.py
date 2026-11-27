from __future__ import annotations

from dataclasses import dataclass, field

from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GroupingConfig,
    LowRankFactorSourceOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from models.neuron.linear_adaptive._hidden import SETTINGS_VALIDATION
from models.neuron.linear_adaptive._residual import ResidualStackOptions

_PACKAGE = "models.neuron.linear_adaptive._hidden"


@dataclass(frozen=True, slots=True)
class StackOptions:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float
    bias_flag: bool

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.positive_integer("stack.hidden_dim", self.hidden_dim)
        SETTINGS_VALIDATION.stack(self)


@dataclass(frozen=True, slots=True)
class GateOptions:
    enabled: bool
    option: LayerGateOptions | None
    activation: ActivationOptions | None
    stack: StackOptions
    shared_config: GateConfig | None = None

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.gate(self.enabled, self.option)


@dataclass(frozen=True, slots=True)
class HaltingOptions:
    enabled: bool
    threshold: float
    dropout_probability: float
    hidden_state_mode: HaltingHiddenStateModeOptions
    stack: StackOptions

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.halting_threshold("halting_threshold", self.threshold)
        SETTINGS_VALIDATION.probability(
            "halting.dropout_probability", self.dropout_probability
        )


@dataclass(frozen=True, slots=True)
class MemoryOptions:
    enabled: bool
    option: type[DynamicMemoryConfig]
    position: MemoryPositionOptions
    test_time_training_learning_rate: float | None
    test_time_training_num_inner_steps: int | None
    stack: StackOptions

    def __post_init__(self) -> None:
        if self.test_time_training_learning_rate is not None:
            SETTINGS_VALIDATION.positive_float(
                "memory.test_time_training_learning_rate",
                self.test_time_training_learning_rate,
            )
        if self.test_time_training_num_inner_steps is not None:
            SETTINGS_VALIDATION.positive_integer(
                "memory.test_time_training_num_inner_steps",
                self.test_time_training_num_inner_steps,
            )


@dataclass(frozen=True, slots=True)
class RecurrenceOptions:
    enabled: bool
    max_steps: int
    initial_iterations: int = field(default=2, kw_only=True)
    gradient_transition_count: int | None = field(default=None, kw_only=True)
    no_gradient_transition_count: int | None = field(default=None, kw_only=True)
    iteration_increment: int = field(default=1, kw_only=True)
    forward_calls_before_iteration_increment: int = field(default=1, kw_only=True)
    smooth_iteration_growth_flag: bool = field(default=False, kw_only=True)
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    gate: GateOptions
    halting: HaltingOptions

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.positive_integer("recurrence.max_steps", self.max_steps)


@dataclass(frozen=True, slots=True)
class GeneratorStackOptions:
    """A resolved generator stack plus whether it replaces the shared stack."""

    independent: bool
    stack: StackOptions


@dataclass(frozen=True, slots=True)
class AdaptiveGenerationOptions:
    weight_input_factor_source: LowRankFactorSourceOptions | None = None
    weight_output_factor_source: LowRankFactorSourceOptions | None = None
    weight_mixture_num_experts: int | None = None
    bias_mixture_num_experts: int | None = None
    bias_mixture_top_k: int | None = None
    bias_mixture_normalize_probabilities_flag: bool | None = None
    weight_input_factor_generator_stack: GeneratorStackOptions | None = None
    weight_output_factor_generator_stack: GeneratorStackOptions | None = None
    weight_coefficient_generator_stack: GeneratorStackOptions | None = None
    weight_mixture_router_generator_stack: GeneratorStackOptions | None = None
    bias_mixture_router_generator_stack: GeneratorStackOptions | None = None


@dataclass(frozen=True, slots=True)
class AdaptiveWeightOptions:
    enabled: bool
    option: type[DynamicWeightConfig] | None
    generator_depth: int
    normalization_option: WeightNormalizationOptions
    normalization_position_option: WeightNormalizationPositionOptions
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack: GeneratorStackOptions
    normalization_learnable_scale_flag: bool = False

    def __post_init__(self) -> None:
        if self.enabled and self.option is None:
            raise ValueError(
                f"{_PACKAGE}: 'weight.option' must be set when weight is enabled"
            )
        SETTINGS_VALIDATION.nonnegative_float("weight.decay_rate", self.decay_rate)
        SETTINGS_VALIDATION.nonnegative_integer(
            "weight.decay_warmup_batches", self.decay_warmup_batches
        )


@dataclass(frozen=True, slots=True)
class AdaptiveBiasOptions:
    enabled: bool
    option: type[DynamicBiasConfig] | None
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack: GeneratorStackOptions

    def __post_init__(self) -> None:
        if self.enabled and self.option is None:
            raise ValueError(
                f"{_PACKAGE}: 'bias.option' must be set when bias is enabled"
            )
        SETTINGS_VALIDATION.nonnegative_float("bias.decay_rate", self.decay_rate)
        SETTINGS_VALIDATION.nonnegative_integer(
            "bias.decay_warmup_batches", self.decay_warmup_batches
        )


@dataclass(frozen=True, slots=True)
class AdaptiveDiagonalOptions:
    enabled: bool
    option: type[DynamicDiagonalConfig] | None
    generator_stack: GeneratorStackOptions

    def __post_init__(self) -> None:
        if self.enabled and self.option is None:
            raise ValueError(
                f"{_PACKAGE}: 'diagonal.option' must be set when diagonal is enabled"
            )


@dataclass(frozen=True, slots=True)
class AdaptiveMaskOptions:
    enabled: bool
    row_mask_option: type[AxisMaskConfig] | None
    dimension_option: MaskDimensionOptions
    threshold: float
    surrogate_scale: float
    floor: float
    transition_width: float
    generator_stack: GeneratorStackOptions

    def __post_init__(self) -> None:
        if self.enabled and self.row_mask_option is None:
            raise ValueError(
                f"{_PACKAGE}: 'mask.row_mask_option' must be set when mask is enabled"
            )
        SETTINGS_VALIDATION.probability("mask.threshold", self.threshold)
        SETTINGS_VALIDATION.probability("mask.floor", self.floor)
        SETTINGS_VALIDATION.positive_float("mask.surrogate_scale", self.surrogate_scale)
        SETTINGS_VALIDATION.positive_float(
            "mask.transition_width", self.transition_width
        )


@dataclass(frozen=True, slots=True)
class ProjectionMaskOptions:
    row_mask_option: type[AxisMaskConfig] | None
    dimension_option: MaskDimensionOptions
    threshold: float
    surrogate_scale: float
    floor: float
    transition_width: float


@dataclass(frozen=True, slots=True)
class AdaptiveProjectionOptions:
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    generation: AdaptiveGenerationOptions = field(
        default_factory=AdaptiveGenerationOptions, kw_only=True
    )
    weight_option: type[DynamicWeightConfig] | None
    generator_depth: int
    weight_decay_schedule: WeightDecayScheduleOptions
    weight_decay_rate: float
    weight_decay_warmup_batches: int
    weight_normalization_option: WeightNormalizationOptions
    weight_normalization_position_option: WeightNormalizationPositionOptions
    weight_bank_expansion_factor: BankExpansionFactorOptions
    bias_option: type[DynamicBiasConfig] | None
    bias_decay_schedule: WeightDecayScheduleOptions
    bias_decay_rate: float
    bias_decay_warmup_batches: int
    bias_bank_expansion_factor: BankExpansionFactorOptions
    diagonal_option: type[DynamicDiagonalConfig] | None
    row_mask_option: type[AxisMaskConfig] | None
    mask_dimension_option: MaskDimensionOptions
    mask_threshold: float
    mask_surrogate_scale: float
    mask_floor: float
    mask_transition_width: float
    weight_normalization_learnable_scale_flag: bool = False

    @property
    def mask(self) -> ProjectionMaskOptions:
        return ProjectionMaskOptions(
            row_mask_option=self.row_mask_option,
            dimension_option=self.mask_dimension_option,
            threshold=self.mask_threshold,
            surrogate_scale=self.mask_surrogate_scale,
            floor=self.mask_floor,
            transition_width=self.mask_transition_width,
        )

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.nonnegative_float(
            "projection.weight_decay_rate", self.weight_decay_rate
        )
        SETTINGS_VALIDATION.nonnegative_integer(
            "projection.weight_decay_warmup_batches",
            self.weight_decay_warmup_batches,
        )
        SETTINGS_VALIDATION.nonnegative_float(
            "projection.bias_decay_rate", self.bias_decay_rate
        )
        SETTINGS_VALIDATION.nonnegative_integer(
            "projection.bias_decay_warmup_batches",
            self.bias_decay_warmup_batches,
        )
        SETTINGS_VALIDATION.probability(
            "projection.mask_threshold", self.mask_threshold
        )
        SETTINGS_VALIDATION.probability("projection.mask_floor", self.mask_floor)
        SETTINGS_VALIDATION.positive_float(
            "projection.mask_surrogate_scale", self.mask_surrogate_scale
        )
        SETTINGS_VALIDATION.positive_float(
            "projection.mask_transition_width", self.mask_transition_width
        )


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    generation: AdaptiveGenerationOptions = field(
        default_factory=AdaptiveGenerationOptions, kw_only=True
    )
    batch_size: int
    learning_rate: float
    input_dim: int
    hidden_dim: int
    output_dim: int
    stack: StackOptions
    submodule_stack: StackOptions
    residual_stack: ResidualStackOptions
    gate: GateOptions
    halting: HaltingOptions
    memory: MemoryOptions
    recurrence: RecurrenceOptions
    adaptive_generator_stack: StackOptions
    weight: AdaptiveWeightOptions
    bias: AdaptiveBiasOptions
    diagonal: AdaptiveDiagonalOptions
    mask: AdaptiveMaskOptions
    input_projection: AdaptiveProjectionOptions
    output_projection: AdaptiveProjectionOptions
    halting_option: type[HaltingConfig] = StickBreakingConfig
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.dimensions(self)
        SETTINGS_VALIDATION.shared_gate(self.gate.enabled, self.gate.shared_config)
