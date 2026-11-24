from dataclasses import dataclass, field
from typing import TypedDict

from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GroupingConfig,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
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
from models._recurrent_variants import RecurrentResidualOptions
from models._stack_options import resolve_expert_controller, resolve_expert_options
from models.experts.linear_adaptive import SETTINGS_VALIDATION, _generation
from models.experts.linear_adaptive._generation import (
    BiasGenerationOptions,
    WeightGenerationOptions,
)
from models.experts.linear_adaptive._residual import (
    ResidualStackOptions,
)
from models.experts.linear_adaptive._residual import (
    inherit_if_unset as inherit_if_unset,
)


@dataclass(frozen=True, slots=True)
class ExpertsStackOptions:
    hidden_dim: int
    bias_flag: bool
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.positive_integer("stack_hidden_dim", self.hidden_dim)
        SETTINGS_VALIDATION.stack(self)


@dataclass(frozen=True, slots=True)
class ExpertsSubmoduleStackOptions:
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
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.positive_integer("stack_hidden_dim", self.hidden_dim)
        SETTINGS_VALIDATION.stack(self)


@dataclass(frozen=True, slots=True)
class ExpertsSubmoduleStackSource:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    normalization: NormalizationOptions | None = field(default=None, kw_only=True)
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float | None
    bias_flag: bool | None

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.optional_stack(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout_probability=self.dropout_probability,
            residual_connection_option=self.residual_connection_option,
            prefix="stack",
        )


def resolve_experts_submodule_stack_options(
    defaults: ExpertsSubmoduleStackOptions,
    *,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
    last_layer_bias_option: LastLayerBiasOptions | None = None,
    apply_output_postprocessing_flag: bool | None = None,
    activation: ActivationOptions | None = None,
    layer_norm_position: LayerNormPositionOptions | None = None,
    normalization: NormalizationOptions | None = None,
    residual_connection_option: type[ResidualConfig] | None = None,
    residual_model_flag: bool | None = None,
    residual_block_size: int | None = None,
    residual_rms_norm_epsilon: float | None = None,
    dropout_probability: float | None = None,
    bias_flag: bool | None = None,
) -> ExpertsSubmoduleStackOptions:
    return resolve_expert_options(
        ExpertsSubmoduleStackOptions,
        defaults,
        residual_stack_last=True,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_postprocessing_flag=apply_output_postprocessing_flag,
        activation=activation,
        layer_norm_position=layer_norm_position,
        normalization=normalization,
        residual_connection_option=residual_connection_option,
        residual_model_flag=residual_model_flag,
        residual_block_size=residual_block_size,
        residual_rms_norm_epsilon=residual_rms_norm_epsilon,
        dropout_probability=dropout_probability,
        bias_flag=bias_flag,
    )


def resolve_experts_controller_stack_options(
    source: ExpertsSubmoduleStackSource,
    defaults: ExpertsSubmoduleStackOptions,
) -> ExpertsSubmoduleStackOptions:
    return resolve_expert_controller(
        source, defaults, resolve_experts_submodule_stack_options
    )


@dataclass(frozen=True, slots=True)
class ExpertsMixtureOptions:
    top_k: int
    num_experts: int
    capacity_factor: float
    dropped_token_behavior: DroppedTokenOptions
    compute_expert_mixture_flag: bool
    weighted_parameters_flag: bool
    weighting_position_option: ExpertWeightingPositionOptions
    routing_initialization_mode: RoutingInitializationMode


@dataclass(frozen=True, slots=True)
class ExpertsSamplerOptions:
    threshold: float
    filter_above_threshold: bool
    num_topk_samples: int
    normalize_probabilities_flag: bool
    noisy_topk_flag: bool
    coefficient_of_variation_loss_weight: float
    switch_loss_weight: float
    zero_centred_loss_weight: float
    mutual_information_loss_weight: float


@dataclass(frozen=True, slots=True)
class ExpertsRouterOptions:
    noisy_topk_flag: bool


@dataclass(frozen=True, slots=True)
class ExpertsLayerControllerOptions:
    stack_gate_flag: bool
    gate_option: LayerGateOptions | None
    gate_activation: ActivationOptions | None
    gate_stack_source: ExpertsSubmoduleStackSource
    stack_halting_flag: bool
    halting_threshold: float
    halting_dropout: float
    halting_hidden_state_mode: HaltingHiddenStateModeOptions
    halting_stack_source: ExpertsSubmoduleStackSource
    halting_output_dim: int
    shared_gate_config: GateConfig | None = None
    halting_option: type[HaltingConfig] = StickBreakingConfig

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.gate(self.stack_gate_flag, self.gate_option)
        SETTINGS_VALIDATION.halting_threshold(
            "halting_threshold", self.halting_threshold
        )
        SETTINGS_VALIDATION.dropout("halting_dropout", self.halting_dropout)
        SETTINGS_VALIDATION.positive_integer(
            "halting_output_dim", self.halting_output_dim
        )
        SETTINGS_VALIDATION.shared_gate(self.stack_gate_flag, self.shared_gate_config)


@dataclass(frozen=True, slots=True)
class ExpertsDynamicMemoryOptions:
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    memory_test_time_training_learning_rate: float | None
    memory_test_time_training_num_inner_steps: int | None
    memory_stack_source: ExpertsSubmoduleStackSource


@dataclass(frozen=True, slots=True)
class ExpertsRecurrentControllerOptions:
    recurrent_flag: bool
    recurrent_max_steps: int
    recurrent_initial_iterations: int = field(default=2, kw_only=True)
    recurrent_gradient_transition_count: int | None = field(default=None, kw_only=True)
    recurrent_no_gradient_transition_count: int | None = field(
        default=None, kw_only=True
    )
    recurrent_iteration_increment: int = field(default=1, kw_only=True)
    recurrent_forward_calls_before_iteration_increment: int = field(
        default=1,
        kw_only=True,
    )
    recurrent_smooth_iteration_growth_flag: bool = field(
        default=False,
        kw_only=True,
    )
    recurrent_layer_norm_position: LayerNormPositionOptions
    recurrent_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    recurrent_stack_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_gate_stack_source: ExpertsSubmoduleStackSource
    recurrent_stack_halting_flag: bool
    recurrent_halting_threshold: float
    recurrent_halting_dropout: float
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    recurrent_halting_stack_source: ExpertsSubmoduleStackSource
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.gate(
            self.recurrent_stack_gate_flag,
            self.recurrent_gate_option,
            prefix="recurrent_",
        )
        SETTINGS_VALIDATION.halting_threshold(
            "recurrent_halting_threshold", self.recurrent_halting_threshold
        )
        SETTINGS_VALIDATION.dropout(
            "recurrent_halting_dropout", self.recurrent_halting_dropout
        )
        SETTINGS_VALIDATION.positive_integer(
            "recurrent_max_steps", self.recurrent_max_steps
        )


@dataclass(frozen=True, slots=True)
class ExpertsAdaptiveGeneratorStackOptions:
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
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.positive_integer("stack_hidden_dim", self.hidden_dim)
        SETTINGS_VALIDATION.stack(self)


@dataclass(frozen=True, slots=True)
class AdaptiveGeneratorStackOptions:
    hidden_dim: int
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    bias_flag: bool
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.positive_integer("stack_hidden_dim", self.hidden_dim)
        SETTINGS_VALIDATION.stack(self)


@dataclass(frozen=True, slots=True)
class HiddenAdaptiveWeightOptions:
    generation: WeightGenerationOptions = field(
        default_factory=WeightGenerationOptions, kw_only=True
    )
    generator_depth: int
    option_flag: bool
    option: type[DynamicWeightConfig] | None
    normalization_option: WeightNormalizationOptions
    normalization_position_option: WeightNormalizationPositionOptions
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack_source: _generation.AdaptiveGeneratorStackSource
    normalization_learnable_scale_flag: bool = False

    def __post_init__(self) -> None:
        if self.option_flag and self.option is None:
            raise ValueError(
                "weight_option must be set when weight_option_flag is True."
            )
        SETTINGS_VALIDATION.nonnegative_float("weight_decay_rate", self.decay_rate)
        SETTINGS_VALIDATION.nonnegative_integer(
            "weight_decay_warmup_batches", self.decay_warmup_batches
        )


@dataclass(frozen=True, slots=True)
class HiddenAdaptiveBiasOptions:
    generation: BiasGenerationOptions = field(
        default_factory=BiasGenerationOptions, kw_only=True
    )
    option_flag: bool
    option: type[DynamicBiasConfig] | None
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack_source: _generation.AdaptiveGeneratorStackSource

    def __post_init__(self) -> None:
        if self.option_flag and self.option is None:
            raise ValueError("bias_option must be set when bias_option_flag is True.")
        SETTINGS_VALIDATION.nonnegative_float("bias_decay_rate", self.decay_rate)
        SETTINGS_VALIDATION.nonnegative_integer(
            "bias_decay_warmup_batches", self.decay_warmup_batches
        )


@dataclass(frozen=True, slots=True)
class HiddenAdaptiveDiagonalOptions:
    option_flag: bool
    option: type[DynamicDiagonalConfig] | None
    generator_stack_source: _generation.AdaptiveGeneratorStackSource

    def __post_init__(self) -> None:
        if self.option_flag and self.option is None:
            raise ValueError(
                "diagonal_option must be set when diagonal_option_flag is True."
            )


@dataclass(frozen=True, slots=True)
class HiddenAdaptiveMaskOptions:
    option_flag: bool
    row_mask_option: type[AxisMaskConfig] | None
    mask_dimension_option: MaskDimensionOptions
    mask_threshold: float
    mask_surrogate_scale: float
    mask_floor: float
    mask_transition_width: float
    generator_stack_source: _generation.AdaptiveGeneratorStackSource

    def __post_init__(self) -> None:
        if self.option_flag and self.row_mask_option is None:
            raise ValueError(
                "row_mask_option must be set when mask_option_flag is True."
            )
        SETTINGS_VALIDATION.probability("mask_threshold", self.mask_threshold)
        SETTINGS_VALIDATION.probability("mask_floor", self.mask_floor)
        SETTINGS_VALIDATION.positive_float(
            "mask_surrogate_scale", self.mask_surrogate_scale
        )
        SETTINGS_VALIDATION.positive_float(
            "mask_transition_width", self.mask_transition_width
        )


@dataclass(frozen=True, slots=True)
class AdaptiveBoundaryModelOptions:
    weight_generation: WeightGenerationOptions = field(
        default_factory=WeightGenerationOptions, kw_only=True
    )
    bias_generation: BiasGenerationOptions = field(
        default_factory=BiasGenerationOptions, kw_only=True
    )
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    weight_option: type[DynamicWeightConfig] | None
    generator_depth: int
    weight_decay_schedule: WeightDecayScheduleOptions
    weight_decay_rate: float
    weight_decay_warmup_batches: int
    weight_normalization_option: WeightNormalizationOptions
    weight_normalization_learnable_scale_flag: bool
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
class RuntimeOptions(RecurrentResidualOptions):
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    router_grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    batch_size: int
    learning_rate: float
    input_dim: int
    output_dim: int
    stack_options: ExpertsStackOptions
    submodule_stack_options: ExpertsSubmoduleStackOptions
    mixture_options: ExpertsMixtureOptions
    expert_stack_options: ExpertsSubmoduleStackOptions
    sampler_options: ExpertsSamplerOptions
    router_options: ExpertsRouterOptions
    router_stack_options: ExpertsSubmoduleStackOptions
    router_layer_controller_options: ExpertsLayerControllerOptions
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions
    layer_controller_options: ExpertsLayerControllerOptions
    dynamic_memory_options: ExpertsDynamicMemoryOptions
    expert_layer_controller_options: ExpertsLayerControllerOptions
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions
    input_boundary_options: AdaptiveBoundaryModelOptions
    output_boundary_options: AdaptiveBoundaryModelOptions
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions
    recurrent_controller_options: ExpertsRecurrentControllerOptions

    @property
    def hidden_dim(self) -> int:
        return self.stack_options.hidden_dim

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.dimensions(self)


class PresetValues(TypedDict, total=False):
    bias_option: type[DynamicBiasConfig] | None
    capacity_factor: float
    diagonal_option: type[DynamicDiagonalConfig] | None
    dropped_token_behavior: DroppedTokenOptions
    generator_depth: int
    layer_norm_position: LayerNormPositionOptions
    memory_flag: bool
    recurrent_flag: bool
    recurrent_stack_gate_flag: bool
    recurrent_stack_halting_flag: bool
    router_weight_option: type[DynamicWeightConfig] | None
    routing_initialization_mode: RoutingInitializationMode
    row_mask_option: type[AxisMaskConfig] | None
    sampler_normalize_probabilities_flag: bool
    sampler_switch_loss_weight: float
    stack_gate_flag: bool
    stack_halting_flag: bool
    stack_residual_connection_option: type[ResidualConfig] | None
    top_k: int
    weight_decay_rate: float
    weight_decay_schedule: WeightDecayScheduleOptions
    weight_decay_warmup_batches: int
    weight_normalization_option: WeightNormalizationOptions
    weight_option: type[DynamicWeightConfig] | None
    weighting_position_option: ExpertWeightingPositionOptions
