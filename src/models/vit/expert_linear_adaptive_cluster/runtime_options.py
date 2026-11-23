from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

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
from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
    SoftMoEConfig,
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
from emperor.memory import (
    DynamicMemoryConfig,
    MemoryPositionOptions,
)
from emperor.neuron import NeuronSequenceRoutingOptions, TerminalRangeOptions
from model_runtime.packages.runtime_values import ResolvedRuntimeOptions
from models import _attention_settings, _sequence_expansion
from models import _attention_source_memory as _source_memory
from models._recurrent_compression import CompressionOptions
from models._stack_options import (
    resolve_controller_options,
    resolve_expert_controller,
    resolve_expert_options,
)
from models.vit.expert_linear_adaptive_cluster import _generation
from models.vit.expert_linear_adaptive_cluster._generation import (
    BiasGenerationOptions,
    WeightGenerationOptions,
)
from models.vit.expert_linear_adaptive_cluster._residual import (
    ResidualStackOptions,
)
from models.vit.expert_linear_adaptive_cluster._residual import (
    inherit_if_unset as inherit_if_unset,
)


@dataclass(frozen=True)
class SubmoduleStackSource:
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
        validation = _attention_settings.strict_float_validation(__package__)
        _attention_settings.validate_stack_source(validation, self)


@dataclass(frozen=True)
class SubmoduleStackOptions:
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
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float
    bias_flag: bool

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("hidden_dim", self.hidden_dim)
        validation.stack(self)


def resolve_controller_stack_options(
    source: SubmoduleStackSource, defaults: SubmoduleStackOptions
) -> SubmoduleStackOptions:
    return resolve_controller_options(source, defaults, SubmoduleStackOptions)


@dataclass(frozen=True)
class MainLayerStackOptions:
    bias_flag: bool
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: type[ResidualConfig]
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.stack(self)


@dataclass(frozen=True)
class LayerControllerOptions:
    stack_gate_flag: bool
    gate_option: LayerGateOptions | None
    gate_activation: ActivationOptions | None
    gate_stack_source: SubmoduleStackSource
    stack_halting_flag: bool
    halting_threshold: float
    halting_dropout: float
    halting_hidden_state_mode: HaltingHiddenStateModeOptions
    halting_stack_source: SubmoduleStackSource
    shared_gate_config: GateConfig | None = None
    halting_option: type[HaltingConfig] = StickBreakingConfig

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        _attention_settings.validate_controller(validation, self)


@dataclass(frozen=True)
class DynamicMemoryOptions:
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    memory_test_time_training_learning_rate: float | None
    memory_num_memory_slots: int = field(default=4, kw_only=True)
    memory_test_time_training_num_inner_steps: int | None
    memory_stack_source: SubmoduleStackSource

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        _attention_settings.validate_memory(validation, self)


@dataclass(frozen=True)
class RecurrentControllerOptions:
    recurrent_min_steps: int = field(default=1, kw_only=True)
    recurrent_ponder_cost_weight: float = field(default=1.0, kw_only=True)
    recurrent_residual_connection_option: type[ResidualConfig] | None = field(
        default=None, kw_only=True
    )
    recurrent_residual_model_flag: bool = field(default=False, kw_only=True)
    recurrent_residual_block_size: int | None = field(default=None, kw_only=True)
    recurrent_residual_rms_norm_epsilon: float | None = field(
        default=None, kw_only=True
    )
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
    recurrent_gate_stack_source: SubmoduleStackSource
    recurrent_stack_halting_flag: bool
    recurrent_halting_threshold: float
    recurrent_halting_dropout: float
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    recurrent_halting_stack_source: SubmoduleStackSource
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("recurrent_max_steps", self.recurrent_max_steps)
        _attention_settings.validate_recurrent_controller(validation, self)
        _attention_settings.validate_residual(
            "recurrent_residual_connection_option",
            self.recurrent_residual_connection_option,
        )


@dataclass(frozen=True)
class TransformerEncoderOptions(_sequence_expansion.SequenceExpansionOptions):
    output_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    dropout_probability: float
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    causal_attention_mask_flag: bool = False

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("hidden_dim", self.hidden_dim)
        validation.positive_integer("num_layers", self.num_layers)
        validation.dropout("dropout_probability", self.dropout_probability)


@dataclass(frozen=True)
class TransformerPositionalEmbeddingOptions:
    option: type[AbsolutePositionalEmbeddingConfig]
    padding_idx: int | None
    auto_expand_flag: bool


@dataclass(frozen=True)
class TransformerAttentionOptions(_source_memory.AttentionSourceMemoryOptions):
    attention_soft_moe_config: SoftMoEConfig | None = field(default=None, kw_only=True)
    num_heads: int
    num_layers: int
    bias_flag: bool
    add_key_value_bias_flag: bool
    source_memory_flag: bool = False

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("num_layers", self.num_layers)
        validation.positive_integer("num_heads", self.num_heads)


@dataclass(frozen=True)
class TransformerFeedForwardOptions:
    soft_moe_config: SoftMoEConfig | None = field(default=None, kw_only=True)
    num_layers: int
    bias_flag: bool

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("num_layers", self.num_layers)


@dataclass(frozen=True)
class VitPatchOptions:
    patch_size: int
    input_channels: int
    image_height: int
    dropout_probability: float
    bias_flag: bool

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("patch_size", self.patch_size)
        validation.positive_integer("input_channels", self.input_channels)
        validation.positive_integer("image_height", self.image_height)
        validation.dropout("dropout_probability", self.dropout_probability)


@dataclass(frozen=True)
class VitOutputOptions:
    bias_flag: bool


@dataclass(frozen=True)
class ExpertsStackOptions:
    hidden_dim: int
    bias_flag: bool
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: type[ResidualConfig]
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("hidden_dim", self.hidden_dim)
        validation.stack(self)


@dataclass(frozen=True)
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
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float
    bias_flag: bool

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("hidden_dim", self.hidden_dim)
        validation.stack(self)


@dataclass(frozen=True)
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
        validation = _attention_settings.strict_float_validation(__package__)
        _attention_settings.validate_stack_source(validation, self)


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
        residual_stack_last=False,
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
    source: ExpertsSubmoduleStackSource, defaults: ExpertsSubmoduleStackOptions
) -> ExpertsSubmoduleStackOptions:
    return resolve_expert_controller(
        source, defaults, resolve_experts_submodule_stack_options
    )


@dataclass(frozen=True)
class ExpertsMixtureOptions:
    top_k: int
    num_experts: int
    capacity_factor: float
    dropped_token_behavior: DroppedTokenOptions
    compute_expert_mixture_flag: bool
    weighted_parameters_flag: bool
    weighting_position_option: ExpertWeightingPositionOptions
    routing_initialization_mode: RoutingInitializationMode


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class ExpertsRouterOptions:
    noisy_topk_flag: bool


@dataclass(frozen=True)
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
        validation = _attention_settings.strict_float_validation(__package__)
        _attention_settings.validate_controller(validation, self)


@dataclass(frozen=True)
class ExpertsDynamicMemoryOptions:
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    memory_test_time_training_learning_rate: float | None
    memory_num_memory_slots: int = field(default=4, kw_only=True)
    memory_test_time_training_num_inner_steps: int | None
    memory_stack_source: ExpertsSubmoduleStackSource

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        _attention_settings.validate_memory(validation, self)


@dataclass(frozen=True)
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
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("recurrent_max_steps", self.recurrent_max_steps)
        _attention_settings.validate_recurrent_controller(validation, self)


@dataclass(frozen=True)
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
    residual_connection_option: type[ResidualConfig]
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("hidden_dim", self.hidden_dim)
        validation.stack(self)


@dataclass(frozen=True)
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
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    bias_flag: bool

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("hidden_dim", self.hidden_dim)
        validation.stack(self)


@dataclass(frozen=True)
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
        validation = _attention_settings.strict_float_validation(__package__)
        if self.option_flag and self.option is None:
            raise ValueError(
                "weight_option must be set when weight_option_flag is True"
            )
        validation.positive_integer("generator_depth", self.generator_depth)
        validation.nonnegative_float("decay_rate", self.decay_rate)
        validation.nonnegative_integer(
            "decay_warmup_batches", self.decay_warmup_batches
        )


@dataclass(frozen=True)
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
        validation = _attention_settings.strict_float_validation(__package__)
        if self.option_flag and self.option is None:
            raise ValueError("bias_option must be set when bias_option_flag is True")
        validation.nonnegative_float("decay_rate", self.decay_rate)
        validation.nonnegative_integer(
            "decay_warmup_batches", self.decay_warmup_batches
        )


@dataclass(frozen=True)
class HiddenAdaptiveDiagonalOptions:
    option_flag: bool
    option: type[DynamicDiagonalConfig] | None
    generator_stack_source: _generation.AdaptiveGeneratorStackSource

    def __post_init__(self) -> None:
        if self.option_flag and self.option is None:
            raise ValueError(
                "diagonal_option must be set when diagonal_option_flag is True"
            )


@dataclass(frozen=True)
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
        validation = _attention_settings.strict_float_validation(__package__)
        if self.option_flag and self.row_mask_option is None:
            raise ValueError(
                "row_mask_option must be set when mask_option_flag is True"
            )
        _attention_settings.validate_mask(validation, self)


@dataclass(frozen=True, slots=True)
class RuntimeOptions(ResolvedRuntimeOptions, CompressionOptions):
    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        ResolvedRuntimeOptions.__post_init__(self)
        _attention_settings.validate_dimensions(validation, self._values)


@dataclass(frozen=True)
class ClusterOptions:
    routing_mode: NeuronSequenceRoutingOptions
    x_axis_total_neurons: int
    y_axis_total_neurons: int
    z_axis_total_neurons: int
    initial_x_axis_total_neurons: int | None
    initial_y_axis_total_neurons: int | None
    initial_z_axis_total_neurons: int | None
    attention_residual_flag: bool
    max_steps: int
    terminal_top_k: int
    terminal_xy_axis_range: TerminalRangeOptions
    terminal_z_axis_range: TerminalRangeOptions
    halting_flag: bool
    halting_threshold: float
    halting_min_steps: int
    halting_ponder_cost_weight: float
    growth_threshold: int | None
    growth_warmup_steps: int | None
    pruning_threshold: int | None

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("x_axis_total_neurons", self.x_axis_total_neurons)
        validation.positive_integer("y_axis_total_neurons", self.y_axis_total_neurons)
        validation.positive_integer("z_axis_total_neurons", self.z_axis_total_neurons)
        validation.positive_integer("max_steps", self.max_steps)
        validation.positive_integer("terminal_top_k", self.terminal_top_k)
        validation.positive_integer("halting_min_steps", self.halting_min_steps)
        validation.halting_threshold("halting_threshold", self.halting_threshold)
        validation.nonnegative_float(
            "halting_ponder_cost_weight", self.halting_ponder_cost_weight
        )

    @property
    def max_visits(self) -> int:
        """Maximum number of consecutive encoders that can expand the same image."""
        if self.routing_mode is NeuronSequenceRoutingOptions.LEARNED:
            return self.max_steps + 1
        return (
            (self.initial_x_axis_total_neurons or self.x_axis_total_neurons)
            * (self.initial_y_axis_total_neurons or self.y_axis_total_neurons)
            * (self.initial_z_axis_total_neurons or self.z_axis_total_neurons)
        )


class PresetValues(TypedDict, total=False):
    attn_add_key_value_bias_flag: bool
    attn_bias_flag: bool
    cluster_routing_mode: NeuronSequenceRoutingOptions
    cluster_x_axis_total_neurons: int
    layer_norm_position: LayerNormPositionOptions
    positional_embedding_option: type[AbsolutePositionalEmbeddingConfig]
    sampler_normalize_probabilities_flag: bool
    sampler_switch_loss_weight: float
    stack_num_layers: int
    top_k: int
    weight_option: type[DynamicWeightConfig] | None
    weight_option_flag: bool
