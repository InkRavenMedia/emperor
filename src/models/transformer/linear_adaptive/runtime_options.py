from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

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
from emperor.embedding.absolute import (
    AbsolutePositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
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
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
)
from models import _attention_settings, _sequence_expansion
from models import _attention_source_memory as _source_memory
from models._stack_options import resolve_transformer_controller_options

from ._residual import inherit_if_unset as inherit_if_unset


@dataclass(frozen=True)
class TransformerStackOptions(_sequence_expansion.SequenceExpansionOptions):
    num_layers: int = 3
    layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    stack_gate_flag: bool = False
    stack_halting_flag: bool = False
    halting_option: type[HaltingConfig] = StickBreakingConfig
    halting_threshold: float | None = None
    memory_flag: bool = False
    recurrent_flag: bool = False
    recurrent_stack_gate_flag: bool = False
    recurrent_stack_halting_flag: bool = False
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig
    recurrent_halting_threshold: float | None = None
    recurrent_max_steps: int = 2
    recurrent_initial_iterations: int = 2
    recurrent_gradient_transition_count: int | None = None
    recurrent_no_gradient_transition_count: int | None = None
    recurrent_iteration_increment: int = 1
    recurrent_forward_calls_before_iteration_increment: int = 1
    recurrent_smooth_iteration_growth_flag: bool = False
    stack_residual_connection_option: type[ResidualConfig] | None = None
    stack_residual_model_flag: bool = field(default=False, kw_only=True)
    stack_residual_block_size: int | None = field(default=None, kw_only=True)
    stack_residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    recurrent_residual_connection_option: type[ResidualConfig] | None = None
    recurrent_residual_model_flag: bool = field(default=False, kw_only=True)
    recurrent_residual_block_size: int | None = field(default=None, kw_only=True)
    recurrent_residual_rms_norm_epsilon: float | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        validation = _attention_settings.integer_compatible_validation(__package__)
        validation.positive_integer("num_layers", self.num_layers)
        if self.halting_threshold is not None:
            validation.halting_threshold("halting_threshold", self.halting_threshold)
        if self.recurrent_halting_threshold is not None:
            validation.halting_threshold(
                "recurrent_halting_threshold", self.recurrent_halting_threshold
            )
        _attention_settings.validate_residual(
            "stack_residual_connection_option", self.stack_residual_connection_option
        )
        _attention_settings.validate_residual(
            "recurrent_residual_connection_option",
            self.recurrent_residual_connection_option,
        )


@dataclass(frozen=True)
class SubmoduleStackOptions:
    hidden_dim: int = 128
    num_layers: int = 1
    last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
    apply_output_postprocessing_flag: bool = False
    activation: ActivationOptions = ActivationOptions.DISABLED
    layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    residual_connection_option: type[ResidualConfig] | None = None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float = 0.0
    bias_flag: bool = True

    def __post_init__(self) -> None:
        validation = _attention_settings.integer_compatible_validation(__package__)
        validation.positive_integer("hidden_dim", self.hidden_dim)
        validation.stack(self)


@dataclass(frozen=True)
class ControllerStackOptions:
    independent_flag: bool = False
    hidden_dim: int | None = None
    num_layers: int | None = None
    last_layer_bias_option: LastLayerBiasOptions | None = None
    apply_output_postprocessing_flag: bool | None = None
    activation: ActivationOptions | None = None
    layer_norm_position: LayerNormPositionOptions | None = None
    normalization: NormalizationOptions | None = field(default=None, kw_only=True)
    residual_connection_option: type[ResidualConfig] | None = None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float | None = None
    bias_flag: bool | None = None

    def __post_init__(self) -> None:
        validation = _attention_settings.integer_compatible_validation(__package__)
        _attention_settings.validate_stack_source(validation, self)


def resolve_controller_stack_options(
    source: ControllerStackOptions,
    defaults: SubmoduleStackOptions,
) -> SubmoduleStackOptions:
    return resolve_transformer_controller_options(
        source, defaults, SubmoduleStackOptions
    )


@dataclass(frozen=True)
class LayerControllerOptions:
    stack_gate_flag: bool = False
    gate_option: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
    gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID
    gate_stack_options: ControllerStackOptions = ControllerStackOptions()
    stack_halting_flag: bool = False
    halting_option: type[HaltingConfig] = StickBreakingConfig
    halting_threshold: float | None = None
    halting_dropout: float = 0.0
    halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        HaltingHiddenStateModeOptions.RAW
    )
    halting_stack_options: ControllerStackOptions = ControllerStackOptions()
    shared_gate_config: GateConfig | None = None

    def __post_init__(self) -> None:
        validation = _attention_settings.integer_compatible_validation(__package__)
        _attention_settings.validate_controller(
            validation, self, allow_unset_threshold=True
        )


@dataclass(frozen=True)
class DynamicMemoryOptions:
    memory_flag: bool = False
    memory_option: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
    memory_position_option: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
    memory_test_time_training_learning_rate: float | None = None
    memory_test_time_training_num_inner_steps: int | None = None
    memory_stack_options: ControllerStackOptions = ControllerStackOptions()
    memory_num_memory_slots: int = 4

    def __post_init__(self) -> None:
        validation = _attention_settings.integer_compatible_validation(__package__)
        _attention_settings.validate_memory(validation, self)


@dataclass(frozen=True)
class RecurrentControllerOptions:
    recurrent_flag: bool = False
    recurrent_max_steps: int = 2
    recurrent_layer_norm_position: LayerNormPositionOptions = (
        LayerNormPositionOptions.DISABLED
    )
    recurrent_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    recurrent_stack_gate_flag: bool = False
    recurrent_gate_option: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
    recurrent_gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID
    recurrent_gate_stack_options: ControllerStackOptions = ControllerStackOptions()
    recurrent_stack_halting_flag: bool = False
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig
    recurrent_halting_threshold: float | None = None
    recurrent_halting_dropout: float = 0.0
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        HaltingHiddenStateModeOptions.RAW
    )
    recurrent_halting_stack_options: ControllerStackOptions = ControllerStackOptions()

    def __post_init__(self) -> None:
        validation = _attention_settings.integer_compatible_validation(__package__)
        validation.positive_integer("recurrent_max_steps", self.recurrent_max_steps)
        _attention_settings.validate_recurrent_controller(
            validation, self, allow_unset_threshold=True
        )


@dataclass(frozen=True)
class TransformerAttentionOptions(_source_memory.AttentionSourceMemoryOptions):
    num_heads: int = 4
    add_key_value_bias_flag: bool = False
    zero_attention_flag: bool = False
    stack_options: SubmoduleStackOptions = SubmoduleStackOptions()
    layer_controller_options: LayerControllerOptions = LayerControllerOptions()
    dynamic_memory_options: DynamicMemoryOptions = DynamicMemoryOptions()
    recurrent_controller_options: RecurrentControllerOptions = (
        RecurrentControllerOptions()
    )
    source_memory_flag: bool = False

    def __post_init__(self) -> None:
        validation = _attention_settings.integer_compatible_validation(__package__)
        validation.positive_integer("num_heads", self.num_heads)


@dataclass(frozen=True)
class TransformerFeedForwardOptions:
    stack_options: SubmoduleStackOptions = SubmoduleStackOptions(
        hidden_dim=512,
        num_layers=2,
        activation=ActivationOptions.RELU,
        dropout_probability=0.1,
    )
    layer_controller_options: LayerControllerOptions = LayerControllerOptions()
    dynamic_memory_options: DynamicMemoryOptions = DynamicMemoryOptions()
    recurrent_controller_options: RecurrentControllerOptions = (
        RecurrentControllerOptions()
    )


@dataclass(frozen=True)
class AdaptiveParameterOptions:
    weight_input_factor_source: LowRankFactorSourceOptions | None = None
    weight_output_factor_source: LowRankFactorSourceOptions | None = None
    weight_mixture_num_experts: int | None = None
    bias_mixture_num_experts: int | None = None
    weight_mixture_top_k: int | None = None
    bias_mixture_top_k: int | None = None
    weight_mixture_normalize_probabilities_flag: bool | None = None
    bias_mixture_normalize_probabilities_flag: bool | None = None
    weight_input_factor_generator_stack_options: ControllerStackOptions = (
        ControllerStackOptions()
    )
    weight_output_factor_generator_stack_options: ControllerStackOptions = (
        ControllerStackOptions()
    )
    weight_coefficient_generator_stack_options: ControllerStackOptions = (
        ControllerStackOptions()
    )
    weight_mixture_router_generator_stack_options: ControllerStackOptions = (
        ControllerStackOptions()
    )
    bias_mixture_router_generator_stack_options: ControllerStackOptions = (
        ControllerStackOptions()
    )
    grouping_config: GroupingConfig | None = None
    weight_option_flag: bool = True
    weight_option: type[DynamicWeightConfig] | None = None
    generator_depth: int = 1
    weight_decay_schedule: WeightDecayScheduleOptions = (
        WeightDecayScheduleOptions.DISABLED
    )
    weight_decay_rate: float = 0.0
    weight_decay_warmup_batches: int = 0
    weight_normalization_option: WeightNormalizationOptions = (
        WeightNormalizationOptions.DISABLED
    )
    weight_normalization_position_option: WeightNormalizationPositionOptions = (
        WeightNormalizationPositionOptions.DISABLED
    )
    weight_bank_expansion_factor: BankExpansionFactorOptions = (
        BankExpansionFactorOptions.FACTOR_OF_ONE
    )
    bias_option_flag: bool = True
    bias_option: type[DynamicBiasConfig] | None = None
    bias_decay_schedule: WeightDecayScheduleOptions = (
        WeightDecayScheduleOptions.DISABLED
    )
    bias_decay_rate: float = 0.0
    bias_decay_warmup_batches: int = 0
    bias_bank_expansion_factor: BankExpansionFactorOptions = (
        BankExpansionFactorOptions.FACTOR_OF_ONE
    )
    diagonal_option_flag: bool = True
    diagonal_option: type[DynamicDiagonalConfig] | None = None
    mask_option_flag: bool = True
    row_mask_option: type[AxisMaskConfig] | None = None
    mask_threshold: float = 0.5
    mask_surrogate_scale: float = 1.0
    mask_floor: float = 0.0
    mask_dimension_option: MaskDimensionOptions = MaskDimensionOptions.ROW
    mask_transition_width: float = 0.1
    generator_stack_options: SubmoduleStackOptions = SubmoduleStackOptions(
        hidden_dim=64,
        num_layers=1,
        activation=ActivationOptions.RELU,
    )
    weight_generator_stack_options: ControllerStackOptions = ControllerStackOptions()
    bias_generator_stack_options: ControllerStackOptions = ControllerStackOptions()
    diagonal_generator_stack_options: ControllerStackOptions = ControllerStackOptions()
    mask_generator_stack_options: ControllerStackOptions = ControllerStackOptions()
    weight_normalization_learnable_scale_flag: bool = False

    def __post_init__(self) -> None:
        validation = _attention_settings.integer_compatible_validation(__package__)
        validation.positive_integer("generator_depth", self.generator_depth)
        validation.nonnegative_float("weight_decay_rate", self.weight_decay_rate)
        validation.nonnegative_integer(
            "weight_decay_warmup_batches", self.weight_decay_warmup_batches
        )
        validation.nonnegative_float("bias_decay_rate", self.bias_decay_rate)
        validation.nonnegative_integer(
            "bias_decay_warmup_batches", self.bias_decay_warmup_batches
        )
        _attention_settings.validate_mask(validation, self)


@dataclass(frozen=True)
class RuntimeOptions:
    residual_coefficient_activation: ActivationOptions = field(
        default=ActivationOptions.SIGMOID, kw_only=True
    )
    encoder_output_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    decoder_output_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    batch_size: int = 64
    learning_rate: float = 1.0
    vocab_size: int = 8192
    model_dim: int = 128
    source_sequence_length: int = 64
    target_sequence_length: int = 64
    dropout_probability: float = 0.1
    residual_stack_independent_flag: bool = False
    residual_stack_hidden_dim: int | None = None
    residual_stack_layer_norm_position: LayerNormPositionOptions | None = None
    residual_stack_normalization: NormalizationOptions | None = field(
        default=None, kw_only=True
    )
    residual_stack_num_layers: int | None = None
    residual_stack_activation: ActivationOptions | None = None
    residual_stack_residual_connection_option: type[ResidualConfig] | None = None
    residual_stack_residual_model_flag: bool = False
    residual_stack_residual_block_size: int | None = field(default=None, kw_only=True)
    residual_stack_residual_rms_norm_epsilon: float | None = field(
        default=None, kw_only=True
    )
    residual_stack_dropout_probability: float | None = None
    residual_stack_last_layer_bias_option: LastLayerBiasOptions | None = None
    residual_stack_apply_output_postprocessing_flag: bool | None = None
    residual_stack_bias_flag: bool | None = None
    positional_embedding_option: type = TextSinusoidalPositionalEmbeddingConfig
    encoder_options: TransformerStackOptions = TransformerStackOptions()
    decoder_options: TransformerStackOptions = TransformerStackOptions()
    encoder_attention_options: TransformerAttentionOptions = (
        TransformerAttentionOptions()
    )
    decoder_self_attention_options: TransformerAttentionOptions = (
        TransformerAttentionOptions()
    )
    decoder_cross_attention_options: TransformerAttentionOptions = (
        TransformerAttentionOptions()
    )
    encoder_feed_forward_options: TransformerFeedForwardOptions = (
        TransformerFeedForwardOptions()
    )
    decoder_feed_forward_options: TransformerFeedForwardOptions = (
        TransformerFeedForwardOptions()
    )
    projection_adaptive_options: AdaptiveParameterOptions = AdaptiveParameterOptions()
    feed_forward_adaptive_options: AdaptiveParameterOptions = AdaptiveParameterOptions()
    encoder_attention_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )
    decoder_self_attention_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )
    decoder_cross_attention_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )
    encoder_feed_forward_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )
    decoder_feed_forward_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )

    def __post_init__(self) -> None:
        validation = _attention_settings.integer_compatible_validation(__package__)
        validation.positive_integer("batch_size", self.batch_size)
        _attention_settings.validate_dimensions(
            validation, {"learning_rate": self.learning_rate}
        )
        validation.positive_integer("vocab_size", self.vocab_size)
        validation.positive_integer("model_dim", self.model_dim)
        validation.positive_integer(
            "source_sequence_length", self.source_sequence_length
        )
        validation.positive_integer(
            "target_sequence_length", self.target_sequence_length
        )


__all__ = [
    "AdaptiveParameterOptions",
    "ControllerStackOptions",
    "DynamicMemoryOptions",
    "LayerControllerOptions",
    "RecurrentControllerOptions",
    "RuntimeOptions",
    "SubmoduleStackOptions",
    "TransformerAttentionOptions",
    "TransformerFeedForwardOptions",
    "TransformerStackOptions",
]


class PresetValues(TypedDict, total=False):
    attn_add_key_value_bias_flag: bool
    attn_bias_flag: bool
    decoder_layer_norm_position: LayerNormPositionOptions
    encoder_layer_norm_position: LayerNormPositionOptions
    feed_forward_adaptive_bias_option: type[DynamicBiasConfig] | None
    feed_forward_adaptive_diagonal_option: type[DynamicDiagonalConfig] | None
    feed_forward_adaptive_row_mask_option: type[AxisMaskConfig] | None
    halting_threshold: float | None
    memory_flag: bool
    positional_embedding_option: type[AbsolutePositionalEmbeddingConfig]
    projection_adaptive_bias_option: type[DynamicBiasConfig] | None
    projection_adaptive_diagonal_option: type[DynamicDiagonalConfig] | None
    projection_adaptive_row_mask_option: type[AxisMaskConfig] | None
    projection_adaptive_weight_option: type[DynamicWeightConfig] | None
    recurrent_flag: bool
    recurrent_halting_threshold: float | None
    recurrent_residual_connection_option: type[ResidualConfig] | None
    recurrent_stack_gate_flag: bool
    recurrent_stack_halting_flag: bool
    stack_gate_flag: bool
    stack_halting_flag: bool
    stack_residual_connection_option: type[ResidualConfig] | None
