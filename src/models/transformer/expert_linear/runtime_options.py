from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from emperor.embedding.absolute import (
    AbsolutePositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
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
from emperor.memory import (
    DynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
)
from models import _attention_settings, _sequence_expansion
from models import _attention_source_memory as _source_memory
from models._recurrent_compression import CompressionOptions
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
class ExpertOptions:
    use_kv_expert_models_flag: bool | None = None
    num_experts: int = 4
    top_k: int = 2
    dropped_token_behavior: DroppedTokenOptions = DroppedTokenOptions.ZEROS
    compute_expert_mixture_flag: bool = True
    weighted_parameters_flag: bool = False
    weighting_position_option: ExpertWeightingPositionOptions = (
        ExpertWeightingPositionOptions.BEFORE_EXPERTS
    )
    routing_initialization_mode: RoutingInitializationMode = (
        RoutingInitializationMode.LAYER
    )
    sampler_threshold: float = 0.0
    sampler_filter_above_threshold: bool = False
    sampler_num_topk_samples: int = 0
    normalize_probabilities_flag: bool = True
    sampler_noisy_topk_flag: bool = False
    coefficient_of_variation_loss_weight: float = 0.0
    switch_loss_weight: float = 0.0
    zero_centred_loss_weight: float = 0.0
    mutual_information_loss_weight: float = 0.0
    capacity_factor: float = 0.0
    router_noisy_topk_flag: bool = False
    router_path_options: TransformerFeedForwardOptions = TransformerFeedForwardOptions(
        stack_options=SubmoduleStackOptions(
            hidden_dim=128,
            num_layers=2,
            activation=ActivationOptions.GELU,
        )
    )
    expert_path_options: TransformerFeedForwardOptions = TransformerFeedForwardOptions(
        stack_options=SubmoduleStackOptions(
            hidden_dim=128,
            num_layers=1,
            activation=ActivationOptions.RELU,
        )
    )


@dataclass(frozen=True)
class RuntimeOptions(CompressionOptions):
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
    attention_expert_options: ExpertOptions = ExpertOptions()
    feed_forward_expert_options: ExpertOptions = ExpertOptions()

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
    "ControllerStackOptions",
    "DynamicMemoryOptions",
    "ExpertOptions",
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
    halting_threshold: float | None
    memory_flag: bool
    normalize_probabilities_flag: bool
    positional_embedding_option: type[AbsolutePositionalEmbeddingConfig]
    recurrent_flag: bool
    recurrent_halting_threshold: float | None
    recurrent_residual_connection_option: type[ResidualConfig] | None
    recurrent_stack_gate_flag: bool
    recurrent_stack_halting_flag: bool
    stack_gate_flag: bool
    stack_halting_flag: bool
    stack_residual_connection_option: type[ResidualConfig] | None
    switch_loss_weight: float
    top_k: int
