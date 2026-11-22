from dataclasses import dataclass, field
from types import ModuleType
from typing import TypedDict, cast

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
from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.embedding.contextual import ByteContextualEmbeddingConfig
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
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
from model_runtime.packages.runtime_values import ResolvedRuntimeOptions
from models import _attention_settings, _sequence_expansion
from models import _attention_source_memory as _source_memory
from models._stack_options import resolve_expert_controller, resolve_expert_options
from models.bert.expert_linear_adaptive import _generation
from models.bert.expert_linear_adaptive._generation import (
    BiasGenerationOptions,
    WeightGenerationOptions,
)
from models.bert.expert_linear_adaptive._residual import (
    ResidualStackOptions,
)
from models.bert.expert_linear_adaptive._residual import (
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
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float
    bias_flag: bool

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("hidden_dim", self.hidden_dim)
        validation.stack(self)


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
        validation = _attention_settings.strict_float_validation(__package__)
        _attention_settings.validate_controller(validation, self)


@dataclass(frozen=True, slots=True)
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
    recurrent_residual_connection_option: type[ResidualConfig] | None = field(
        default=None,
        kw_only=True,
    )
    recurrent_residual_model_flag: bool = field(default=False, kw_only=True)
    recurrent_residual_block_size: int | None = field(default=None, kw_only=True)
    recurrent_residual_rms_norm_epsilon: float | None = field(
        default=None, kw_only=True
    )
    residual_stack_options: ResidualStackOptions | None = field(
        default=None,
        kw_only=True,
    )
    recurrent_stack_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_gate_stack_source: ExpertsSubmoduleStackSource
    recurrent_min_steps: int = field(default=1, kw_only=True)
    recurrent_ponder_cost_weight: float = field(default=1.0, kw_only=True)
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
        _attention_settings.validate_residual(
            "recurrent_residual_connection_option",
            self.recurrent_residual_connection_option,
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
        validation = _attention_settings.strict_float_validation(__package__)
        if self.option_flag and self.option is None:
            raise ValueError("bias_option must be set when bias_option_flag is True")
        validation.nonnegative_float("decay_rate", self.decay_rate)
        validation.nonnegative_integer(
            "decay_warmup_batches", self.decay_warmup_batches
        )


@dataclass(frozen=True, slots=True)
class HiddenAdaptiveDiagonalOptions:
    option_flag: bool
    option: type[DynamicDiagonalConfig] | None
    generator_stack_source: _generation.AdaptiveGeneratorStackSource

    def __post_init__(self) -> None:
        if self.option_flag and self.option is None:
            raise ValueError(
                "diagonal_option must be set when diagonal_option_flag is True"
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
        validation = _attention_settings.strict_float_validation(__package__)
        if self.option_flag and self.row_mask_option is None:
            raise ValueError(
                "row_mask_option must be set when mask_option_flag is True"
            )
        _attention_settings.validate_mask(validation, self)


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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
    residual_connection_option: type[ResidualConfig]
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class RecurrentControllerOptions:
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class TransformerPositionalEmbeddingOptions:
    option: type[AbsolutePositionalEmbeddingConfig]
    padding_idx: int | None
    auto_expand_flag: bool


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class TransformerFeedForwardOptions:
    soft_moe_config: SoftMoEConfig | None = field(default=None, kw_only=True)
    num_layers: int
    bias_flag: bool

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("num_layers", self.num_layers)


@dataclass(frozen=True, slots=True)
class BertEmbeddingOptions:
    hierarchical_masked_word_flag: bool = field(default=False, kw_only=True)
    contextual_embedding_config: ByteContextualEmbeddingConfig | None = field(
        default=None, kw_only=True
    )
    hierarchical_embedding_config: HierarchicalByteEmbeddingConfig | None = field(
        default=None, kw_only=True
    )
    token_type_vocab_size: int
    layer_norm_flag: bool
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    dropout_probability: float

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("token_type_vocab_size", self.token_type_vocab_size)
        validation.dropout("dropout_probability", self.dropout_probability)


@dataclass(frozen=True, slots=True)
class BertMlmHeadOptions:
    activation: ActivationOptions
    dense_bias_flag: bool
    layer_norm_flag: bool
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    decoder_bias_flag: bool
    decoder_weight_tying_flag: bool


@dataclass(frozen=True, slots=True)
class BertNspHeadOptions:
    pooler_activation: ActivationOptions
    pooler_bias_flag: bool
    output_dim: int
    head_bias_flag: bool

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("nsp_output_dim", self.output_dim)


@dataclass(frozen=True, slots=True)
class ConstructionOptions:
    attention_grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    feed_forward_grouping_config: GroupingConfig | None = field(
        default=None, kw_only=True
    )
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    router_grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    batch_size: int
    learning_rate: float
    input_dim: int
    output_dim: int
    sequence_length: int
    embedding_options: BertEmbeddingOptions
    mlm_head_options: BertMlmHeadOptions
    nsp_head_options: BertNspHeadOptions
    encoder_options: TransformerEncoderOptions
    positional_embedding_options: TransformerPositionalEmbeddingOptions
    attention_options: TransformerAttentionOptions
    feed_forward_options: TransformerFeedForwardOptions
    attention_projection_stack_options: ExpertsSubmoduleStackOptions | None
    attention_projection_layer_controller_options: ExpertsLayerControllerOptions | None
    attention_projection_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    attention_projection_recurrent_controller_options: (
        ExpertsRecurrentControllerOptions | None
    )
    feed_forward_stack_options: ExpertsSubmoduleStackOptions | None
    feed_forward_layer_controller_options: ExpertsLayerControllerOptions | None
    feed_forward_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    feed_forward_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    submodule_stack_options: ExpertsSubmoduleStackOptions
    layer_controller_options: ExpertsLayerControllerOptions
    dynamic_memory_options: ExpertsDynamicMemoryOptions
    recurrent_controller_options: ExpertsRecurrentControllerOptions
    mixture_options: ExpertsMixtureOptions
    mixture_submodule_stack_options: ExpertsSubmoduleStackOptions
    mixture_layer_controller_options: ExpertsLayerControllerOptions
    mixture_dynamic_memory_options: ExpertsDynamicMemoryOptions
    mixture_recurrent_controller_options: ExpertsRecurrentControllerOptions
    expert_stack_options: ExpertsSubmoduleStackOptions
    sampler_options: ExpertsSamplerOptions
    router_options: ExpertsRouterOptions
    router_stack_options: ExpertsSubmoduleStackOptions
    router_layer_controller_options: ExpertsLayerControllerOptions
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions
    expert_layer_controller_options: ExpertsLayerControllerOptions
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions
    expert_attention_use_kv_expert_models_flag: bool


@dataclass(frozen=True, slots=True)
class RuntimeOptions(ResolvedRuntimeOptions):
    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        ResolvedRuntimeOptions.__post_init__(self)
        _attention_settings.validate_dimensions(validation, self._values)

    def construction_options(
        self,
        config_module: ModuleType,
        *,
        group_defaults: "RuntimeOptions",
    ) -> ConstructionOptions:
        values = self._values

        def option_group(name: str) -> object:
            selected = values.get(name) or group_defaults._values[name]
            if selected is None:
                raise ValueError(f"Missing construction defaults for {name}")
            return selected

        return ConstructionOptions(
            attention_grouping_config=cast(
                GroupingConfig | None, values.get("attention_grouping_config")
            ),
            feed_forward_grouping_config=cast(
                GroupingConfig | None, values.get("feed_forward_grouping_config")
            ),
            batch_size=cast(int, values.get("batch_size", config_module.BATCH_SIZE)),
            learning_rate=cast(
                float,
                values.get("learning_rate", config_module.LEARNING_RATE),
            ),
            input_dim=cast(int, values.get("input_dim", config_module.INPUT_DIM)),
            output_dim=cast(int, values.get("output_dim", config_module.OUTPUT_DIM)),
            sequence_length=cast(
                int,
                values.get("sequence_length", config_module.SEQUENCE_LENGTH),
            ),
            embedding_options=cast(
                BertEmbeddingOptions,
                option_group("embedding_options"),
            ),
            mlm_head_options=cast(
                BertMlmHeadOptions,
                option_group("mlm_head_options"),
            ),
            nsp_head_options=cast(
                BertNspHeadOptions,
                option_group("nsp_head_options"),
            ),
            encoder_options=cast(
                TransformerEncoderOptions,
                option_group("encoder_options"),
            ),
            positional_embedding_options=cast(
                TransformerPositionalEmbeddingOptions,
                option_group("positional_embedding_options"),
            ),
            attention_options=cast(
                TransformerAttentionOptions,
                option_group("attention_options"),
            ),
            feed_forward_options=cast(
                TransformerFeedForwardOptions,
                option_group("feed_forward_options"),
            ),
            attention_projection_stack_options=cast(
                ExpertsSubmoduleStackOptions,
                option_group("attention_projection_stack_options"),
            ),
            attention_projection_layer_controller_options=cast(
                ExpertsLayerControllerOptions,
                option_group("attention_projection_layer_controller_options"),
            ),
            attention_projection_dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions,
                option_group("attention_projection_dynamic_memory_options"),
            ),
            attention_projection_recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions,
                option_group("attention_projection_recurrent_controller_options"),
            ),
            feed_forward_stack_options=cast(
                ExpertsSubmoduleStackOptions,
                option_group("feed_forward_stack_options"),
            ),
            feed_forward_layer_controller_options=cast(
                ExpertsLayerControllerOptions,
                option_group("feed_forward_layer_controller_options"),
            ),
            feed_forward_dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions,
                option_group("feed_forward_dynamic_memory_options"),
            ),
            feed_forward_recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions,
                option_group("feed_forward_recurrent_controller_options"),
            ),
            submodule_stack_options=cast(
                ExpertsSubmoduleStackOptions,
                option_group("submodule_stack_options"),
            ),
            layer_controller_options=cast(
                ExpertsLayerControllerOptions,
                option_group("layer_controller_options"),
            ),
            dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions,
                option_group("dynamic_memory_options"),
            ),
            recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions,
                option_group("recurrent_controller_options"),
            ),
            mixture_options=cast(
                ExpertsMixtureOptions,
                option_group("mixture_options"),
            ),
            mixture_submodule_stack_options=cast(
                ExpertsSubmoduleStackOptions,
                option_group("mixture_submodule_stack_options"),
            ),
            mixture_layer_controller_options=cast(
                ExpertsLayerControllerOptions,
                option_group("mixture_layer_controller_options"),
            ),
            mixture_dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions,
                option_group("mixture_dynamic_memory_options"),
            ),
            mixture_recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions,
                option_group("mixture_recurrent_controller_options"),
            ),
            expert_stack_options=cast(
                ExpertsSubmoduleStackOptions,
                option_group("expert_stack_options"),
            ),
            sampler_options=cast(
                ExpertsSamplerOptions,
                option_group("sampler_options"),
            ),
            router_options=cast(
                ExpertsRouterOptions,
                option_group("router_options"),
            ),
            router_stack_options=cast(
                ExpertsSubmoduleStackOptions,
                option_group("router_stack_options"),
            ),
            router_layer_controller_options=cast(
                ExpertsLayerControllerOptions,
                option_group("router_layer_controller_options"),
            ),
            router_dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions,
                option_group("router_dynamic_memory_options"),
            ),
            router_recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions,
                option_group("router_recurrent_controller_options"),
            ),
            expert_layer_controller_options=cast(
                ExpertsLayerControllerOptions,
                option_group("expert_layer_controller_options"),
            ),
            expert_dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions,
                option_group("expert_dynamic_memory_options"),
            ),
            expert_recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions,
                option_group("expert_recurrent_controller_options"),
            ),
            adaptive_generator_stack_options=cast(
                AdaptiveGeneratorStackOptions,
                option_group("adaptive_generator_stack_options"),
            ),
            grouping_config=cast(GroupingConfig | None, values.get("grouping_config")),
            hidden_adaptive_weight_options=cast(
                HiddenAdaptiveWeightOptions,
                option_group("hidden_adaptive_weight_options"),
            ),
            hidden_adaptive_bias_options=cast(
                HiddenAdaptiveBiasOptions,
                option_group("hidden_adaptive_bias_options"),
            ),
            hidden_adaptive_diagonal_options=cast(
                HiddenAdaptiveDiagonalOptions,
                option_group("hidden_adaptive_diagonal_options"),
            ),
            hidden_adaptive_mask_options=cast(
                HiddenAdaptiveMaskOptions,
                option_group("hidden_adaptive_mask_options"),
            ),
            router_grouping_config=cast(
                GroupingConfig | None, values.get("router_grouping_config")
            ),
            router_adaptive_weight_options=cast(
                HiddenAdaptiveWeightOptions,
                option_group("router_adaptive_weight_options"),
            ),
            router_adaptive_bias_options=cast(
                HiddenAdaptiveBiasOptions,
                option_group("router_adaptive_bias_options"),
            ),
            router_adaptive_diagonal_options=cast(
                HiddenAdaptiveDiagonalOptions,
                option_group("router_adaptive_diagonal_options"),
            ),
            router_adaptive_mask_options=cast(
                HiddenAdaptiveMaskOptions,
                option_group("router_adaptive_mask_options"),
            ),
            expert_attention_use_kv_expert_models_flag=cast(
                bool,
                values.get(
                    "expert_attention_use_kv_expert_models_flag",
                    config_module.EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG,
                ),
            ),
        )


class PresetValues(TypedDict, total=False):
    attn_add_key_value_bias_flag: bool
    attn_bias_flag: bool
    attn_num_heads: int
    bias_option: type[DynamicBiasConfig] | None
    bias_option_flag: bool
    capacity_factor: float
    causal_attention_mask_flag: bool
    diagonal_option: type[DynamicDiagonalConfig] | None
    diagonal_option_flag: bool
    expert_attention_use_kv_expert_models_flag: bool
    expert_bias_flag: bool
    ff_stack_hidden_dim: int
    generator_depth: int
    hidden_dim: int
    layer_norm_position: LayerNormPositionOptions
    mask_option_flag: bool
    memory_flag: bool
    num_experts: int
    positional_embedding_option: type[AbsolutePositionalEmbeddingConfig]
    recurrent_flag: bool
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    recurrent_halting_option: type[HaltingConfig]
    recurrent_halting_threshold: float
    recurrent_max_steps: int
    recurrent_residual_connection_option: type[ResidualConfig] | None
    recurrent_stack_gate_flag: bool
    recurrent_stack_halting_flag: bool
    sampler_normalize_probabilities_flag: bool
    sampler_switch_loss_weight: float
    sampler_zero_centred_loss_weight: float
    sequence_length: int
    stack_dropout_probability: float
    stack_gate_flag: bool
    stack_halting_flag: bool
    stack_num_layers: int
    top_k: int
    weight_decay_rate: float
    weight_decay_schedule: WeightDecayScheduleOptions
    weight_decay_warmup_batches: int
    weight_normalization_option: WeightNormalizationOptions
    weight_normalization_position_option: WeightNormalizationPositionOptions
    weight_option: type[DynamicWeightConfig] | None
    weight_option_flag: bool
