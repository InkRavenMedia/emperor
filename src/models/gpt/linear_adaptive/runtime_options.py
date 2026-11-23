from __future__ import annotations

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
from emperor.experts import SoftMoEConfig
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
from models._recurrent_compression import CompressionOptions
from models._stack_options import resolve_controller_options
from models.gpt.linear_adaptive import _generation
from models.gpt.linear_adaptive._generation import (
    BiasGenerationOptions,
    WeightGenerationOptions,
)
from models.gpt.linear_adaptive._residual import ResidualStackOptions
from models.gpt.linear_adaptive._residual import inherit_if_unset as inherit_if_unset


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
class GptEmbeddingOptions:
    hierarchical_language_model_flag: bool = field(default=False, kw_only=True)
    byte_embedding_dim: int = field(default=32, kw_only=True)
    byte_encoder_num_layers: int = field(default=1, kw_only=True)
    byte_decoder_num_layers: int = field(default=1, kw_only=True)
    byte_num_heads: int = field(default=4, kw_only=True)
    byte_feed_forward_dim: int = field(default=64, kw_only=True)
    byte_limit: int = field(default=64, kw_only=True)
    layer_norm_flag: bool
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    dropout_probability: float

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.dropout("dropout_probability", self.dropout_probability)


@dataclass(frozen=True)
class GptLmHeadOptions:
    weight_tying_flag: bool
    bias_flag: bool


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


@dataclass(frozen=True)
class TransformerDecoderOptions(_sequence_expansion.SequenceExpansionOptions):
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
class _ConstructionOptions:
    attention_grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    feed_forward_grouping_config: GroupingConfig | None = field(
        default=None, kw_only=True
    )
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    batch_size: int
    learning_rate: float
    input_dim: int
    output_dim: int
    sequence_length: int
    embedding_options: GptEmbeddingOptions | None
    decoder_options: TransformerDecoderOptions | None
    positional_embedding_options: TransformerPositionalEmbeddingOptions | None
    attention_options: TransformerAttentionOptions | None
    feed_forward_options: TransformerFeedForwardOptions | None
    lm_head_options: GptLmHeadOptions | None
    attention_projection_stack_options: SubmoduleStackOptions | None
    attention_projection_layer_controller_options: LayerControllerOptions | None
    attention_projection_dynamic_memory_options: DynamicMemoryOptions | None
    attention_projection_recurrent_controller_options: RecurrentControllerOptions | None
    feed_forward_stack_options: SubmoduleStackOptions | None
    feed_forward_layer_controller_options: LayerControllerOptions | None
    feed_forward_dynamic_memory_options: DynamicMemoryOptions | None
    feed_forward_recurrent_controller_options: RecurrentControllerOptions | None
    stack_options: MainLayerStackOptions | None
    submodule_stack_options: SubmoduleStackOptions | None
    layer_controller_options: LayerControllerOptions | None
    dynamic_memory_options: DynamicMemoryOptions | None
    recurrent_controller_options: RecurrentControllerOptions | None
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None
    attention_adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None
    attention_hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    attention_hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    attention_hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    attention_hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None
    feed_forward_adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None
    feed_forward_hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    feed_forward_hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    feed_forward_hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    feed_forward_hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None


@dataclass(frozen=True, slots=True)
class RuntimeOptions(ResolvedRuntimeOptions, CompressionOptions):
    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        ResolvedRuntimeOptions.__post_init__(self)
        _attention_settings.validate_dimensions(validation, self._values)

    def _construction_options(
        self,
        config_module: ModuleType,
    ) -> _ConstructionOptions:
        values = self._values
        return _ConstructionOptions(
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
                GptEmbeddingOptions | None,
                values.get("embedding_options"),
            ),
            decoder_options=cast(
                TransformerDecoderOptions | None,
                values.get("decoder_options"),
            ),
            positional_embedding_options=cast(
                TransformerPositionalEmbeddingOptions | None,
                values.get("positional_embedding_options"),
            ),
            attention_options=cast(
                TransformerAttentionOptions | None,
                values.get("attention_options"),
            ),
            feed_forward_options=cast(
                TransformerFeedForwardOptions | None,
                values.get("feed_forward_options"),
            ),
            lm_head_options=cast(
                GptLmHeadOptions | None,
                values.get("lm_head_options"),
            ),
            attention_projection_stack_options=cast(
                SubmoduleStackOptions | None,
                values.get("attention_projection_stack_options"),
            ),
            attention_projection_layer_controller_options=cast(
                LayerControllerOptions | None,
                values.get("attention_projection_layer_controller_options"),
            ),
            attention_projection_dynamic_memory_options=cast(
                DynamicMemoryOptions | None,
                values.get("attention_projection_dynamic_memory_options"),
            ),
            attention_projection_recurrent_controller_options=cast(
                RecurrentControllerOptions | None,
                values.get("attention_projection_recurrent_controller_options"),
            ),
            feed_forward_stack_options=cast(
                SubmoduleStackOptions | None,
                values.get("feed_forward_stack_options"),
            ),
            feed_forward_layer_controller_options=cast(
                LayerControllerOptions | None,
                values.get("feed_forward_layer_controller_options"),
            ),
            feed_forward_dynamic_memory_options=cast(
                DynamicMemoryOptions | None,
                values.get("feed_forward_dynamic_memory_options"),
            ),
            feed_forward_recurrent_controller_options=cast(
                RecurrentControllerOptions | None,
                values.get("feed_forward_recurrent_controller_options"),
            ),
            stack_options=cast(
                MainLayerStackOptions | None,
                values.get("stack_options"),
            ),
            submodule_stack_options=cast(
                SubmoduleStackOptions | None,
                values.get("submodule_stack_options"),
            ),
            layer_controller_options=cast(
                LayerControllerOptions | None,
                values.get("layer_controller_options"),
            ),
            dynamic_memory_options=cast(
                DynamicMemoryOptions | None,
                values.get("dynamic_memory_options"),
            ),
            recurrent_controller_options=cast(
                RecurrentControllerOptions | None,
                values.get("recurrent_controller_options"),
            ),
            adaptive_generator_stack_options=cast(
                AdaptiveGeneratorStackOptions | None,
                values.get("adaptive_generator_stack_options"),
            ),
            grouping_config=cast(GroupingConfig | None, values.get("grouping_config")),
            hidden_adaptive_weight_options=cast(
                HiddenAdaptiveWeightOptions | None,
                values.get("hidden_adaptive_weight_options"),
            ),
            hidden_adaptive_bias_options=cast(
                HiddenAdaptiveBiasOptions | None,
                values.get("hidden_adaptive_bias_options"),
            ),
            hidden_adaptive_diagonal_options=cast(
                HiddenAdaptiveDiagonalOptions | None,
                values.get("hidden_adaptive_diagonal_options"),
            ),
            hidden_adaptive_mask_options=cast(
                HiddenAdaptiveMaskOptions | None,
                values.get("hidden_adaptive_mask_options"),
            ),
            attention_adaptive_generator_stack_options=cast(
                AdaptiveGeneratorStackOptions | None,
                values.get("attention_adaptive_generator_stack_options"),
            ),
            attention_grouping_config=cast(
                GroupingConfig | None,
                values.get("attention_grouping_config"),
            ),
            attention_hidden_adaptive_weight_options=cast(
                HiddenAdaptiveWeightOptions | None,
                values.get("attention_hidden_adaptive_weight_options"),
            ),
            attention_hidden_adaptive_bias_options=cast(
                HiddenAdaptiveBiasOptions | None,
                values.get("attention_hidden_adaptive_bias_options"),
            ),
            attention_hidden_adaptive_diagonal_options=cast(
                HiddenAdaptiveDiagonalOptions | None,
                values.get("attention_hidden_adaptive_diagonal_options"),
            ),
            attention_hidden_adaptive_mask_options=cast(
                HiddenAdaptiveMaskOptions | None,
                values.get("attention_hidden_adaptive_mask_options"),
            ),
            feed_forward_adaptive_generator_stack_options=cast(
                AdaptiveGeneratorStackOptions | None,
                values.get("feed_forward_adaptive_generator_stack_options"),
            ),
            feed_forward_grouping_config=cast(
                GroupingConfig | None,
                values.get("feed_forward_grouping_config"),
            ),
            feed_forward_hidden_adaptive_weight_options=cast(
                HiddenAdaptiveWeightOptions | None,
                values.get("feed_forward_hidden_adaptive_weight_options"),
            ),
            feed_forward_hidden_adaptive_bias_options=cast(
                HiddenAdaptiveBiasOptions | None,
                values.get("feed_forward_hidden_adaptive_bias_options"),
            ),
            feed_forward_hidden_adaptive_diagonal_options=cast(
                HiddenAdaptiveDiagonalOptions | None,
                values.get("feed_forward_hidden_adaptive_diagonal_options"),
            ),
            feed_forward_hidden_adaptive_mask_options=cast(
                HiddenAdaptiveMaskOptions | None,
                values.get("feed_forward_hidden_adaptive_mask_options"),
            ),
        )


class PresetValues(TypedDict, total=False):
    attn_add_key_value_bias_flag: bool
    attn_bias_flag: bool
    bias_option: type[DynamicBiasConfig] | None
    diagonal_option: type[DynamicDiagonalConfig] | None
    generator_depth: int
    hierarchical_language_model_flag: bool
    layer_norm_position: LayerNormPositionOptions
    lm_head_weight_tying_flag: bool
    memory_flag: bool
    positional_embedding_option: type[AbsolutePositionalEmbeddingConfig]
    recurrent_flag: bool
    recurrent_stack_gate_flag: bool
    recurrent_stack_halting_flag: bool
    row_mask_option: type[AxisMaskConfig] | None
    stack_gate_flag: bool
    stack_halting_flag: bool
    stack_residual_connection_option: type[ResidualConfig] | None
    weight_decay_rate: float
    weight_decay_schedule: WeightDecayScheduleOptions
    weight_decay_warmup_batches: int
    weight_normalization_option: WeightNormalizationOptions
    weight_option: type[DynamicWeightConfig] | None
