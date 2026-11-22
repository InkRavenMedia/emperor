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
from models._stack_options import resolve_controller_options
from models.vit.linear_adaptive import _generation
from models.vit.linear_adaptive._generation import (
    BiasGenerationOptions,
    WeightGenerationOptions,
)
from models.vit.linear_adaptive._residual import ResidualStackOptions
from models.vit.linear_adaptive._residual import inherit_if_unset as inherit_if_unset


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
class RuntimeOptions(ResolvedRuntimeOptions):
    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        ResolvedRuntimeOptions.__post_init__(self)
        _attention_settings.validate_dimensions(validation, self._values)


class PresetValues(TypedDict, total=False):
    attn_add_key_value_bias_flag: bool
    attn_bias_flag: bool
    bias_option: type[DynamicBiasConfig] | None
    diagonal_option: type[DynamicDiagonalConfig] | None
    generator_depth: int
    layer_norm_position: LayerNormPositionOptions
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
