from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import TypedDict, cast

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
from models._stack_options import (
    resolve_controller_options,
    resolve_expert_controller,
    resolve_expert_options,
)
from models.bert.expert_linear._residual import ResidualStackOptions
from models.bert.expert_linear._residual import inherit_if_unset as inherit_if_unset


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
class BertEmbeddingOptions:
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


@dataclass(frozen=True)
class BertMlmHeadOptions:
    activation: ActivationOptions
    dense_bias_flag: bool
    layer_norm_flag: bool
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    decoder_bias_flag: bool
    decoder_weight_tying_flag: bool


@dataclass(frozen=True)
class BertNspHeadOptions:
    pooler_activation: ActivationOptions
    pooler_bias_flag: bool
    output_dim: int
    head_bias_flag: bool

    def __post_init__(self) -> None:
        validation = _attention_settings.strict_float_validation(__package__)
        validation.positive_integer("nsp_output_dim", self.output_dim)


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


@dataclass(frozen=True, slots=True)
class _ConstructionOptions:
    batch_size: int
    learning_rate: float
    input_dim: int
    output_dim: int
    sequence_length: int
    embedding_options: BertEmbeddingOptions | None
    encoder_options: TransformerEncoderOptions | None
    positional_embedding_options: TransformerPositionalEmbeddingOptions | None
    attention_options: TransformerAttentionOptions | None
    feed_forward_options: TransformerFeedForwardOptions | None
    mlm_head_options: BertMlmHeadOptions | None
    nsp_head_options: BertNspHeadOptions | None
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
    mixture_options: ExpertsMixtureOptions | None
    expert_stack_options: ExpertsSubmoduleStackOptions | None
    sampler_options: ExpertsSamplerOptions | None
    router_options: ExpertsRouterOptions | None
    router_stack_options: ExpertsSubmoduleStackOptions | None
    expert_layer_controller_options: ExpertsLayerControllerOptions | None
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_attention_use_kv_expert_models_flag: bool


@dataclass(frozen=True, slots=True)
class RuntimeOptions(ResolvedRuntimeOptions):
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
                BertEmbeddingOptions | None,
                values.get("embedding_options"),
            ),
            encoder_options=cast(
                TransformerEncoderOptions | None,
                values.get("encoder_options"),
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
            mlm_head_options=cast(
                BertMlmHeadOptions | None,
                values.get("mlm_head_options"),
            ),
            nsp_head_options=cast(
                BertNspHeadOptions | None,
                values.get("nsp_head_options"),
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
            mixture_options=cast(
                ExpertsMixtureOptions | None,
                values.get("mixture_options"),
            ),
            expert_stack_options=cast(
                ExpertsSubmoduleStackOptions | None,
                values.get("expert_stack_options"),
            ),
            sampler_options=cast(
                ExpertsSamplerOptions | None,
                values.get("sampler_options"),
            ),
            router_options=cast(
                ExpertsRouterOptions | None,
                values.get("router_options"),
            ),
            router_stack_options=cast(
                ExpertsSubmoduleStackOptions | None,
                values.get("router_stack_options"),
            ),
            expert_layer_controller_options=cast(
                ExpertsLayerControllerOptions | None,
                values.get("expert_layer_controller_options"),
            ),
            expert_dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions | None,
                values.get("expert_dynamic_memory_options"),
            ),
            expert_recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions | None,
                values.get("expert_recurrent_controller_options"),
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
    causal_attention_mask_flag: bool
    layer_norm_position: LayerNormPositionOptions
    memory_flag: bool
    positional_embedding_option: type[AbsolutePositionalEmbeddingConfig]
    recurrent_flag: bool
    recurrent_stack_gate_flag: bool
    recurrent_stack_halting_flag: bool
    sampler_normalize_probabilities_flag: bool
    sampler_switch_loss_weight: float
    stack_gate_flag: bool
    stack_halting_flag: bool
    top_k: int
