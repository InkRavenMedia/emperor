from typing import Annotated

from emperor.attention import AttentionSourceMemoryConfig  # noqa: F401
from emperor.augmentations.adaptive_parameters import (  # noqa: F401
    AdaptiveLinearLayerConfig,  # noqa: F401
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    AdditiveDynamicBiasConfig,
    AttentionGroupingConfig,
    MeanGroupingConfig,
    MeanStdGroupingConfig,
    RMSGroupingConfig,
    SumGroupingConfig,
    SummaryNormalizationOptions,
    WeightDecayScheduleOptions,
)
from emperor.config import ConfigBase  # noqa: F401
from emperor.embedding.absolute import (
    AbsolutePositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,  # noqa: F401
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    StickBreakingConfig,
)
from emperor.layers import (  # noqa: F401
    ActivationOptions,
    AdditiveResidualConfig,  # noqa: F401
    AttentionResidualConfig,  # noqa: F401
    HierarchicalReasoningModelRecurrentConfig,  # noqa: F401
    LastLayerBiasOptions,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentCompositionConfig,
    RecurrentLayerConfig,  # noqa: F401
    ResidualConfig,
    TinyRecursiveModelRecurrentConfig,  # noqa: F401
    WeightedBlendResidualConfig,  # noqa: F401
    WeightedResidualConfig,  # noqa: F401
)
from emperor.linears import LinearLayerConfig  # noqa: F401
from emperor.memory import (
    AttentionDynamicMemoryConfig,  # noqa: F401
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,  # noqa: F401
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,  # noqa: F401
)
from emperor.transformer import SequenceExpansionConfig  # noqa: F401
from model_runtime.packages.annotations import inherit_alias_annotations
from model_runtime.packages.configuration_metadata import Section
from model_runtime.packages.runtime_values import (
    RuntimeValueConstraints,
    positive_runtime_fields,
)
from models import _attention_source_memory, _component_defaults

_general = Section(("Global",))
BATCH_SIZE: Annotated[int, _general] = 64
LEARNING_RATE: Annotated[float, _general] = 1.0
VOCAB_SIZE: Annotated[int, _general] = 8192
MODEL_DIM: Annotated[int, _general] = 128
RUNTIME_VALUE_CONSTRAINTS: Annotated[RuntimeValueConstraints, _general] = (
    positive_runtime_fields(
        "MODEL_DIM",
        "ENCODER_NUM_LAYERS",
        "DECODER_NUM_LAYERS",
        "ATTN_NUM_HEADS",
        "FF_STACK_HIDDEN_DIM",
        "FF_NUM_LAYERS",
    )
)
SOURCE_SEQUENCE_LENGTH: Annotated[int, _general] = 64
TARGET_SEQUENCE_LENGTH: Annotated[int, _general] = 64
SEQUENCE_LENGTH: Annotated[int, _general] = 64
DROPOUT_PROBABILITY: Annotated[float, _general] = 0.1
POSITIONAL_EMBEDDING_OPTION: Annotated[
    type[AbsolutePositionalEmbeddingConfig], _general
] = TextSinusoidalPositionalEmbeddingConfig

_attention = Section(("Attention Options",))
ATTN_NUM_HEADS: Annotated[int, _attention] = 4
ATTN_ADD_KEY_VALUE_BIAS_FLAG: Annotated[bool, _attention] = False
ATTN_ZERO_ATTENTION_FLAG: Annotated[bool, _attention] = False

_attention_source_memory_section = _attention.child("Attention Source Memory Options")
ATTN_SOURCE_MEMORY_FLAG: Annotated[bool, _attention_source_memory_section] = False
ATTN_SOURCE_MEMORY_CONFIG: Annotated[
    _attention_source_memory.AttentionSourceMemoryConfig | None,
    _attention_source_memory_section,
] = None
ATTN_SOURCE_MEMORY_MODEL_CONFIG: Annotated[
    _attention_source_memory.ConfigBase | None, _attention_source_memory_section
] = None
ATTN_SOURCE_MEMORY_CHUNK_SIZE: Annotated[
    int | None, _attention_source_memory_section
] = None
ATTN_SOURCE_MEMORY_GROUPING_METHOD: Annotated[
    type[_attention_source_memory.GroupingConfig] | None,
    _attention_source_memory_section,
] = None
ATTN_SOURCE_MEMORY_GROUPING_MODEL_CONFIG: Annotated[
    _attention_source_memory.ConfigBase | None, _attention_source_memory_section
] = None
ATTN_SOURCE_MEMORY_SUMMARY_NORMALIZATION: Annotated[
    _attention_source_memory.SummaryNormalizationOptions | None,
    _attention_source_memory_section,
] = None
ATTN_SOURCE_MEMORY_RMS_NORM_EPSILON: Annotated[
    float | None, _attention_source_memory_section
] = None

_attention_projection_stack = _attention.child("Attention Projection Stack Options")
ATTN_NUM_LAYERS: Annotated[int, _attention_projection_stack] = 1
ATTN_BIAS_FLAG: Annotated[bool, _attention_projection_stack] = True
ATTN_STACK_HIDDEN_DIM: Annotated[int, _attention_projection_stack] = MODEL_DIM
ATTN_STACK_ACTIVATION: Annotated[ActivationOptions, _attention_projection_stack] = (
    ActivationOptions.DISABLED
)
ATTN_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attention_projection_stack
] = None
ATTN_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _attention_projection_stack] = False
ATTN_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _attention_projection_stack] = (
    None
)
ATTN_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attention_projection_stack
] = None
ATTN_STACK_DROPOUT_PROBABILITY: Annotated[float, _attention_projection_stack] = 0.0
ATTN_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _attention_projection_stack
] = LayerNormPositionOptions.DISABLED
ATTN_STACK_NORMALIZATION: Annotated[
    NormalizationOptions, _attention_projection_stack
] = NormalizationOptions.RMS_NORM
ATTN_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions, _attention_projection_stack
] = LastLayerBiasOptions.DEFAULT
ATTN_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool, _attention_projection_stack
] = False

_attention_projection_gate = _attention_projection_stack.child(
    "Attention Projection Gate Options"
)
ATTN_STACK_GATE_FLAG: Annotated[bool, _attention_projection_gate] = False
ATTN_GATE_OPTION: Annotated[LayerGateOptions | None, _attention_projection_gate] = (
    LayerGateOptions.MULTIPLIER
)
ATTN_GATE_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_gate
] = ActivationOptions.SIGMOID

_attention_projection_gate_stack = _attention_projection_gate.child(
    "Attention Projection Gate Stack Options"
)
ATTN_GATE_STACK_INDEPENDENT_FLAG: Annotated[bool, _attention_projection_gate_stack] = (
    False
)
ATTN_GATE_STACK_HIDDEN_DIM: Annotated[int | None, _attention_projection_gate_stack] = (
    None
)
ATTN_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attention_projection_gate_stack
] = None
ATTN_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attention_projection_gate_stack
] = None
ATTN_GATE_STACK_NUM_LAYERS: Annotated[int | None, _attention_projection_gate_stack] = (
    None
)
ATTN_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_gate_stack
] = ActivationOptions.TANH
ATTN_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attention_projection_gate_stack
] = None
ATTN_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attention_projection_gate_stack
] = False
ATTN_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attention_projection_gate_stack
] = None
ATTN_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attention_projection_gate_stack
] = None
ATTN_GATE_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attention_projection_gate_stack
] = None
ATTN_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attention_projection_gate_stack
] = None
ATTN_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attention_projection_gate_stack
] = True
ATTN_GATE_STACK_BIAS_FLAG: Annotated[bool | None, _attention_projection_gate_stack] = (
    True
)

_attention_projection_halting = _attention_projection_stack.child(
    "Attention Projection Halting Options"
)
ATTN_STACK_HALTING_FLAG: Annotated[bool, _attention_projection_halting] = False
ATTN_HALTING_OPTION: Annotated[type[HaltingConfig], _attention_projection_halting] = (
    StickBreakingConfig
)
ATTN_HALTING_THRESHOLD: Annotated[float, _attention_projection_halting] = 0.999
ATTN_HALTING_DROPOUT: Annotated[float, _attention_projection_halting] = 0.0
ATTN_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _attention_projection_halting
] = HaltingHiddenStateModeOptions.RAW

_attention_projection_halting_stack = _attention_projection_halting.child(
    "Attention Projection Halting Stack Options"
)
ATTN_HALTING_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attention_projection_halting_stack
] = False
ATTN_HALTING_STACK_HIDDEN_DIM: Annotated[
    int | None, _attention_projection_halting_stack
] = None
ATTN_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attention_projection_halting_stack
] = LayerNormPositionOptions.DISABLED
ATTN_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attention_projection_halting_stack
] = None
ATTN_HALTING_STACK_NUM_LAYERS: Annotated[
    int | None, _attention_projection_halting_stack
] = None
ATTN_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_halting_stack
] = None
ATTN_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attention_projection_halting_stack
] = None
ATTN_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attention_projection_halting_stack
] = False
ATTN_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attention_projection_halting_stack
] = None
ATTN_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attention_projection_halting_stack
] = None
ATTN_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attention_projection_halting_stack
] = None
ATTN_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attention_projection_halting_stack
] = LastLayerBiasOptions.DISABLED
ATTN_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attention_projection_halting_stack
] = None
ATTN_HALTING_STACK_BIAS_FLAG: Annotated[
    bool | None, _attention_projection_halting_stack
] = None

_attention_projection_memory = _attention_projection_stack.child(
    "Attention Projection Memory Options"
)
ATTN_MEMORY_FLAG: Annotated[bool, _attention_projection_memory] = False
ATTN_MEMORY_NUM_MEMORY_SLOTS: Annotated[int, _attention_projection_memory] = 4
ATTN_MEMORY_OPTION: Annotated[
    type[DynamicMemoryConfig], _attention_projection_memory
] = GatedResidualDynamicMemoryConfig
ATTN_MEMORY_POSITION_OPTION: Annotated[
    MemoryPositionOptions, _attention_projection_memory
] = MemoryPositionOptions.AFTER_AFFINE
ATTN_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: Annotated[
    float | None, _attention_projection_memory
] = None
ATTN_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: Annotated[
    int | None, _attention_projection_memory
] = None

_attention_projection_memory_stack = _attention_projection_memory.child(
    "Attention Projection Memory Stack Options"
)
ATTN_MEMORY_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attention_projection_memory_stack
] = False
ATTN_MEMORY_STACK_HIDDEN_DIM: Annotated[
    int | None, _attention_projection_memory_stack
] = None
ATTN_MEMORY_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attention_projection_memory_stack
] = None
ATTN_MEMORY_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attention_projection_memory_stack
] = None
ATTN_MEMORY_STACK_NUM_LAYERS: Annotated[
    int | None, _attention_projection_memory_stack
] = None
ATTN_MEMORY_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_memory_stack
] = None
ATTN_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attention_projection_memory_stack
] = None
ATTN_MEMORY_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attention_projection_memory_stack
] = False
ATTN_MEMORY_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attention_projection_memory_stack
] = None
ATTN_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attention_projection_memory_stack
] = None
ATTN_MEMORY_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attention_projection_memory_stack
] = None
ATTN_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attention_projection_memory_stack
] = None
ATTN_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attention_projection_memory_stack
] = None
ATTN_MEMORY_STACK_BIAS_FLAG: Annotated[
    bool | None, _attention_projection_memory_stack
] = None

_attention_projection_recurrent_layer = _attention_projection_stack.child(
    "Attention Projection Recurrent Layer Options"
)
ATTN_RECURRENT_FLAG: Annotated[bool, _attention_projection_recurrent_layer] = False
ATTN_RECURRENT_COMPOSITION_OPTION: Annotated[
    type[RecurrentCompositionConfig], _attention_projection_recurrent_layer
] = RecurrentLayerConfig
ATTN_RECURRENT_MAX_STEPS: Annotated[int, _attention_projection_recurrent_layer] = 2
ATTN_RECURRENT_NO_GRADIENT_TRANSITION_COUNT: Annotated[
    int | None, _attention_projection_recurrent_layer
] = None
ATTN_RECURRENT_REINJECT_ORIGINAL_HIDDEN_FLAG: Annotated[
    bool, _attention_projection_recurrent_layer
] = False
ATTN_RECURRENT_LATENT_UPDATES_PER_ANSWER_UPDATE: Annotated[
    int, _attention_projection_recurrent_layer
] = 2
ATTN_RECURRENT_ANSWER_UPDATE_COUNT: Annotated[
    int, _attention_projection_recurrent_layer
] = 2
ATTN_RECURRENT_HIGH_CYCLES: Annotated[int, _attention_projection_recurrent_layer] = 2
ATTN_RECURRENT_LOW_CYCLES: Annotated[int, _attention_projection_recurrent_layer] = 2
ATTN_RECURRENT_INITIALIZATION_STANDARD_DEVIATION: Annotated[
    float, _attention_projection_recurrent_layer
] = 1.0
ATTN_RECURRENT_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _attention_projection_recurrent_layer
] = LayerNormPositionOptions.DISABLED
ATTN_RECURRENT_NORMALIZATION: Annotated[
    NormalizationOptions, _attention_projection_recurrent_layer
] = NormalizationOptions.LAYER_NORM

_attention_projection_recurrent_gate = _attention_projection_recurrent_layer.child(
    "Attention Projection Recurrent Gate Options"
)
ATTN_RECURRENT_STACK_GATE_FLAG: Annotated[
    bool, _attention_projection_recurrent_gate
] = False
ATTN_RECURRENT_GATE_OPTION: Annotated[
    LayerGateOptions | None, _attention_projection_recurrent_gate
] = LayerGateOptions.MULTIPLIER
ATTN_RECURRENT_GATE_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_recurrent_gate
] = ActivationOptions.SIGMOID

_attention_projection_recurrent_gate_stack = _attention_projection_recurrent_gate.child(
    "Attention Projection Recurrent Gate Stack Options"
)
ATTN_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attention_projection_recurrent_gate_stack
] = False
ATTN_RECURRENT_GATE_STACK_HIDDEN_DIM: Annotated[
    int | None, _attention_projection_recurrent_gate_stack
] = None
ATTN_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attention_projection_recurrent_gate_stack
] = None
ATTN_RECURRENT_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attention_projection_recurrent_gate_stack
] = None
ATTN_RECURRENT_GATE_STACK_NUM_LAYERS: Annotated[
    int | None, _attention_projection_recurrent_gate_stack
] = None
ATTN_RECURRENT_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_recurrent_gate_stack
] = None
ATTN_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attention_projection_recurrent_gate_stack
] = None
ATTN_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attention_projection_recurrent_gate_stack
] = False
ATTN_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attention_projection_recurrent_gate_stack
] = None
ATTN_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attention_projection_recurrent_gate_stack
] = None
ATTN_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attention_projection_recurrent_gate_stack
] = None
ATTN_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attention_projection_recurrent_gate_stack
] = None
ATTN_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attention_projection_recurrent_gate_stack
] = None
ATTN_RECURRENT_GATE_STACK_BIAS_FLAG: Annotated[
    bool | None, _attention_projection_recurrent_gate_stack
] = None

_attention_projection_recurrent_halting = _attention_projection_recurrent_layer.child(
    "Attention Projection Recurrent Halting Options"
)
ATTN_RECURRENT_STACK_HALTING_FLAG: Annotated[
    bool, _attention_projection_recurrent_halting
] = False
ATTN_RECURRENT_HALTING_OPTION: Annotated[
    type[HaltingConfig], _attention_projection_recurrent_halting
] = StickBreakingConfig
ATTN_RECURRENT_HALTING_THRESHOLD: Annotated[
    float, _attention_projection_recurrent_halting
] = 0.999
ATTN_RECURRENT_HALTING_DROPOUT: Annotated[
    float, _attention_projection_recurrent_halting
] = 0.0
ATTN_RECURRENT_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _attention_projection_recurrent_halting
] = HaltingHiddenStateModeOptions.RAW

_attention_projection_recurrent_halting_stack = (
    _attention_projection_recurrent_halting.child(
        "Attention Projection Recurrent Halting Stack Options"
    )
)
ATTN_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attention_projection_recurrent_halting_stack
] = False
ATTN_RECURRENT_HALTING_STACK_HIDDEN_DIM: Annotated[
    int | None, _attention_projection_recurrent_halting_stack
] = None
ATTN_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attention_projection_recurrent_halting_stack
] = None
ATTN_RECURRENT_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attention_projection_recurrent_halting_stack
] = None
ATTN_RECURRENT_HALTING_STACK_NUM_LAYERS: Annotated[
    int | None, _attention_projection_recurrent_halting_stack
] = None
ATTN_RECURRENT_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_recurrent_halting_stack
] = None
ATTN_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attention_projection_recurrent_halting_stack
] = None
ATTN_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attention_projection_recurrent_halting_stack
] = False
ATTN_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attention_projection_recurrent_halting_stack
] = None
ATTN_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attention_projection_recurrent_halting_stack
] = None
ATTN_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attention_projection_recurrent_halting_stack
] = None
ATTN_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attention_projection_recurrent_halting_stack
] = None
ATTN_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attention_projection_recurrent_halting_stack
] = None
ATTN_RECURRENT_HALTING_STACK_BIAS_FLAG: Annotated[
    bool | None, _attention_projection_recurrent_halting_stack
] = None


_feed_forward_stack = Section(("Feed-Forward Stack Options",))
FF_NUM_LAYERS: Annotated[int, _feed_forward_stack] = 1
FF_BIAS_FLAG: Annotated[bool, _feed_forward_stack] = True
FF_STACK_HIDDEN_DIM: Annotated[int, _feed_forward_stack] = 512
FF_STACK_ACTIVATION: Annotated[ActivationOptions, _feed_forward_stack] = (
    ActivationOptions.RELU
)
FF_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _feed_forward_stack
] = None
FF_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _feed_forward_stack] = False
FF_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _feed_forward_stack] = None
FF_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _feed_forward_stack] = None
FF_STACK_DROPOUT_PROBABILITY: Annotated[float, _feed_forward_stack] = (
    DROPOUT_PROBABILITY
)
FF_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _feed_forward_stack
] = LayerNormPositionOptions.DISABLED
FF_STACK_NORMALIZATION: Annotated[NormalizationOptions, _feed_forward_stack] = (
    NormalizationOptions.RMS_NORM
)
FF_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions, _feed_forward_stack
] = LastLayerBiasOptions.DEFAULT
FF_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool, _feed_forward_stack] = False

_feed_forward_gate = _feed_forward_stack.child("Feed-Forward Gate Options")
FF_STACK_GATE_FLAG: Annotated[bool, _feed_forward_gate] = False
FF_GATE_OPTION: Annotated[LayerGateOptions | None, _feed_forward_gate] = (
    LayerGateOptions.MULTIPLIER
)
FF_GATE_ACTIVATION: Annotated[ActivationOptions | None, _feed_forward_gate] = (
    ActivationOptions.SIGMOID
)

_feed_forward_gate_stack = _feed_forward_gate.child("Feed-Forward Gate Stack Options")
FF_GATE_STACK_INDEPENDENT_FLAG: Annotated[bool, _feed_forward_gate_stack] = False
FF_GATE_STACK_HIDDEN_DIM: Annotated[int | None, _feed_forward_gate_stack] = None
FF_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _feed_forward_gate_stack
] = None
FF_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _feed_forward_gate_stack
] = None
FF_GATE_STACK_NUM_LAYERS: Annotated[int | None, _feed_forward_gate_stack] = None
FF_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _feed_forward_gate_stack
] = ActivationOptions.TANH
FF_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _feed_forward_gate_stack
] = None
FF_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _feed_forward_gate_stack] = False
FF_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _feed_forward_gate_stack] = (
    None
)
FF_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _feed_forward_gate_stack
] = None
FF_GATE_STACK_DROPOUT_PROBABILITY: Annotated[float | None, _feed_forward_gate_stack] = (
    None
)
FF_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _feed_forward_gate_stack
] = None
FF_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _feed_forward_gate_stack
] = True
FF_GATE_STACK_BIAS_FLAG: Annotated[bool | None, _feed_forward_gate_stack] = True

_feed_forward_halting = _feed_forward_stack.child("Feed-Forward Halting Options")
FF_STACK_HALTING_FLAG: Annotated[bool, _feed_forward_halting] = False
FF_HALTING_OPTION: Annotated[type[HaltingConfig], _feed_forward_halting] = (
    StickBreakingConfig
)
FF_HALTING_THRESHOLD: Annotated[float, _feed_forward_halting] = 0.999
FF_HALTING_DROPOUT: Annotated[float, _feed_forward_halting] = 0.0
FF_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _feed_forward_halting
] = HaltingHiddenStateModeOptions.RAW

_feed_forward_halting_stack = _feed_forward_halting.child(
    "Feed-Forward Halting Stack Options"
)
FF_HALTING_STACK_INDEPENDENT_FLAG: Annotated[bool, _feed_forward_halting_stack] = False
FF_HALTING_STACK_HIDDEN_DIM: Annotated[int | None, _feed_forward_halting_stack] = None
FF_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _feed_forward_halting_stack
] = LayerNormPositionOptions.DISABLED
FF_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _feed_forward_halting_stack
] = None
FF_HALTING_STACK_NUM_LAYERS: Annotated[int | None, _feed_forward_halting_stack] = None
FF_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _feed_forward_halting_stack
] = None
FF_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _feed_forward_halting_stack
] = None
FF_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _feed_forward_halting_stack] = (
    False
)
FF_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _feed_forward_halting_stack
] = None
FF_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _feed_forward_halting_stack
] = None
FF_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _feed_forward_halting_stack
] = None
FF_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _feed_forward_halting_stack
] = LastLayerBiasOptions.DISABLED
FF_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _feed_forward_halting_stack
] = None
FF_HALTING_STACK_BIAS_FLAG: Annotated[bool | None, _feed_forward_halting_stack] = None

_feed_forward_memory = _feed_forward_stack.child("Feed-Forward Memory Options")
FF_MEMORY_FLAG: Annotated[bool, _feed_forward_memory] = False
FF_MEMORY_NUM_MEMORY_SLOTS: Annotated[int, _feed_forward_memory] = 4
FF_MEMORY_OPTION: Annotated[type[DynamicMemoryConfig], _feed_forward_memory] = (
    GatedResidualDynamicMemoryConfig
)
FF_MEMORY_POSITION_OPTION: Annotated[MemoryPositionOptions, _feed_forward_memory] = (
    MemoryPositionOptions.AFTER_AFFINE
)
FF_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: Annotated[
    float | None, _feed_forward_memory
] = None
FF_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: Annotated[
    int | None, _feed_forward_memory
] = None

_feed_forward_memory_stack = _feed_forward_memory.child(
    "Feed-Forward Memory Stack Options"
)
FF_MEMORY_STACK_INDEPENDENT_FLAG: Annotated[bool, _feed_forward_memory_stack] = False
FF_MEMORY_STACK_HIDDEN_DIM: Annotated[int | None, _feed_forward_memory_stack] = None
FF_MEMORY_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _feed_forward_memory_stack
] = None
FF_MEMORY_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _feed_forward_memory_stack
] = None
FF_MEMORY_STACK_NUM_LAYERS: Annotated[int | None, _feed_forward_memory_stack] = None
FF_MEMORY_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _feed_forward_memory_stack
] = None
FF_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _feed_forward_memory_stack
] = None
FF_MEMORY_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _feed_forward_memory_stack] = False
FF_MEMORY_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _feed_forward_memory_stack
] = None
FF_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _feed_forward_memory_stack
] = None
FF_MEMORY_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _feed_forward_memory_stack
] = None
FF_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _feed_forward_memory_stack
] = None
FF_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _feed_forward_memory_stack
] = None
FF_MEMORY_STACK_BIAS_FLAG: Annotated[bool | None, _feed_forward_memory_stack] = None

_feed_forward_recurrent_layer = _feed_forward_stack.child(
    "Feed-Forward Recurrent Layer Options"
)
FF_RECURRENT_FLAG: Annotated[bool, _feed_forward_recurrent_layer] = False
FF_RECURRENT_COMPOSITION_OPTION: Annotated[
    type[RecurrentCompositionConfig], _feed_forward_recurrent_layer
] = RecurrentLayerConfig
FF_RECURRENT_MAX_STEPS: Annotated[int, _feed_forward_recurrent_layer] = 2
FF_RECURRENT_NO_GRADIENT_TRANSITION_COUNT: Annotated[
    int | None, _feed_forward_recurrent_layer
] = None
FF_RECURRENT_REINJECT_ORIGINAL_HIDDEN_FLAG: Annotated[
    bool, _feed_forward_recurrent_layer
] = False
FF_RECURRENT_LATENT_UPDATES_PER_ANSWER_UPDATE: Annotated[
    int, _feed_forward_recurrent_layer
] = 2
FF_RECURRENT_ANSWER_UPDATE_COUNT: Annotated[int, _feed_forward_recurrent_layer] = 2
FF_RECURRENT_HIGH_CYCLES: Annotated[int, _feed_forward_recurrent_layer] = 2
FF_RECURRENT_LOW_CYCLES: Annotated[int, _feed_forward_recurrent_layer] = 2
FF_RECURRENT_INITIALIZATION_STANDARD_DEVIATION: Annotated[
    float, _feed_forward_recurrent_layer
] = 1.0
FF_RECURRENT_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _feed_forward_recurrent_layer
] = LayerNormPositionOptions.DISABLED
FF_RECURRENT_NORMALIZATION: Annotated[
    NormalizationOptions, _feed_forward_recurrent_layer
] = NormalizationOptions.LAYER_NORM

_feed_forward_recurrent_gate = _feed_forward_recurrent_layer.child(
    "Feed-Forward Recurrent Gate Options"
)
FF_RECURRENT_STACK_GATE_FLAG: Annotated[bool, _feed_forward_recurrent_gate] = False
FF_RECURRENT_GATE_OPTION: Annotated[
    LayerGateOptions | None, _feed_forward_recurrent_gate
] = LayerGateOptions.MULTIPLIER
FF_RECURRENT_GATE_ACTIVATION: Annotated[
    ActivationOptions | None, _feed_forward_recurrent_gate
] = ActivationOptions.SIGMOID

_feed_forward_recurrent_gate_stack = _feed_forward_recurrent_gate.child(
    "Feed-Forward Recurrent Gate Stack Options"
)
FF_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _feed_forward_recurrent_gate_stack
] = False
FF_RECURRENT_GATE_STACK_HIDDEN_DIM: Annotated[
    int | None, _feed_forward_recurrent_gate_stack
] = None
FF_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _feed_forward_recurrent_gate_stack
] = None
FF_RECURRENT_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _feed_forward_recurrent_gate_stack
] = None
FF_RECURRENT_GATE_STACK_NUM_LAYERS: Annotated[
    int | None, _feed_forward_recurrent_gate_stack
] = None
FF_RECURRENT_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _feed_forward_recurrent_gate_stack
] = None
FF_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _feed_forward_recurrent_gate_stack
] = None
FF_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _feed_forward_recurrent_gate_stack
] = False
FF_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _feed_forward_recurrent_gate_stack
] = None
FF_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _feed_forward_recurrent_gate_stack
] = None
FF_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _feed_forward_recurrent_gate_stack
] = None
FF_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _feed_forward_recurrent_gate_stack
] = None
FF_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _feed_forward_recurrent_gate_stack
] = None
FF_RECURRENT_GATE_STACK_BIAS_FLAG: Annotated[
    bool | None, _feed_forward_recurrent_gate_stack
] = None

_feed_forward_recurrent_halting = _feed_forward_recurrent_layer.child(
    "Feed-Forward Recurrent Halting Options"
)
FF_RECURRENT_STACK_HALTING_FLAG: Annotated[bool, _feed_forward_recurrent_halting] = (
    False
)
FF_RECURRENT_HALTING_OPTION: Annotated[
    type[HaltingConfig], _feed_forward_recurrent_halting
] = StickBreakingConfig
FF_RECURRENT_HALTING_THRESHOLD: Annotated[float, _feed_forward_recurrent_halting] = (
    0.999
)
FF_RECURRENT_HALTING_DROPOUT: Annotated[float, _feed_forward_recurrent_halting] = 0.0
FF_RECURRENT_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _feed_forward_recurrent_halting
] = HaltingHiddenStateModeOptions.RAW

_feed_forward_recurrent_halting_stack = _feed_forward_recurrent_halting.child(
    "Feed-Forward Recurrent Halting Stack Options"
)
FF_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _feed_forward_recurrent_halting_stack
] = False
FF_RECURRENT_HALTING_STACK_HIDDEN_DIM: Annotated[
    int | None, _feed_forward_recurrent_halting_stack
] = None
FF_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _feed_forward_recurrent_halting_stack
] = None
FF_RECURRENT_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _feed_forward_recurrent_halting_stack
] = None
FF_RECURRENT_HALTING_STACK_NUM_LAYERS: Annotated[
    int | None, _feed_forward_recurrent_halting_stack
] = None
FF_RECURRENT_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _feed_forward_recurrent_halting_stack
] = None
FF_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _feed_forward_recurrent_halting_stack
] = None
FF_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _feed_forward_recurrent_halting_stack
] = False
FF_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _feed_forward_recurrent_halting_stack
] = None
FF_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _feed_forward_recurrent_halting_stack
] = None
FF_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _feed_forward_recurrent_halting_stack
] = None
FF_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _feed_forward_recurrent_halting_stack
] = None
FF_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _feed_forward_recurrent_halting_stack
] = None
FF_RECURRENT_HALTING_STACK_BIAS_FLAG: Annotated[
    bool | None, _feed_forward_recurrent_halting_stack
] = None

_transformer = Section(("Transformer Options",))
ENCODER_NUM_LAYERS: Annotated[int, _transformer] = 3
DECODER_NUM_LAYERS: Annotated[int, _transformer] = 3
ENCODER_LAYER_NORM_POSITION: Annotated[LayerNormPositionOptions, _transformer] = (
    LayerNormPositionOptions.BEFORE
)
ENCODER_NORMALIZATION: Annotated[NormalizationOptions, _transformer] = (
    NormalizationOptions.RMS_NORM
)
ENCODER_OUTPUT_NORMALIZATION: Annotated[NormalizationOptions, _transformer] = (
    NormalizationOptions.LAYER_NORM
)
DECODER_OUTPUT_NORMALIZATION: Annotated[NormalizationOptions, _transformer] = (
    NormalizationOptions.LAYER_NORM
)
DECODER_LAYER_NORM_POSITION: Annotated[LayerNormPositionOptions, _transformer] = (
    LayerNormPositionOptions.BEFORE
)
DECODER_NORMALIZATION: Annotated[NormalizationOptions, _transformer] = (
    NormalizationOptions.RMS_NORM
)
ENCODER_ATTN_NUM_HEADS: Annotated[int, _transformer] = ATTN_NUM_HEADS
DECODER_SELF_ATTN_NUM_HEADS: Annotated[int, _transformer] = ATTN_NUM_HEADS
DECODER_CROSS_ATTN_NUM_HEADS: Annotated[int, _transformer] = ATTN_NUM_HEADS
ENCODER_FEED_FORWARD_HIDDEN_DIM: Annotated[int, _transformer] = FF_STACK_HIDDEN_DIM
DECODER_FEED_FORWARD_HIDDEN_DIM: Annotated[int, _transformer] = FF_STACK_HIDDEN_DIM
ENCODER_FEED_FORWARD_NUM_LAYERS: Annotated[int, _transformer] = FF_NUM_LAYERS
DECODER_FEED_FORWARD_NUM_LAYERS: Annotated[int, _transformer] = FF_NUM_LAYERS

_controller = Section(("Controller Options",))
STACK_GATE_FLAG: Annotated[bool, _controller] = False
STACK_HALTING_FLAG: Annotated[bool, _controller] = False
HALTING_THRESHOLD: Annotated[float | None, _controller] = None
MEMORY_FLAG: Annotated[bool, _controller] = False

_recurrent_layer = _controller.child("Recurrent Layer Options")
RECURRENT_FLAG: Annotated[bool, _recurrent_layer] = False
RECURRENT_COMPOSITION_OPTION: Annotated[
    type[RecurrentCompositionConfig], _recurrent_layer
] = RecurrentLayerConfig
RECURRENT_MAX_STEPS: Annotated[int, _recurrent_layer] = 2
RECURRENT_INITIAL_ITERATIONS: Annotated[int, _recurrent_layer] = 2
RECURRENT_GRADIENT_TRANSITION_COUNT: Annotated[int | None, _recurrent_layer] = None
RECURRENT_ITERATION_INCREMENT: Annotated[int, _recurrent_layer] = 1
RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT: Annotated[int, _recurrent_layer] = 1
RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG: Annotated[bool, _recurrent_layer] = False
RECURRENT_NO_GRADIENT_TRANSITION_COUNT: Annotated[int | None, _recurrent_layer] = None
RECURRENT_REINJECT_ORIGINAL_HIDDEN_FLAG: Annotated[bool, _recurrent_layer] = False
RECURRENT_LATENT_UPDATES_PER_ANSWER_UPDATE: Annotated[int, _recurrent_layer] = 2
RECURRENT_ANSWER_UPDATE_COUNT: Annotated[int, _recurrent_layer] = 2
RECURRENT_HIGH_CYCLES: Annotated[int, _recurrent_layer] = 2
RECURRENT_LOW_CYCLES: Annotated[int, _recurrent_layer] = 2
RECURRENT_INITIALIZATION_STANDARD_DEVIATION: Annotated[float, _recurrent_layer] = 1.0
RECURRENT_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _recurrent_layer
] = None
RECURRENT_RESIDUAL_MODEL_FLAG: Annotated[bool, _recurrent_layer] = False
RECURRENT_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _recurrent_layer] = None
RECURRENT_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _recurrent_layer] = None

_recurrent_gate = _recurrent_layer.child("Recurrent Gate Options")
RECURRENT_STACK_GATE_FLAG: Annotated[bool, _recurrent_gate] = False

_recurrent_halting = _recurrent_layer.child("Recurrent Halting Options")
RECURRENT_STACK_HALTING_FLAG: Annotated[bool, _recurrent_halting] = False
RECURRENT_HALTING_THRESHOLD: Annotated[float | None, _recurrent_halting] = None

_residual = _controller.child("Residual Options")
# - False uses the residual variant's learned query or coefficient parameters.
# - True uses the residual stack for input-dependent queries or coefficients.
# - For every attention residual selector, supply its RESIDUAL_BLOCK_SIZE and
#   RESIDUAL_RMS_NORM_EPSILON explicitly (suggested: 1 for full attention, 1e-6).
STACK_RESIDUAL_CONNECTION_OPTION: Annotated[type[ResidualConfig] | None, _residual] = (
    None
)
STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _residual] = False
STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _residual] = None
STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _residual] = None
# - Applies to scalar and model-generated weighted residual coefficients.
#   SIGMOID is recommended; DISABLED leaves the raw coefficient unchanged.
RESIDUAL_COEFFICIENT_ACTIVATION: Annotated[ActivationOptions, _residual] = (
    ActivationOptions.SIGMOID
)

_residual_stack = _residual.child("Residual Stack Options")
# - If False, residual stack options inherit layer stack submodule options.
RESIDUAL_STACK_INDEPENDENT_FLAG: Annotated[bool, _residual_stack] = False
RESIDUAL_STACK_HIDDEN_DIM: Annotated[int | None, _residual_stack] = None
RESIDUAL_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _residual_stack
] = None
RESIDUAL_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _residual_stack
] = None
RESIDUAL_STACK_NUM_LAYERS: Annotated[int | None, _residual_stack] = None
RESIDUAL_STACK_ACTIVATION: Annotated[ActivationOptions | None, _residual_stack] = None
RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _residual_stack
] = None
RESIDUAL_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _residual_stack] = False
RESIDUAL_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _residual_stack] = None
RESIDUAL_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _residual_stack] = (
    None
)
RESIDUAL_STACK_DROPOUT_PROBABILITY: Annotated[float | None, _residual_stack] = None
RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _residual_stack
] = None
RESIDUAL_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _residual_stack
] = None
RESIDUAL_STACK_BIAS_FLAG: Annotated[bool | None, _residual_stack] = None


_trainer = Section(("Trainer",))
NUM_EPOCHS: Annotated[int, _trainer] = 30
TRAINER_ACCELERATOR: Annotated[str, _trainer] = "auto"
TRAINER_DEVICES: Annotated[str | int, _trainer] = "auto"
TRAINER_ACCUMULATE_GRAD_BATCHES: Annotated[int, _trainer] = 1
TRAINER_PRECISION: Annotated[str, _trainer] = "32-true"
TRAINER_DETERMINISTIC: Annotated[bool, _trainer] = True
TRAINER_BENCHMARK: Annotated[bool, _trainer] = False
TRAINER_MAX_STEPS: Annotated[int, _trainer] = -1
TRAINER_MAX_TIME: Annotated[str | None, _trainer] = None
TRAINER_VAL_CHECK_INTERVAL: Annotated[float, _trainer] = 1.0
TRAINER_LIMIT_TRAIN_BATCHES: Annotated[float, _trainer] = 1.0
TRAINER_LIMIT_VAL_BATCHES: Annotated[float, _trainer] = 1.0
TRAINER_OVERFIT_BATCHES: Annotated[int | float, _trainer] = 0.0
TRAINER_NUM_SANITY_VAL_STEPS: Annotated[int, _trainer] = 2
TRAINER_GRADIENT_CLIP_VAL: Annotated[float, _trainer] = 1.0
TRAINER_GRADIENT_CLIP_ALGORITHM: Annotated[str, _trainer] = "norm"
TRAINER_LOG_EVERY_N_STEPS: Annotated[int, _trainer] = 10
TRAINER_ENABLE_PROGRESS_BAR: Annotated[bool, _trainer] = False
TRAINER_ENABLE_CHECKPOINTING: Annotated[bool, _trainer] = False
TRAINER_ENABLE_MODEL_SUMMARY: Annotated[bool, _trainer] = False
TRAINER_PROFILER: Annotated[str | None, _trainer] = None

MONITOR_LOG_EVERY_N_STEPS: Annotated[int, _trainer] = 100

_callback = Section(("Callback",))
CALLBACK_EARLY_STOPPING_PATIENCE: Annotated[int, _callback] = 5
CALLBACK_EARLY_STOPPING_METRIC: Annotated[str, _callback] = "validation/loss"
CALLBACK_EARLY_STOPPING_MIN_DELTA: Annotated[float, _callback] = 0.0
CALLBACK_EARLY_STOPPING_STRICT: Annotated[bool, _callback] = True
CALLBACK_EARLY_STOPPING_CHECK_FINITE: Annotated[bool, _callback] = True
CALLBACK_CHECKPOINT_FLAG: Annotated[bool, _callback] = False

_run = Section(("Run",))
DATA_NUM_WORKERS: Annotated[int, _run] = 0
RUN_TEST_AFTER_FIT: Annotated[bool, _run] = True
SEED: Annotated[int | None, _run] = 0


_CONFIG_FIELD_METADATA_ALIASES: Annotated[dict[str, str], _run] = {}


def _copy_path_runtime_defaults(source_prefix: str, target_prefix: str) -> None:
    for name, value in tuple(globals().items()):
        if not name.isupper() or not name.startswith(source_prefix):
            continue
        target = f"{target_prefix}{name.removeprefix(source_prefix)}"
        if target == "DECODER_CROSS_ATTN_SOURCE_MEMORY_FLAG":
            value = False
        globals()[target] = value
        _CONFIG_FIELD_METADATA_ALIASES[target] = name


for _target_prefix in (
    "ENCODER_ATTN_",
    "DECODER_SELF_ATTN_",
    "DECODER_CROSS_ATTN_",
):
    _copy_path_runtime_defaults("ATTN_", _target_prefix)
for _target_prefix in ("ENCODER_FF_", "DECODER_FF_"):
    _copy_path_runtime_defaults("FF_", _target_prefix)

ENCODER_FF_STACK_HIDDEN_DIM: Annotated[int, _run] = ENCODER_FEED_FORWARD_HIDDEN_DIM
DECODER_FF_STACK_HIDDEN_DIM: Annotated[int, _run] = DECODER_FEED_FORWARD_HIDDEN_DIM
ENCODER_FF_NUM_LAYERS: Annotated[int, _run] = ENCODER_FEED_FORWARD_NUM_LAYERS
DECODER_FF_NUM_LAYERS: Annotated[int, _run] = DECODER_FEED_FORWARD_NUM_LAYERS

inherit_alias_annotations(__annotations__, _CONFIG_FIELD_METADATA_ALIASES)

del _target_prefix


_sequence_expansion = Section(("Sequence Expansion",))
SEQUENCE_EXPANSION_FLAG: Annotated[bool, _sequence_expansion] = False
SEQUENCE_EXPANSION_CONFIG: Annotated[
    SequenceExpansionConfig | None, _sequence_expansion
] = None
SEQUENCE_EXPANSION_MAX_NEW_TOKENS: Annotated[int | None, _sequence_expansion] = None
SEQUENCE_EXPANSION_MODEL_CONFIG: Annotated[ConfigBase | None, _sequence_expansion] = (
    None
)
SEQUENCE_EXPANSION_POSITIONAL_EMBEDDING_CONFIG: Annotated[
    TextLearnedPositionalEmbeddingConfig
    | TextSinusoidalPositionalEmbeddingConfig
    | None,
    _sequence_expansion,
] = None

_encoder_sequence_expansion = Section(("Encoder Sequence Expansion",))
ENCODER_SEQUENCE_EXPANSION_FLAG: Annotated[bool | None, _encoder_sequence_expansion] = (
    None
)
ENCODER_SEQUENCE_EXPANSION_CONFIG: Annotated[
    SequenceExpansionConfig | None, _encoder_sequence_expansion
] = None
ENCODER_SEQUENCE_EXPANSION_MAX_NEW_TOKENS: Annotated[
    int | None, _encoder_sequence_expansion
] = None
ENCODER_SEQUENCE_EXPANSION_MODEL_CONFIG: Annotated[
    ConfigBase | None, _encoder_sequence_expansion
] = None
ENCODER_SEQUENCE_EXPANSION_POSITIONAL_EMBEDDING_CONFIG: Annotated[
    TextLearnedPositionalEmbeddingConfig
    | TextSinusoidalPositionalEmbeddingConfig
    | None,
    _encoder_sequence_expansion,
] = None

_decoder_sequence_expansion = Section(("Decoder Sequence Expansion",))
DECODER_SEQUENCE_EXPANSION_FLAG: Annotated[bool | None, _decoder_sequence_expansion] = (
    None
)
DECODER_SEQUENCE_EXPANSION_CONFIG: Annotated[
    SequenceExpansionConfig | None, _decoder_sequence_expansion
] = None
DECODER_SEQUENCE_EXPANSION_MAX_NEW_TOKENS: Annotated[
    int | None, _decoder_sequence_expansion
] = None
DECODER_SEQUENCE_EXPANSION_MODEL_CONFIG: Annotated[
    ConfigBase | None, _decoder_sequence_expansion
] = None
DECODER_SEQUENCE_EXPANSION_POSITIONAL_EMBEDDING_CONFIG: Annotated[
    TextLearnedPositionalEmbeddingConfig
    | TextSinusoidalPositionalEmbeddingConfig
    | None,
    _decoder_sequence_expansion,
] = None

_component_defaults.declare_component_defaults(globals())
