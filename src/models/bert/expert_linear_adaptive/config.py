from typing import Annotated

from emperor.attention import AttentionSourceMemoryConfig  # noqa: F401
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,  # noqa: F401
    AdaptiveParameterAugmentationConfig,  # noqa: F401
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    AdditiveDynamicBiasConfig,  # noqa: F401
    AffineTransformDynamicBiasConfig,  # noqa: F401
    AntiDynamicDiagonalConfig,  # noqa: F401
    AttentionGroupingConfig,  # noqa: F401
    AxisMaskConfig,
    BankExpansionFactorOptions,
    BroadcastSumDynamicWeightConfig,  # noqa: F401
    CombinedDynamicDiagonalConfig,  # noqa: F401
    DiagonalAxisMaskConfig,  # noqa: F401
    DiagonallyModulatedLowRankDynamicWeightConfig,  # noqa: F401
    DualModelDynamicWeightConfig,  # noqa: F401
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GeneratorDynamicBiasConfig,  # noqa: F401
    GroupingConfig,
    HypernetworkDynamicWeightConfig,  # noqa: F401
    LayeredWeightedBankDynamicWeightConfig,  # noqa: F401
    LowRankDynamicWeightConfig,  # noqa: F401
    LowRankFactorSourceOptions,
    MaskDimensionOptions,
    MatrixBiasMixtureConfig,  # noqa: F401
    MatrixWeightsMixtureConfig,  # noqa: F401
    MeanGroupingConfig,  # noqa: F401
    MeanStdGroupingConfig,  # noqa: F401
    MultiplicativeDynamicBiasConfig,  # noqa: F401
    OuterProductMaskConfig,  # noqa: F401
    PerAxisScoreMaskConfig,  # noqa: F401
    RMSGroupingConfig,  # noqa: F401
    SigmoidGatedDynamicBiasConfig,  # noqa: F401
    SingleModelDynamicWeightConfig,  # noqa: F401
    SoftWeightedBankDynamicWeightConfig,  # noqa: F401
    StandardDynamicDiagonalConfig,  # noqa: F401
    SumGroupingConfig,  # noqa: F401
    SummaryNormalizationOptions,
    TanhGatedDynamicBiasConfig,  # noqa: F401
    TopSliceAxisMaskConfig,  # noqa: F401
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,  # noqa: F401
    WeightInformedScoreAxisMaskConfig,  # noqa: F401
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.config import ConfigBase
from emperor.datasets.text.bert_pretraining import (
    BERT_PRETRAINING_TARGET_VOCAB_SIZE,
)
from emperor.embedding.absolute import (
    AbsolutePositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,  # noqa: F401
)
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
from emperor.layers import (  # noqa: F401
    ActivationOptions,
    AdditiveResidualConfig,  # noqa: F401
    AttentionResidualConfig,  # noqa: F401
    LastLayerBiasOptions,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    ResidualConfig,
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
from model_runtime.packages.configuration_metadata import Section
from model_runtime.packages.metric_direction import MetricDirection
from model_runtime.packages.runtime_values import (
    RuntimeValueConstraints,
    positive_runtime_fields,
)
from model_runtime.packages.soft_moe import declare_soft_moe_runtime_defaults
from models import (
    _attention_source_memory,
    _bert_embedding,
    _component_defaults,
    _recurrent_compression,
)

_trainer = Section(("Trainer",))
TRAINER_ACCELERATOR: Annotated[str, _trainer] = "cpu"
TRAINER_DEVICES: Annotated[int, _trainer] = 1
TRAINER_GRADIENT_CLIP_VAL: Annotated[float, _trainer] = 1.0
TRAINER_GRADIENT_CLIP_ALGORITHM: Annotated[str, _trainer] = "norm"
TRAINER_ACCUMULATE_GRAD_BATCHES: Annotated[int, _trainer] = 1
TRAINER_PRECISION: Annotated[str, _trainer] = "32-true"
TRAINER_DETERMINISTIC: Annotated[bool, _trainer] = False
TRAINER_BENCHMARK: Annotated[bool, _trainer] = True
TRAINER_MAX_STEPS: Annotated[int, _trainer] = -1
TRAINER_MAX_TIME: Annotated[str | None, _trainer] = None
TRAINER_VAL_CHECK_INTERVAL: Annotated[float, _trainer] = 1.0
TRAINER_LIMIT_TRAIN_BATCHES: Annotated[float, _trainer] = 1.0
TRAINER_LIMIT_VAL_BATCHES: Annotated[float, _trainer] = 1.0
TRAINER_OVERFIT_BATCHES: Annotated[int | float, _trainer] = 0.0
TRAINER_NUM_SANITY_VAL_STEPS: Annotated[int, _trainer] = 2
TRAINER_LOG_EVERY_N_STEPS: Annotated[int, _trainer] = 50
TRAINER_ENABLE_PROGRESS_BAR: Annotated[bool, _trainer] = False
TRAINER_ENABLE_CHECKPOINTING: Annotated[bool, _trainer] = False
TRAINER_ENABLE_MODEL_SUMMARY: Annotated[bool, _trainer] = False
TRAINER_PROFILER: Annotated[str | None, _trainer] = None

MONITOR_LOG_EVERY_N_STEPS: Annotated[int, _trainer] = 100

_run = Section(("Run",))
DATA_NUM_WORKERS: Annotated[int, _run] = 4
RUN_TEST_AFTER_FIT: Annotated[bool, _run] = True
SEED: Annotated[int | None, _run] = None

_callback = Section(("Callback",))
CALLBACK_EARLY_STOPPING_PATIENCE: Annotated[int, _callback] = 0
CALLBACK_EARLY_STOPPING_METRIC: Annotated[str, _callback] = "validation/loss"
CALLBACK_EARLY_STOPPING_MODE: Annotated[MetricDirection | None, _callback] = (
    None  # Optional min/max direction for the monitored metric.
)
CALLBACK_EARLY_STOPPING_MIN_DELTA: Annotated[float, _callback] = 0.0
CALLBACK_EARLY_STOPPING_STRICT: Annotated[bool, _callback] = True
CALLBACK_EARLY_STOPPING_CHECK_FINITE: Annotated[bool, _callback] = True
CALLBACK_CHECKPOINT_FLAG: Annotated[bool, _callback] = False

_general = Section(("Global",))
INPUT_DIM: Annotated[int, _general] = BERT_PRETRAINING_TARGET_VOCAB_SIZE
HIDDEN_DIM: Annotated[int, _general] = 32
OUTPUT_DIM: Annotated[int, _general] = BERT_PRETRAINING_TARGET_VOCAB_SIZE
BATCH_SIZE: Annotated[int, _general] = 64
LEARNING_RATE: Annotated[float, _general] = 1e-3
NUM_EPOCHS: Annotated[int, _general] = 10
CONFIG_OVERRIDE_SKIP_KEYS: Annotated[set[str], _general] = {
    "BERT_PRETRAINING_TARGET_VOCAB_SIZE",
}
SEQUENCE_LENGTH: Annotated[int, _general] = 35

_positional_embedding = Section(("Positional Embedding",))
POSITIONAL_EMBEDDING_OPTION: Annotated[
    type[AbsolutePositionalEmbeddingConfig], _positional_embedding
] = TextLearnedPositionalEmbeddingConfig
POSITIONAL_EMBEDDING_PADDING_IDX: Annotated[int, _positional_embedding] = 0
POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG: Annotated[bool, _positional_embedding] = False

_layer_stack = Section(("Layer Stack Options",))
# - hidden_dim comes from the global HIDDEN_DIM field above.
STACK_NUM_LAYERS: Annotated[int, _layer_stack] = 2
STACK_ACTIVATION: Annotated[ActivationOptions, _layer_stack] = ActivationOptions.GELU
STACK_DROPOUT_PROBABILITY: Annotated[float, _layer_stack] = 0.0
LAYER_NORM_POSITION: Annotated[LayerNormPositionOptions, _layer_stack] = (
    LayerNormPositionOptions.AFTER
)
NORMALIZATION: Annotated[NormalizationOptions, _layer_stack] = (
    NormalizationOptions.RMS_NORM
)
ENCODER_OUTPUT_NORMALIZATION: Annotated[NormalizationOptions, _layer_stack] = (
    NormalizationOptions.LAYER_NORM
)
CAUSAL_ATTENTION_MASK_FLAG: Annotated[bool, _layer_stack] = False

_embedding_pipeline = Section(("Embedding Pipeline",))
TOKEN_TYPE_VOCAB_SIZE: Annotated[int, _embedding_pipeline] = 2
EMBEDDING_LAYER_NORM_FLAG: Annotated[bool, _embedding_pipeline] = True
EMBEDDING_NORMALIZATION: Annotated[NormalizationOptions, _embedding_pipeline] = (
    NormalizationOptions.LAYER_NORM
)
EMBEDDING_DROPOUT_PROBABILITY: Annotated[float, _embedding_pipeline] = (
    STACK_DROPOUT_PROBABILITY
)
HIERARCHICAL_EMBEDDING_CONFIG: Annotated[
    HierarchicalByteEmbeddingConfig | None, _embedding_pipeline
] = None
CONTEXTUAL_EMBEDDING_CONFIG: Annotated[
    ByteContextualEmbeddingConfig | None, _embedding_pipeline
] = None

_masked_language_modeling_head = Section(("Masked-Language-Modeling Head",))
MLM_ACTIVATION: Annotated[ActivationOptions, _masked_language_modeling_head] = (
    ActivationOptions.GELU
)
MLM_DENSE_BIAS_FLAG: Annotated[bool, _masked_language_modeling_head] = True
MLM_LAYER_NORM_FLAG: Annotated[bool, _masked_language_modeling_head] = True
MLM_NORMALIZATION: Annotated[NormalizationOptions, _masked_language_modeling_head] = (
    NormalizationOptions.LAYER_NORM
)
MLM_DECODER_BIAS_FLAG: Annotated[bool, _masked_language_modeling_head] = True
MLM_DECODER_WEIGHT_TYING_FLAG: Annotated[bool, _masked_language_modeling_head] = True

_next_sentence_prediction_head = Section(("Next-Sentence-Prediction Head",))
NSP_POOLER_ACTIVATION: Annotated[ActivationOptions, _next_sentence_prediction_head] = (
    ActivationOptions.TANH
)
NSP_POOLER_BIAS_FLAG: Annotated[bool, _next_sentence_prediction_head] = True
NSP_HEAD_BIAS_FLAG: Annotated[bool, _next_sentence_prediction_head] = True

_submodule_stack = Section(("Layer Stack Submodule Options",))
SUBMODULE_STACK_HIDDEN_DIM: Annotated[int, _submodule_stack] = HIDDEN_DIM
SUBMODULE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _submodule_stack
] = LAYER_NORM_POSITION
SUBMODULE_STACK_NORMALIZATION: Annotated[NormalizationOptions, _submodule_stack] = (
    NORMALIZATION
)
SUBMODULE_STACK_NUM_LAYERS: Annotated[int, _submodule_stack] = 2
SUBMODULE_STACK_ACTIVATION: Annotated[ActivationOptions, _submodule_stack] = (
    ActivationOptions.GELU
)
SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _submodule_stack
] = None
SUBMODULE_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _submodule_stack] = False
SUBMODULE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _submodule_stack] = None
SUBMODULE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _submodule_stack] = (
    None
)
SUBMODULE_STACK_DROPOUT_PROBABILITY: Annotated[float, _submodule_stack] = 0.0
SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions, _submodule_stack
] = LastLayerBiasOptions.DEFAULT
SUBMODULE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool, _submodule_stack] = (
    False
)
SUBMODULE_STACK_BIAS_FLAG: Annotated[bool, _submodule_stack] = True

_residual = Section(("Residual Options",))
# - False uses the residual variant's learned query or coefficient parameters.
# - True uses the residual stack for input-dependent queries or coefficients.
# - For every attention residual selector, supply its RESIDUAL_BLOCK_SIZE and
#   RESIDUAL_RMS_NORM_EPSILON explicitly (suggested: 1 for full attention, 1e-6).
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


_gate = Section(("Gate Options",))
STACK_GATE_FLAG: Annotated[bool, _gate] = False
GATE_OPTION: Annotated[LayerGateOptions | None, _gate] = LayerGateOptions.MULTIPLIER
GATE_ACTIVATION: Annotated[ActivationOptions | None, _gate] = ActivationOptions.SIGMOID
_gate_stack = _gate.child("Gate Stack Options")
GATE_STACK_INDEPENDENT_FLAG: Annotated[bool, _gate_stack] = False
GATE_STACK_HIDDEN_DIM: Annotated[int | None, _gate_stack] = None
GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _gate_stack
] = None
GATE_STACK_NORMALIZATION: Annotated[NormalizationOptions | None, _gate_stack] = None
GATE_STACK_NUM_LAYERS: Annotated[int | None, _gate_stack] = None
GATE_STACK_ACTIVATION: Annotated[ActivationOptions | None, _gate_stack] = (
    ActivationOptions.TANH
)
GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _gate_stack
] = None
GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _gate_stack] = False
GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _gate_stack] = None
GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _gate_stack] = None
GATE_STACK_DROPOUT_PROBABILITY: Annotated[float | None, _gate_stack] = None
GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _gate_stack
] = None
GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool | None, _gate_stack] = True
GATE_STACK_BIAS_FLAG: Annotated[bool | None, _gate_stack] = True

_halting = Section(("Halting Options",))
STACK_HALTING_FLAG: Annotated[bool, _halting] = False
HALTING_OPTION: Annotated[type[HaltingConfig], _halting] = StickBreakingConfig
HALTING_THRESHOLD: Annotated[float, _halting] = 0.999
HALTING_DROPOUT: Annotated[float, _halting] = 0.0
HALTING_HIDDEN_STATE_MODE: Annotated[HaltingHiddenStateModeOptions, _halting] = (
    HaltingHiddenStateModeOptions.RAW
)
HALTING_OUTPUT_DIM: Annotated[int, _halting] = 2
_halting_stack = _halting.child("Halting Stack Options")
HALTING_STACK_INDEPENDENT_FLAG: Annotated[bool, _halting_stack] = False
HALTING_STACK_HIDDEN_DIM: Annotated[int | None, _halting_stack] = None
HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _halting_stack
] = LayerNormPositionOptions.DISABLED
HALTING_STACK_NORMALIZATION: Annotated[NormalizationOptions | None, _halting_stack] = (
    None
)
HALTING_STACK_NUM_LAYERS: Annotated[int | None, _halting_stack] = None
HALTING_STACK_ACTIVATION: Annotated[ActivationOptions | None, _halting_stack] = None
HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _halting_stack
] = None
HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _halting_stack] = False
HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _halting_stack] = None
HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _halting_stack] = None
HALTING_STACK_DROPOUT_PROBABILITY: Annotated[float | None, _halting_stack] = None
HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _halting_stack
] = LastLayerBiasOptions.DISABLED
HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _halting_stack
] = None
HALTING_STACK_BIAS_FLAG: Annotated[bool | None, _halting_stack] = None

_memory = Section(("Memory Options",))
MEMORY_FLAG: Annotated[bool, _memory] = False
MEMORY_OPTION: Annotated[type[DynamicMemoryConfig], _memory] = (
    GatedResidualDynamicMemoryConfig
)
MEMORY_POSITION_OPTION: Annotated[MemoryPositionOptions, _memory] = (
    MemoryPositionOptions.AFTER_AFFINE
)
MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: Annotated[float | None, _memory] = None
MEMORY_NUM_MEMORY_SLOTS: Annotated[int, _memory] = 4
MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: Annotated[int | None, _memory] = None
_memory_stack = _memory.child("Memory Stack Options")
MEMORY_STACK_INDEPENDENT_FLAG: Annotated[bool, _memory_stack] = False
MEMORY_STACK_HIDDEN_DIM: Annotated[int | None, _memory_stack] = None
MEMORY_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _memory_stack
] = None
MEMORY_STACK_NORMALIZATION: Annotated[NormalizationOptions | None, _memory_stack] = None
MEMORY_STACK_NUM_LAYERS: Annotated[int | None, _memory_stack] = None
MEMORY_STACK_ACTIVATION: Annotated[ActivationOptions | None, _memory_stack] = None
MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _memory_stack
] = None
MEMORY_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _memory_stack] = False
MEMORY_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _memory_stack] = None
MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _memory_stack] = None
MEMORY_STACK_DROPOUT_PROBABILITY: Annotated[float | None, _memory_stack] = None
MEMORY_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _memory_stack
] = None
MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool | None, _memory_stack] = (
    None
)
MEMORY_STACK_BIAS_FLAG: Annotated[bool | None, _memory_stack] = None

_recurrent_layer = Section(("Recurrent Layer Options",))
RECURRENT_FLAG: Annotated[bool, _recurrent_layer] = False
RECURRENT_MAX_STEPS: Annotated[int, _recurrent_layer] = 4
RECURRENT_INITIAL_ITERATIONS: Annotated[int, _recurrent_layer] = 2
RECURRENT_GRADIENT_TRANSITION_COUNT: Annotated[int | None, _recurrent_layer] = None
RECURRENT_NO_GRADIENT_TRANSITION_COUNT: Annotated[int | None, _recurrent_layer] = None
RECURRENT_ITERATION_INCREMENT: Annotated[int, _recurrent_layer] = 1
RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT: Annotated[int, _recurrent_layer] = 1
RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG: Annotated[bool, _recurrent_layer] = False
RECURRENT_LAYER_NORM_POSITION: Annotated[LayerNormPositionOptions, _recurrent_layer] = (
    LayerNormPositionOptions.DISABLED
)
RECURRENT_NORMALIZATION: Annotated[NormalizationOptions, _recurrent_layer] = (
    NormalizationOptions.LAYER_NORM
)
RECURRENT_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _recurrent_layer
] = None
RECURRENT_RESIDUAL_MODEL_FLAG: Annotated[bool, _recurrent_layer] = False
RECURRENT_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _recurrent_layer] = None
RECURRENT_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _recurrent_layer] = None

_recurrent_gate = _recurrent_layer.child("Recurrent Gate Options")
RECURRENT_STACK_GATE_FLAG: Annotated[bool, _recurrent_gate] = False
RECURRENT_GATE_OPTION: Annotated[LayerGateOptions | None, _recurrent_gate] = (
    LayerGateOptions.MULTIPLIER
)
RECURRENT_GATE_ACTIVATION: Annotated[ActivationOptions | None, _recurrent_gate] = (
    ActivationOptions.SIGMOID
)
_recurrent_gate_stack = _recurrent_gate.child("Recurrent Gate Stack Options")
RECURRENT_GATE_STACK_INDEPENDENT_FLAG: Annotated[bool, _recurrent_gate_stack] = False
RECURRENT_GATE_STACK_HIDDEN_DIM: Annotated[int | None, _recurrent_gate_stack] = None
RECURRENT_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _recurrent_gate_stack
] = None
RECURRENT_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _recurrent_gate_stack
] = None
RECURRENT_GATE_STACK_NUM_LAYERS: Annotated[int | None, _recurrent_gate_stack] = None
RECURRENT_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _recurrent_gate_stack
] = None
RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _recurrent_gate_stack
] = None
RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _recurrent_gate_stack] = False
RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _recurrent_gate_stack
] = None
RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _recurrent_gate_stack
] = None
RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _recurrent_gate_stack
] = None
RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _recurrent_gate_stack
] = None
RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _recurrent_gate_stack
] = None
RECURRENT_GATE_STACK_BIAS_FLAG: Annotated[bool | None, _recurrent_gate_stack] = None

_recurrent_halting = _recurrent_layer.child("Recurrent Halting Options")
RECURRENT_STACK_HALTING_FLAG: Annotated[bool, _recurrent_halting] = False
RECURRENT_MIN_STEPS: Annotated[int, _recurrent_halting] = 1
RECURRENT_PONDER_COST_WEIGHT: Annotated[float, _recurrent_halting] = 1.0
RECURRENT_HALTING_OPTION: Annotated[type[HaltingConfig], _recurrent_halting] = (
    StickBreakingConfig
)
RECURRENT_HALTING_THRESHOLD: Annotated[float, _recurrent_halting] = HALTING_THRESHOLD
RECURRENT_HALTING_DROPOUT: Annotated[float, _recurrent_halting] = HALTING_DROPOUT
RECURRENT_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _recurrent_halting
] = HALTING_HIDDEN_STATE_MODE
_recurrent_halting_stack = _recurrent_halting.child("Recurrent Halting Stack Options")
RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: Annotated[bool, _recurrent_halting_stack] = (
    False
)
RECURRENT_HALTING_STACK_HIDDEN_DIM: Annotated[int | None, _recurrent_halting_stack] = (
    None
)
RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _recurrent_halting_stack
] = None
RECURRENT_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _recurrent_halting_stack
] = None
RECURRENT_HALTING_STACK_NUM_LAYERS: Annotated[int | None, _recurrent_halting_stack] = (
    None
)
RECURRENT_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _recurrent_halting_stack
] = None
RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _recurrent_halting_stack
] = None
RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _recurrent_halting_stack
] = False
RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _recurrent_halting_stack
] = None
RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _recurrent_halting_stack
] = None
RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _recurrent_halting_stack
] = None
RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _recurrent_halting_stack
] = None
RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _recurrent_halting_stack
] = None
RECURRENT_HALTING_STACK_BIAS_FLAG: Annotated[bool | None, _recurrent_halting_stack] = (
    None
)


_attention = Section(("Attention Options",))
ATTENTION_SOFT_MOE_CONFIG: Annotated[SoftMoEConfig | None, _attention] = None
ATTENTION_SOFT_MOE_FLAG: Annotated[bool, _attention] = False
ATTN_NUM_HEADS: Annotated[int, _attention] = 4
ATTN_ADD_KEY_VALUE_BIAS_FLAG: Annotated[bool, _attention] = False

_attention_projection_stack = _attention.child("Attention Projection Stack Options")
ATTN_NUM_LAYERS: Annotated[int, _attention_projection_stack] = 1
ATTN_BIAS_FLAG: Annotated[bool, _attention_projection_stack] = False
ATTN_STACK_HIDDEN_DIM: Annotated[int, _attention_projection_stack] = HIDDEN_DIM
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
    GATE_OPTION
)
ATTN_GATE_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_gate
] = GATE_ACTIVATION
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
] = GATE_STACK_ACTIVATION
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
] = GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
ATTN_GATE_STACK_BIAS_FLAG: Annotated[bool | None, _attention_projection_gate_stack] = (
    GATE_STACK_BIAS_FLAG
)

_attention_projection_halting = _attention_projection_stack.child(
    "Attention Projection Halting Options"
)
ATTN_STACK_HALTING_FLAG: Annotated[bool, _attention_projection_halting] = False
ATTN_HALTING_OPTION: Annotated[type[HaltingConfig], _attention_projection_halting] = (
    StickBreakingConfig
)
ATTN_HALTING_THRESHOLD: Annotated[float, _attention_projection_halting] = (
    HALTING_THRESHOLD
)
ATTN_HALTING_DROPOUT: Annotated[float, _attention_projection_halting] = HALTING_DROPOUT
ATTN_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _attention_projection_halting
] = HALTING_HIDDEN_STATE_MODE
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
] = HALTING_STACK_LAYER_NORM_POSITION
ATTN_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attention_projection_halting_stack
] = HALTING_STACK_NORMALIZATION
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
] = HALTING_STACK_LAST_LAYER_BIAS_OPTION
ATTN_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attention_projection_halting_stack
] = None
ATTN_HALTING_STACK_BIAS_FLAG: Annotated[
    bool | None, _attention_projection_halting_stack
] = None

_attention_source_memory_section = _attention_projection_stack.child(
    "Attention Source Memory Options"
)
# Temporary memory tokens appended to attention keys and values.
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
_attention_projection_memory = _attention_projection_stack.child(
    "Attention Projection Memory Options"
)
ATTN_MEMORY_FLAG: Annotated[bool, _attention_projection_memory] = False
ATTN_MEMORY_OPTION: Annotated[
    type[DynamicMemoryConfig], _attention_projection_memory
] = MEMORY_OPTION
ATTN_MEMORY_POSITION_OPTION: Annotated[
    MemoryPositionOptions, _attention_projection_memory
] = MEMORY_POSITION_OPTION
ATTN_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: Annotated[
    float | None, _attention_projection_memory
] = MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
ATTN_MEMORY_NUM_MEMORY_SLOTS: Annotated[int, _attention_projection_memory] = 4
ATTN_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: Annotated[
    int | None, _attention_projection_memory
] = MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
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
ATTN_RECURRENT_MAX_STEPS: Annotated[int, _attention_projection_recurrent_layer] = (
    RECURRENT_MAX_STEPS
)
ATTN_RECURRENT_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _attention_projection_recurrent_layer
] = RECURRENT_LAYER_NORM_POSITION
ATTN_RECURRENT_NORMALIZATION: Annotated[
    NormalizationOptions, _attention_projection_recurrent_layer
] = RECURRENT_NORMALIZATION

_attention_projection_recurrent_gate = _attention_projection_recurrent_layer.child(
    "Attention Projection Recurrent Gate Options"
)
ATTN_RECURRENT_STACK_GATE_FLAG: Annotated[
    bool, _attention_projection_recurrent_gate
] = False
ATTN_RECURRENT_GATE_OPTION: Annotated[
    LayerGateOptions | None, _attention_projection_recurrent_gate
] = RECURRENT_GATE_OPTION
ATTN_RECURRENT_GATE_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_recurrent_gate
] = RECURRENT_GATE_ACTIVATION
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
] = RECURRENT_HALTING_THRESHOLD
ATTN_RECURRENT_HALTING_DROPOUT: Annotated[
    float, _attention_projection_recurrent_halting
] = RECURRENT_HALTING_DROPOUT
ATTN_RECURRENT_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _attention_projection_recurrent_halting
] = RECURRENT_HALTING_HIDDEN_STATE_MODE
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
SOFT_MOE_CONFIG: Annotated[SoftMoEConfig | None, _feed_forward_stack] = None
SOFT_MOE_FLAG: Annotated[bool, _feed_forward_stack] = False
# Shared defaults for the attention and feedforward Soft MoE configurations.
# Explicit SoftMoEConfig instances retain their own router and slot settings.
SOFT_MOE_SLOTS_PER_EXPERT: Annotated[int, _feed_forward_stack] = 2
SOFT_MOE_SCORING_HIDDEN_DIM: Annotated[int, _feed_forward_stack] = 32
SOFT_MOE_SCORING_NUM_LAYERS: Annotated[int, _feed_forward_stack] = 2
FF_BIAS_FLAG: Annotated[bool, _feed_forward_stack] = True
FF_STACK_HIDDEN_DIM: Annotated[int, _feed_forward_stack] = HIDDEN_DIM
FF_STACK_ACTIVATION: Annotated[ActivationOptions, _feed_forward_stack] = (
    STACK_ACTIVATION
)
FF_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _feed_forward_stack
] = None
FF_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _feed_forward_stack] = False
FF_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _feed_forward_stack] = None
FF_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _feed_forward_stack] = None
FF_STACK_DROPOUT_PROBABILITY: Annotated[float, _feed_forward_stack] = (
    STACK_DROPOUT_PROBABILITY
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
FF_GATE_OPTION: Annotated[LayerGateOptions | None, _feed_forward_gate] = GATE_OPTION
FF_GATE_ACTIVATION: Annotated[ActivationOptions | None, _feed_forward_gate] = (
    GATE_ACTIVATION
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
] = GATE_STACK_ACTIVATION
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
] = GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
FF_GATE_STACK_BIAS_FLAG: Annotated[bool | None, _feed_forward_gate_stack] = (
    GATE_STACK_BIAS_FLAG
)

_feed_forward_halting = _feed_forward_stack.child("Feed-Forward Halting Options")
FF_STACK_HALTING_FLAG: Annotated[bool, _feed_forward_halting] = False
FF_HALTING_OPTION: Annotated[type[HaltingConfig], _feed_forward_halting] = (
    StickBreakingConfig
)
FF_HALTING_THRESHOLD: Annotated[float, _feed_forward_halting] = HALTING_THRESHOLD
FF_HALTING_DROPOUT: Annotated[float, _feed_forward_halting] = HALTING_DROPOUT
FF_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _feed_forward_halting
] = HALTING_HIDDEN_STATE_MODE
_feed_forward_halting_stack = _feed_forward_halting.child(
    "Feed-Forward Halting Stack Options"
)
FF_HALTING_STACK_INDEPENDENT_FLAG: Annotated[bool, _feed_forward_halting_stack] = False
FF_HALTING_STACK_HIDDEN_DIM: Annotated[int | None, _feed_forward_halting_stack] = None
FF_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _feed_forward_halting_stack
] = HALTING_STACK_LAYER_NORM_POSITION
FF_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _feed_forward_halting_stack
] = HALTING_STACK_NORMALIZATION
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
] = HALTING_STACK_LAST_LAYER_BIAS_OPTION
FF_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _feed_forward_halting_stack
] = None
FF_HALTING_STACK_BIAS_FLAG: Annotated[bool | None, _feed_forward_halting_stack] = None

_feed_forward_memory = _feed_forward_stack.child("Feed-Forward Memory Options")
FF_MEMORY_FLAG: Annotated[bool, _feed_forward_memory] = False
FF_MEMORY_OPTION: Annotated[type[DynamicMemoryConfig], _feed_forward_memory] = (
    MEMORY_OPTION
)
FF_MEMORY_POSITION_OPTION: Annotated[MemoryPositionOptions, _feed_forward_memory] = (
    MEMORY_POSITION_OPTION
)
FF_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: Annotated[
    float | None, _feed_forward_memory
] = MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
FF_MEMORY_NUM_MEMORY_SLOTS: Annotated[int, _feed_forward_memory] = 4
FF_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: Annotated[
    int | None, _feed_forward_memory
] = MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
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
FF_RECURRENT_MAX_STEPS: Annotated[int, _feed_forward_recurrent_layer] = (
    RECURRENT_MAX_STEPS
)
FF_RECURRENT_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _feed_forward_recurrent_layer
] = RECURRENT_LAYER_NORM_POSITION
FF_RECURRENT_NORMALIZATION: Annotated[
    NormalizationOptions, _feed_forward_recurrent_layer
] = RECURRENT_NORMALIZATION

_feed_forward_recurrent_gate = _feed_forward_recurrent_layer.child(
    "Feed-Forward Recurrent Gate Options"
)
FF_RECURRENT_STACK_GATE_FLAG: Annotated[bool, _feed_forward_recurrent_gate] = False
FF_RECURRENT_GATE_OPTION: Annotated[
    LayerGateOptions | None, _feed_forward_recurrent_gate
] = RECURRENT_GATE_OPTION
FF_RECURRENT_GATE_ACTIVATION: Annotated[
    ActivationOptions | None, _feed_forward_recurrent_gate
] = RECURRENT_GATE_ACTIVATION
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
    RECURRENT_HALTING_THRESHOLD
)
FF_RECURRENT_HALTING_DROPOUT: Annotated[float, _feed_forward_recurrent_halting] = (
    RECURRENT_HALTING_DROPOUT
)
FF_RECURRENT_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _feed_forward_recurrent_halting
] = RECURRENT_HALTING_HIDDEN_STATE_MODE
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

_mixture_of_experts_model = Section(("Mixture Of Experts Model Options",))
EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG: Annotated[
    bool, _mixture_of_experts_model
] = False
TOP_K: Annotated[int, _mixture_of_experts_model] = 3
NUM_EXPERTS: Annotated[int, _mixture_of_experts_model] = 12
CAPACITY_FACTOR: Annotated[float, _mixture_of_experts_model] = 0.0
DROPPED_TOKEN_BEHAVIOR: Annotated[DroppedTokenOptions, _mixture_of_experts_model] = (
    DroppedTokenOptions.ZEROS
)
COMPUTE_EXPERT_MIXTURE_FLAG: Annotated[bool, _mixture_of_experts_model] = True
WEIGHTED_PARAMETERS_FLAG: Annotated[bool, _mixture_of_experts_model] = True
WEIGHTING_POSITION_OPTION: Annotated[
    ExpertWeightingPositionOptions, _mixture_of_experts_model
] = ExpertWeightingPositionOptions.AFTER_EXPERTS
ROUTING_INITIALIZATION_MODE: Annotated[
    RoutingInitializationMode, _mixture_of_experts_model
] = RoutingInitializationMode.LAYER

_expert_stack = _mixture_of_experts_model.child("Expert Stack Options")
EXPERT_STACK_HIDDEN_DIM: Annotated[int, _expert_stack] = SUBMODULE_STACK_HIDDEN_DIM
EXPERT_STACK_NUM_LAYERS: Annotated[int, _expert_stack] = SUBMODULE_STACK_NUM_LAYERS
EXPERT_STACK_ACTIVATION: Annotated[ActivationOptions, _expert_stack] = (
    SUBMODULE_STACK_ACTIVATION
)
EXPERT_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _expert_stack
] = SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
EXPERT_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _expert_stack] = False
EXPERT_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _expert_stack] = None
EXPERT_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _expert_stack] = None
EXPERT_STACK_DROPOUT_PROBABILITY: Annotated[float, _expert_stack] = (
    SUBMODULE_STACK_DROPOUT_PROBABILITY
)
EXPERT_STACK_LAYER_NORM_POSITION: Annotated[LayerNormPositionOptions, _expert_stack] = (
    LayerNormPositionOptions.DISABLED
)
EXPERT_STACK_NORMALIZATION: Annotated[NormalizationOptions, _expert_stack] = (
    NormalizationOptions.RMS_NORM
)
EXPERT_STACK_LAST_LAYER_BIAS_OPTION: Annotated[LastLayerBiasOptions, _expert_stack] = (
    SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION
)
EXPERT_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool, _expert_stack] = True
EXPERT_BIAS_FLAG: Annotated[bool, _expert_stack] = SUBMODULE_STACK_BIAS_FLAG

_expert_gate = _mixture_of_experts_model.child("Expert Gate Options")
# If `EXPERT_STACK_GATE_FLAG` is False, the expert gate parameters below are ignored.
EXPERT_STACK_GATE_FLAG: Annotated[bool, _expert_gate] = False
EXPERT_GATE_OPTION: Annotated[LayerGateOptions | None, _expert_gate] = GATE_OPTION
EXPERT_GATE_ACTIVATION: Annotated[ActivationOptions | None, _expert_gate] = (
    GATE_ACTIVATION
)
_expert_gate_stack = _expert_gate.child("Expert Gate Stack Options")
# If False, expert gate stack options inherit the expert stack options.
EXPERT_GATE_STACK_INDEPENDENT_FLAG: Annotated[bool, _expert_gate_stack] = False
EXPERT_GATE_STACK_HIDDEN_DIM: Annotated[int | None, _expert_gate_stack] = None
EXPERT_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _expert_gate_stack
] = None
EXPERT_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _expert_gate_stack
] = None
EXPERT_GATE_STACK_NUM_LAYERS: Annotated[int | None, _expert_gate_stack] = None
EXPERT_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _expert_gate_stack
] = GATE_STACK_ACTIVATION
EXPERT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _expert_gate_stack
] = None
EXPERT_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _expert_gate_stack] = False
EXPERT_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _expert_gate_stack] = None
EXPERT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _expert_gate_stack
] = None
EXPERT_GATE_STACK_DROPOUT_PROBABILITY: Annotated[float | None, _expert_gate_stack] = (
    None
)
EXPERT_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _expert_gate_stack
] = None
EXPERT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _expert_gate_stack
] = GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
EXPERT_GATE_STACK_BIAS_FLAG: Annotated[bool | None, _expert_gate_stack] = (
    GATE_STACK_BIAS_FLAG
)

_expert_halting = _mixture_of_experts_model.child("Expert Halting Options")
# If `EXPERT_STACK_HALTING_FLAG` is False, the expert halting parameters are ignored.
EXPERT_STACK_HALTING_FLAG: Annotated[bool, _expert_halting] = False
EXPERT_HALTING_OPTION: Annotated[type[HaltingConfig], _expert_halting] = (
    StickBreakingConfig
)
EXPERT_HALTING_THRESHOLD: Annotated[float, _expert_halting] = HALTING_THRESHOLD
EXPERT_HALTING_DROPOUT: Annotated[float, _expert_halting] = HALTING_DROPOUT
EXPERT_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _expert_halting
] = HALTING_HIDDEN_STATE_MODE
EXPERT_HALTING_OUTPUT_DIM: Annotated[int, _expert_halting] = HALTING_OUTPUT_DIM
_expert_halting_stack = _expert_halting.child("Expert Halting Stack Options")
# If False, expert halting stack options inherit the expert stack options.
EXPERT_HALTING_STACK_INDEPENDENT_FLAG: Annotated[bool, _expert_halting_stack] = False
EXPERT_HALTING_STACK_HIDDEN_DIM: Annotated[int | None, _expert_halting_stack] = None
EXPERT_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _expert_halting_stack
] = HALTING_STACK_LAYER_NORM_POSITION
EXPERT_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _expert_halting_stack
] = HALTING_STACK_NORMALIZATION
EXPERT_HALTING_STACK_NUM_LAYERS: Annotated[int | None, _expert_halting_stack] = None
EXPERT_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _expert_halting_stack
] = None
EXPERT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _expert_halting_stack
] = None
EXPERT_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _expert_halting_stack] = False
EXPERT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _expert_halting_stack
] = None
EXPERT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _expert_halting_stack
] = None
EXPERT_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _expert_halting_stack
] = None
EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _expert_halting_stack
] = HALTING_STACK_LAST_LAYER_BIAS_OPTION
EXPERT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _expert_halting_stack
] = None
EXPERT_HALTING_STACK_BIAS_FLAG: Annotated[bool | None, _expert_halting_stack] = None

_expert_memory = _mixture_of_experts_model.child("Expert Memory Options")
# If `EXPERT_MEMORY_FLAG` is False, the expert memory parameters are ignored.
EXPERT_MEMORY_FLAG: Annotated[bool, _expert_memory] = False
EXPERT_MEMORY_OPTION: Annotated[type[DynamicMemoryConfig], _expert_memory] = (
    MEMORY_OPTION
)
EXPERT_MEMORY_POSITION_OPTION: Annotated[MemoryPositionOptions, _expert_memory] = (
    MEMORY_POSITION_OPTION
)
EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: Annotated[
    float | None, _expert_memory
] = MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
EXPERT_MEMORY_NUM_MEMORY_SLOTS: Annotated[int, _expert_memory] = 4
EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: Annotated[
    int | None, _expert_memory
] = MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
_expert_memory_stack = _expert_memory.child("Expert Memory Stack Options")
# If False, expert memory stack options inherit the expert stack options.
EXPERT_MEMORY_STACK_INDEPENDENT_FLAG: Annotated[bool, _expert_memory_stack] = False
EXPERT_MEMORY_STACK_HIDDEN_DIM: Annotated[int | None, _expert_memory_stack] = None
EXPERT_MEMORY_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _expert_memory_stack
] = None
EXPERT_MEMORY_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _expert_memory_stack
] = None
EXPERT_MEMORY_STACK_NUM_LAYERS: Annotated[int | None, _expert_memory_stack] = None
EXPERT_MEMORY_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _expert_memory_stack
] = None
EXPERT_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _expert_memory_stack
] = None
EXPERT_MEMORY_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _expert_memory_stack] = False
EXPERT_MEMORY_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _expert_memory_stack] = (
    None
)
EXPERT_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _expert_memory_stack
] = None
EXPERT_MEMORY_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _expert_memory_stack
] = None
EXPERT_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _expert_memory_stack
] = None
EXPERT_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _expert_memory_stack
] = None
EXPERT_MEMORY_STACK_BIAS_FLAG: Annotated[bool | None, _expert_memory_stack] = None

_expert_recurrent_layer = _mixture_of_experts_model.child(
    "Expert Recurrent Layer Options"
)
# If `EXPERT_RECURRENT_FLAG` is False, expert recurrence is disabled.
EXPERT_RECURRENT_FLAG: Annotated[bool, _expert_recurrent_layer] = False
EXPERT_RECURRENT_MAX_STEPS: Annotated[int, _expert_recurrent_layer] = (
    RECURRENT_MAX_STEPS
)
EXPERT_RECURRENT_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _expert_recurrent_layer
] = RECURRENT_LAYER_NORM_POSITION
EXPERT_RECURRENT_NORMALIZATION: Annotated[
    NormalizationOptions, _expert_recurrent_layer
] = RECURRENT_NORMALIZATION

_expert_recurrent_gate = _expert_recurrent_layer.child("Expert Recurrent Gate Options")
EXPERT_RECURRENT_STACK_GATE_FLAG: Annotated[bool, _expert_recurrent_gate] = False
EXPERT_RECURRENT_GATE_OPTION: Annotated[
    LayerGateOptions | None, _expert_recurrent_gate
] = RECURRENT_GATE_OPTION
EXPERT_RECURRENT_GATE_ACTIVATION: Annotated[
    ActivationOptions | None, _expert_recurrent_gate
] = RECURRENT_GATE_ACTIVATION
_expert_recurrent_gate_stack = _expert_recurrent_gate.child(
    "Expert Recurrent Gate Stack Options"
)
# If False, expert recurrent gate stack options inherit the expert stack options.
EXPERT_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _expert_recurrent_gate_stack
] = False
EXPERT_RECURRENT_GATE_STACK_HIDDEN_DIM: Annotated[
    int | None, _expert_recurrent_gate_stack
] = None
EXPERT_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _expert_recurrent_gate_stack
] = None
EXPERT_RECURRENT_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _expert_recurrent_gate_stack
] = None
EXPERT_RECURRENT_GATE_STACK_NUM_LAYERS: Annotated[
    int | None, _expert_recurrent_gate_stack
] = None
EXPERT_RECURRENT_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _expert_recurrent_gate_stack
] = None
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _expert_recurrent_gate_stack
] = None
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _expert_recurrent_gate_stack
] = False
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _expert_recurrent_gate_stack
] = None
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _expert_recurrent_gate_stack
] = None
EXPERT_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _expert_recurrent_gate_stack
] = None
EXPERT_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _expert_recurrent_gate_stack
] = None
EXPERT_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _expert_recurrent_gate_stack
] = None
EXPERT_RECURRENT_GATE_STACK_BIAS_FLAG: Annotated[
    bool | None, _expert_recurrent_gate_stack
] = None

_expert_recurrent_halting = _expert_recurrent_layer.child(
    "Expert Recurrent Halting Options"
)
EXPERT_RECURRENT_STACK_HALTING_FLAG: Annotated[bool, _expert_recurrent_halting] = False
EXPERT_RECURRENT_HALTING_OPTION: Annotated[
    type[HaltingConfig], _expert_recurrent_halting
] = StickBreakingConfig
EXPERT_RECURRENT_HALTING_THRESHOLD: Annotated[float, _expert_recurrent_halting] = (
    RECURRENT_HALTING_THRESHOLD
)
EXPERT_RECURRENT_HALTING_DROPOUT: Annotated[float, _expert_recurrent_halting] = (
    RECURRENT_HALTING_DROPOUT
)
EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _expert_recurrent_halting
] = RECURRENT_HALTING_HIDDEN_STATE_MODE
_expert_recurrent_halting_stack = _expert_recurrent_halting.child(
    "Expert Recurrent Halting Stack Options"
)
# If False, expert recurrent halting stack options inherit the expert stack options.
EXPERT_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _expert_recurrent_halting_stack
] = False
EXPERT_RECURRENT_HALTING_STACK_HIDDEN_DIM: Annotated[
    int | None, _expert_recurrent_halting_stack
] = None
EXPERT_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _expert_recurrent_halting_stack
] = None
EXPERT_RECURRENT_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _expert_recurrent_halting_stack
] = None
EXPERT_RECURRENT_HALTING_STACK_NUM_LAYERS: Annotated[
    int | None, _expert_recurrent_halting_stack
] = None
EXPERT_RECURRENT_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _expert_recurrent_halting_stack
] = None
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _expert_recurrent_halting_stack
] = None
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _expert_recurrent_halting_stack
] = False
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _expert_recurrent_halting_stack
] = None
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _expert_recurrent_halting_stack
] = None
EXPERT_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _expert_recurrent_halting_stack
] = None
EXPERT_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _expert_recurrent_halting_stack
] = None
EXPERT_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _expert_recurrent_halting_stack
] = None
EXPERT_RECURRENT_HALTING_STACK_BIAS_FLAG: Annotated[
    bool | None, _expert_recurrent_halting_stack
] = None

_sampler_model = Section(("Sampler Model Options",))
SAMPLER_THRESHOLD: Annotated[float, _sampler_model] = 0.0
SAMPLER_FILTER_ABOVE_THRESHOLD: Annotated[bool, _sampler_model] = False
SAMPLER_NUM_TOPK_SAMPLES: Annotated[int, _sampler_model] = 0
SAMPLER_NORMALIZE_PROBABILITIES_FLAG: Annotated[bool, _sampler_model] = True
SAMPLER_NOISY_TOPK_FLAG: Annotated[bool, _sampler_model] = False
SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: Annotated[float, _sampler_model] = 0.0
SAMPLER_SWITCH_LOSS_WEIGHT: Annotated[float, _sampler_model] = 0.1
SAMPLER_ZERO_CENTRED_LOSS_WEIGHT: Annotated[float, _sampler_model] = 0.0
SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT: Annotated[float, _sampler_model] = 0.0

_router = _sampler_model.child("Router Options")
ROUTER_NOISY_TOPK_FLAG: Annotated[bool, _router] = False

_router_stack = _router.child("Router Stack Options")
ROUTER_STACK_HIDDEN_DIM: Annotated[int, _router_stack] = HIDDEN_DIM
ROUTER_STACK_NUM_LAYERS: Annotated[int, _router_stack] = 2
ROUTER_STACK_ACTIVATION: Annotated[ActivationOptions, _router_stack] = (
    ActivationOptions.GELU
)
ROUTER_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_stack
] = None
ROUTER_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _router_stack] = False
ROUTER_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _router_stack] = None
ROUTER_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _router_stack] = None
ROUTER_STACK_DROPOUT_PROBABILITY: Annotated[float, _router_stack] = 0.0
ROUTER_STACK_LAYER_NORM_POSITION: Annotated[LayerNormPositionOptions, _router_stack] = (
    LayerNormPositionOptions.BEFORE
)
ROUTER_STACK_NORMALIZATION: Annotated[NormalizationOptions, _router_stack] = (
    NormalizationOptions.RMS_NORM
)
ROUTER_STACK_LAST_LAYER_BIAS_OPTION: Annotated[LastLayerBiasOptions, _router_stack] = (
    LastLayerBiasOptions.DEFAULT
)
ROUTER_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool, _router_stack] = False
ROUTER_BIAS_FLAG: Annotated[bool, _router_stack] = True

_router_gate = _router.child("Router Gate Options")
# If `ROUTER_STACK_GATE_FLAG` is False, the router gate parameters are ignored.
ROUTER_STACK_GATE_FLAG: Annotated[bool, _router_gate] = False
ROUTER_GATE_OPTION: Annotated[LayerGateOptions | None, _router_gate] = (
    EXPERT_GATE_OPTION
)
ROUTER_GATE_ACTIVATION: Annotated[ActivationOptions | None, _router_gate] = (
    EXPERT_GATE_ACTIVATION
)
_router_gate_stack = _router_gate.child("Router Gate Stack Options")
# If False, router gate stack options inherit layer stack submodule options.
ROUTER_GATE_STACK_INDEPENDENT_FLAG: Annotated[bool, _router_gate_stack] = False
ROUTER_GATE_STACK_HIDDEN_DIM: Annotated[int | None, _router_gate_stack] = None
ROUTER_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_gate_stack
] = None
ROUTER_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_gate_stack
] = None
ROUTER_GATE_STACK_NUM_LAYERS: Annotated[int | None, _router_gate_stack] = None
ROUTER_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_gate_stack
] = EXPERT_GATE_STACK_ACTIVATION
ROUTER_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_gate_stack
] = None
ROUTER_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _router_gate_stack] = False
ROUTER_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _router_gate_stack] = None
ROUTER_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_gate_stack
] = None
ROUTER_GATE_STACK_DROPOUT_PROBABILITY: Annotated[float | None, _router_gate_stack] = (
    None
)
ROUTER_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_gate_stack
] = None
ROUTER_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_gate_stack
] = EXPERT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
ROUTER_GATE_STACK_BIAS_FLAG: Annotated[bool | None, _router_gate_stack] = (
    EXPERT_GATE_STACK_BIAS_FLAG
)

_router_halting = _router.child("Router Halting Options")
# If `ROUTER_STACK_HALTING_FLAG` is False, the router halting parameters are ignored.
ROUTER_STACK_HALTING_FLAG: Annotated[bool, _router_halting] = False
ROUTER_HALTING_OPTION: Annotated[type[HaltingConfig], _router_halting] = (
    StickBreakingConfig
)
ROUTER_HALTING_THRESHOLD: Annotated[float, _router_halting] = EXPERT_HALTING_THRESHOLD
ROUTER_HALTING_DROPOUT: Annotated[float, _router_halting] = EXPERT_HALTING_DROPOUT
ROUTER_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _router_halting
] = EXPERT_HALTING_HIDDEN_STATE_MODE
ROUTER_HALTING_OUTPUT_DIM: Annotated[int, _router_halting] = EXPERT_HALTING_OUTPUT_DIM
_router_halting_stack = _router_halting.child("Router Halting Stack Options")
# If False, router halting stack options inherit layer stack submodule options.
ROUTER_HALTING_STACK_INDEPENDENT_FLAG: Annotated[bool, _router_halting_stack] = False
ROUTER_HALTING_STACK_HIDDEN_DIM: Annotated[int | None, _router_halting_stack] = None
ROUTER_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_halting_stack
] = EXPERT_HALTING_STACK_LAYER_NORM_POSITION
ROUTER_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_halting_stack
] = EXPERT_HALTING_STACK_NORMALIZATION
ROUTER_HALTING_STACK_NUM_LAYERS: Annotated[int | None, _router_halting_stack] = None
ROUTER_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_halting_stack
] = None
ROUTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_halting_stack
] = None
ROUTER_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _router_halting_stack] = False
ROUTER_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _router_halting_stack
] = None
ROUTER_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_halting_stack
] = None
ROUTER_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_halting_stack
] = None
ROUTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_halting_stack
] = EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
ROUTER_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_halting_stack
] = None
ROUTER_HALTING_STACK_BIAS_FLAG: Annotated[bool | None, _router_halting_stack] = None

_router_memory = _router.child("Router Memory Options")
# If `ROUTER_MEMORY_FLAG` is False, the router memory parameters are ignored.
ROUTER_MEMORY_FLAG: Annotated[bool, _router_memory] = False
ROUTER_MEMORY_OPTION: Annotated[type[DynamicMemoryConfig], _router_memory] = (
    EXPERT_MEMORY_OPTION
)
ROUTER_MEMORY_POSITION_OPTION: Annotated[MemoryPositionOptions, _router_memory] = (
    EXPERT_MEMORY_POSITION_OPTION
)
ROUTER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: Annotated[
    float | None, _router_memory
] = EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
ROUTER_MEMORY_NUM_MEMORY_SLOTS: Annotated[int, _router_memory] = 4
ROUTER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: Annotated[
    int | None, _router_memory
] = EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
_router_memory_stack = _router_memory.child("Router Memory Stack Options")
# If False, router memory stack options inherit layer stack submodule options.
ROUTER_MEMORY_STACK_INDEPENDENT_FLAG: Annotated[bool, _router_memory_stack] = False
ROUTER_MEMORY_STACK_HIDDEN_DIM: Annotated[int | None, _router_memory_stack] = None
ROUTER_MEMORY_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_memory_stack
] = None
ROUTER_MEMORY_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_memory_stack
] = None
ROUTER_MEMORY_STACK_NUM_LAYERS: Annotated[int | None, _router_memory_stack] = None
ROUTER_MEMORY_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_memory_stack
] = None
ROUTER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_memory_stack
] = None
ROUTER_MEMORY_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _router_memory_stack] = False
ROUTER_MEMORY_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _router_memory_stack] = (
    None
)
ROUTER_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_memory_stack
] = None
ROUTER_MEMORY_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_memory_stack
] = None
ROUTER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_memory_stack
] = None
ROUTER_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_memory_stack
] = None
ROUTER_MEMORY_STACK_BIAS_FLAG: Annotated[bool | None, _router_memory_stack] = None

_router_recurrent_layer = _router.child("Router Recurrent Layer Options")
# If `ROUTER_RECURRENT_FLAG` is False, router recurrence is disabled.
ROUTER_RECURRENT_FLAG: Annotated[bool, _router_recurrent_layer] = False
ROUTER_RECURRENT_MAX_STEPS: Annotated[int, _router_recurrent_layer] = (
    EXPERT_RECURRENT_MAX_STEPS
)
ROUTER_RECURRENT_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _router_recurrent_layer
] = EXPERT_RECURRENT_LAYER_NORM_POSITION
ROUTER_RECURRENT_NORMALIZATION: Annotated[
    NormalizationOptions, _router_recurrent_layer
] = EXPERT_RECURRENT_NORMALIZATION

_router_recurrent_gate = _router_recurrent_layer.child("Router Recurrent Gate Options")
ROUTER_RECURRENT_STACK_GATE_FLAG: Annotated[bool, _router_recurrent_gate] = False
ROUTER_RECURRENT_GATE_OPTION: Annotated[
    LayerGateOptions | None, _router_recurrent_gate
] = EXPERT_RECURRENT_GATE_OPTION
ROUTER_RECURRENT_GATE_ACTIVATION: Annotated[
    ActivationOptions | None, _router_recurrent_gate
] = EXPERT_RECURRENT_GATE_ACTIVATION
_router_recurrent_gate_stack = _router_recurrent_gate.child(
    "Router Recurrent Gate Stack Options"
)
# If False, router recurrent gate stack options inherit layer stack submodule options.
ROUTER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _router_recurrent_gate_stack
] = False
ROUTER_RECURRENT_GATE_STACK_HIDDEN_DIM: Annotated[
    int | None, _router_recurrent_gate_stack
] = None
ROUTER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_recurrent_gate_stack
] = None
ROUTER_RECURRENT_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_recurrent_gate_stack
] = None
ROUTER_RECURRENT_GATE_STACK_NUM_LAYERS: Annotated[
    int | None, _router_recurrent_gate_stack
] = None
ROUTER_RECURRENT_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_recurrent_gate_stack
] = None
ROUTER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_recurrent_gate_stack
] = None
ROUTER_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _router_recurrent_gate_stack
] = False
ROUTER_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _router_recurrent_gate_stack
] = None
ROUTER_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_recurrent_gate_stack
] = None
ROUTER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_recurrent_gate_stack
] = None
ROUTER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_recurrent_gate_stack
] = None
ROUTER_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_recurrent_gate_stack
] = None
ROUTER_RECURRENT_GATE_STACK_BIAS_FLAG: Annotated[
    bool | None, _router_recurrent_gate_stack
] = None

_router_recurrent_halting = _router_recurrent_layer.child(
    "Router Recurrent Halting Options"
)
ROUTER_RECURRENT_STACK_HALTING_FLAG: Annotated[bool, _router_recurrent_halting] = False
ROUTER_RECURRENT_HALTING_OPTION: Annotated[
    type[HaltingConfig], _router_recurrent_halting
] = StickBreakingConfig
ROUTER_RECURRENT_HALTING_THRESHOLD: Annotated[float, _router_recurrent_halting] = (
    EXPERT_RECURRENT_HALTING_THRESHOLD
)
ROUTER_RECURRENT_HALTING_DROPOUT: Annotated[float, _router_recurrent_halting] = (
    EXPERT_RECURRENT_HALTING_DROPOUT
)
ROUTER_RECURRENT_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _router_recurrent_halting
] = EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE
_router_recurrent_halting_stack = _router_recurrent_halting.child(
    "Router Recurrent Halting Stack Options"
)
# If False, router recurrent halting stack options inherit layer stack submodule options.
ROUTER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _router_recurrent_halting_stack
] = False
ROUTER_RECURRENT_HALTING_STACK_HIDDEN_DIM: Annotated[
    int | None, _router_recurrent_halting_stack
] = None
ROUTER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_recurrent_halting_stack
] = None
ROUTER_RECURRENT_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_recurrent_halting_stack
] = None
ROUTER_RECURRENT_HALTING_STACK_NUM_LAYERS: Annotated[
    int | None, _router_recurrent_halting_stack
] = None
ROUTER_RECURRENT_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_recurrent_halting_stack
] = None
ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_recurrent_halting_stack
] = None
ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _router_recurrent_halting_stack
] = False
ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _router_recurrent_halting_stack
] = None
ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_recurrent_halting_stack
] = None
ROUTER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_recurrent_halting_stack
] = None
ROUTER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_recurrent_halting_stack
] = None
ROUTER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_recurrent_halting_stack
] = None
ROUTER_RECURRENT_HALTING_STACK_BIAS_FLAG: Annotated[
    bool | None, _router_recurrent_halting_stack
] = None

_adaptive_generator_stack = Section(("Adaptive Generator Stack Options",))
ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: Annotated[int, _adaptive_generator_stack] = (
    HIDDEN_DIM
)
ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _adaptive_generator_stack
] = LayerNormPositionOptions.BEFORE
ADAPTIVE_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions, _adaptive_generator_stack
] = NormalizationOptions.RMS_NORM
ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: Annotated[int, _adaptive_generator_stack] = 2
ADAPTIVE_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions, _adaptive_generator_stack
] = ActivationOptions.GELU
ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _adaptive_generator_stack
] = None
ADAPTIVE_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _adaptive_generator_stack
] = False
ADAPTIVE_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _adaptive_generator_stack
] = None
ADAPTIVE_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _adaptive_generator_stack
] = None
ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float, _adaptive_generator_stack
] = 0.0
ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions, _adaptive_generator_stack
] = LastLayerBiasOptions.DEFAULT
ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool, _adaptive_generator_stack
] = False
ADAPTIVE_GENERATOR_STACK_BIAS_FLAG: Annotated[bool, _adaptive_generator_stack] = True

_weight_generator = Section(("Weight Generator Options",))
# If `WEIGHT_OPTION_FLAG` is False, the expert weight parameters below are ignored.
WEIGHT_OPTION_FLAG: Annotated[bool, _weight_generator] = False
WEIGHT_OPTION: Annotated[type[DynamicWeightConfig] | None, _weight_generator] = None
GENERATOR_DEPTH: Annotated[int, _weight_generator] = 3
WEIGHT_DECAY_SCHEDULE: Annotated[WeightDecayScheduleOptions, _weight_generator] = (
    WeightDecayScheduleOptions.DISABLED
)
WEIGHT_DECAY_RATE: Annotated[float, _weight_generator] = 0.0
WEIGHT_DECAY_WARMUP_BATCHES: Annotated[int, _weight_generator] = 0
WEIGHT_NORMALIZATION_OPTION: Annotated[
    WeightNormalizationOptions, _weight_generator
] = WeightNormalizationOptions.DISABLED
WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG: Annotated[bool, _weight_generator] = False
WEIGHT_NORMALIZATION_POSITION_OPTION: Annotated[
    WeightNormalizationPositionOptions, _weight_generator
] = WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
WEIGHT_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions, _weight_generator
] = BankExpansionFactorOptions.FACTOR_OF_THREE
_weight_generator_stack = _weight_generator.child("Weight Generator Stack Options")
# If False, weight generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[bool, _weight_generator_stack] = (
    False
)
WEIGHT_GENERATOR_STACK_HIDDEN_DIM: Annotated[int | None, _weight_generator_stack] = None
WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _weight_generator_stack
] = None
WEIGHT_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _weight_generator_stack
] = None
WEIGHT_GENERATOR_STACK_NUM_LAYERS: Annotated[int | None, _weight_generator_stack] = None
WEIGHT_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _weight_generator_stack
] = None
WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _weight_generator_stack
] = None
WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _weight_generator_stack] = (
    False
)
WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _weight_generator_stack
] = None
WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _weight_generator_stack
] = None
WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _weight_generator_stack
] = None
WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _weight_generator_stack
] = None
WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _weight_generator_stack
] = None
WEIGHT_GENERATOR_STACK_BIAS_FLAG: Annotated[bool | None, _weight_generator_stack] = None

_bias_generator = Section(("Bias Generator Options",))
# If `BIAS_OPTION_FLAG` is False, the expert bias parameters below are ignored.
BIAS_OPTION_FLAG: Annotated[bool, _bias_generator] = False
BIAS_OPTION: Annotated[type[DynamicBiasConfig] | None, _bias_generator] = None
BIAS_DECAY_SCHEDULE: Annotated[WeightDecayScheduleOptions, _bias_generator] = (
    WeightDecayScheduleOptions.DISABLED
)
BIAS_DECAY_RATE: Annotated[float, _bias_generator] = 0.0
BIAS_DECAY_WARMUP_BATCHES: Annotated[int, _bias_generator] = 0
BIAS_BANK_EXPANSION_FACTOR: Annotated[BankExpansionFactorOptions, _bias_generator] = (
    BankExpansionFactorOptions.FACTOR_OF_TWO
)
_bias_generator_stack = _bias_generator.child("Bias Generator Stack Options")
# If False, bias generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
BIAS_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[bool, _bias_generator_stack] = False
BIAS_GENERATOR_STACK_HIDDEN_DIM: Annotated[int | None, _bias_generator_stack] = None
BIAS_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _bias_generator_stack
] = None
BIAS_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _bias_generator_stack
] = None
BIAS_GENERATOR_STACK_NUM_LAYERS: Annotated[int | None, _bias_generator_stack] = None
BIAS_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _bias_generator_stack
] = None
BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _bias_generator_stack
] = None
BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _bias_generator_stack] = False
BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _bias_generator_stack
] = None
BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _bias_generator_stack
] = None
BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _bias_generator_stack
] = None
BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _bias_generator_stack
] = None
BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _bias_generator_stack
] = None
BIAS_GENERATOR_STACK_BIAS_FLAG: Annotated[bool | None, _bias_generator_stack] = None

_diagonal_generator = Section(("Diagonal Generator Options",))
# If `DIAGONAL_OPTION_FLAG` is False, the expert diagonal parameters below are
# ignored.
DIAGONAL_OPTION_FLAG: Annotated[bool, _diagonal_generator] = False
DIAGONAL_OPTION: Annotated[type[DynamicDiagonalConfig] | None, _diagonal_generator] = (
    None
)
_diagonal_generator_stack = _diagonal_generator.child(
    "Diagonal Generator Stack Options"
)
# If False, diagonal generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _diagonal_generator_stack
] = False
DIAGONAL_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _diagonal_generator_stack
] = None
DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _diagonal_generator_stack
] = None
DIAGONAL_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _diagonal_generator_stack
] = None
DIAGONAL_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _diagonal_generator_stack
] = None
DIAGONAL_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _diagonal_generator_stack
] = None
DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _diagonal_generator_stack
] = None
DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _diagonal_generator_stack
] = False
DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _diagonal_generator_stack
] = None
DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _diagonal_generator_stack
] = None
DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _diagonal_generator_stack
] = None
DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _diagonal_generator_stack
] = None
DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _diagonal_generator_stack
] = None
DIAGONAL_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _diagonal_generator_stack
] = None

_mask = Section(("Mask Options",))
# If `MASK_OPTION_FLAG` is False, the expert mask parameters below are ignored.
MASK_OPTION_FLAG: Annotated[bool, _mask] = False
ROW_MASK_OPTION: Annotated[type[AxisMaskConfig] | None, _mask] = None
MASK_THRESHOLD: Annotated[float, _mask] = 0.5
MASK_FLOOR: Annotated[float, _mask] = 0.0
MASK_TRANSITION_WIDTH: Annotated[float, _mask] = 0.1
MASK_SURROGATE_SCALE: Annotated[float, _mask] = 10.0
MASK_DIMENSION_OPTION: Annotated[MaskDimensionOptions, _mask] = (
    MaskDimensionOptions.COLUMN
)
_mask_stack = _mask.child("Mask Stack Options")
# If False, mask generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
MASK_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[bool, _mask_stack] = False
MASK_GENERATOR_STACK_HIDDEN_DIM: Annotated[int | None, _mask_stack] = None
MASK_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _mask_stack
] = None
MASK_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _mask_stack
] = None
MASK_GENERATOR_STACK_NUM_LAYERS: Annotated[int | None, _mask_stack] = None
MASK_GENERATOR_STACK_ACTIVATION: Annotated[ActivationOptions | None, _mask_stack] = None
MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _mask_stack
] = None
MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _mask_stack] = False
MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _mask_stack] = None
MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _mask_stack] = (
    None
)
MASK_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[float | None, _mask_stack] = None
MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _mask_stack
] = None
MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _mask_stack
] = None
MASK_GENERATOR_STACK_BIAS_FLAG: Annotated[bool | None, _mask_stack] = None


#########################################################################
# Sampler Model Options
## Router Options
_router_weight_generator = _router.child("Router Weight Generator Options")
# If `ROUTER_WEIGHT_OPTION_FLAG` is False, router weight parameters are ignored.
ROUTER_WEIGHT_OPTION_FLAG: Annotated[bool, _router_weight_generator] = False
ROUTER_WEIGHT_OPTION: Annotated[
    type[DynamicWeightConfig] | None, _router_weight_generator
] = None
ROUTER_GENERATOR_DEPTH: Annotated[int, _router_weight_generator] = GENERATOR_DEPTH
ROUTER_WEIGHT_DECAY_SCHEDULE: Annotated[
    WeightDecayScheduleOptions, _router_weight_generator
] = WEIGHT_DECAY_SCHEDULE
ROUTER_WEIGHT_DECAY_RATE: Annotated[float, _router_weight_generator] = WEIGHT_DECAY_RATE
ROUTER_WEIGHT_DECAY_WARMUP_BATCHES: Annotated[int, _router_weight_generator] = (
    WEIGHT_DECAY_WARMUP_BATCHES
)
ROUTER_WEIGHT_NORMALIZATION_OPTION: Annotated[
    WeightNormalizationOptions, _router_weight_generator
] = WEIGHT_NORMALIZATION_OPTION
ROUTER_WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG: Annotated[
    bool, _router_weight_generator
] = WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG
ROUTER_WEIGHT_NORMALIZATION_POSITION_OPTION: Annotated[
    WeightNormalizationPositionOptions, _router_weight_generator
] = WEIGHT_NORMALIZATION_POSITION_OPTION
ROUTER_WEIGHT_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions, _router_weight_generator
] = WEIGHT_BANK_EXPANSION_FACTOR
_router_weight_generator_stack = _router_weight_generator.child(
    "Router Weight Generator Stack Options"
)
# If False, router weight generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
ROUTER_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _router_weight_generator_stack
] = False
ROUTER_WEIGHT_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _router_weight_generator_stack
] = None
ROUTER_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_weight_generator_stack
] = None
ROUTER_WEIGHT_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_weight_generator_stack
] = None
ROUTER_WEIGHT_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _router_weight_generator_stack
] = None
ROUTER_WEIGHT_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_weight_generator_stack
] = None
ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_weight_generator_stack
] = None
ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _router_weight_generator_stack
] = False
ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _router_weight_generator_stack
] = None
ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_weight_generator_stack
] = None
ROUTER_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_weight_generator_stack
] = None
ROUTER_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_weight_generator_stack
] = None
ROUTER_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_weight_generator_stack
] = None
ROUTER_WEIGHT_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _router_weight_generator_stack
] = None

_router_bias_generator = _router.child("Router Bias Generator Options")
# If `ROUTER_BIAS_OPTION_FLAG` is False, router bias parameters are ignored.
ROUTER_BIAS_OPTION_FLAG: Annotated[bool, _router_bias_generator] = False
ROUTER_BIAS_OPTION: Annotated[
    type[DynamicBiasConfig] | None, _router_bias_generator
] = None
ROUTER_BIAS_DECAY_SCHEDULE: Annotated[
    WeightDecayScheduleOptions, _router_bias_generator
] = BIAS_DECAY_SCHEDULE
ROUTER_BIAS_DECAY_RATE: Annotated[float, _router_bias_generator] = BIAS_DECAY_RATE
ROUTER_BIAS_DECAY_WARMUP_BATCHES: Annotated[int, _router_bias_generator] = (
    BIAS_DECAY_WARMUP_BATCHES
)
ROUTER_BIAS_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions, _router_bias_generator
] = BIAS_BANK_EXPANSION_FACTOR
_router_bias_generator_stack = _router_bias_generator.child(
    "Router Bias Generator Stack Options"
)
# If False, router bias generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
ROUTER_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _router_bias_generator_stack
] = False
ROUTER_BIAS_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _router_bias_generator_stack
] = None
ROUTER_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_bias_generator_stack
] = None
ROUTER_BIAS_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_bias_generator_stack
] = None
ROUTER_BIAS_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _router_bias_generator_stack
] = None
ROUTER_BIAS_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_bias_generator_stack
] = None
ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_bias_generator_stack
] = None
ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _router_bias_generator_stack
] = False
ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _router_bias_generator_stack
] = None
ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_bias_generator_stack
] = None
ROUTER_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_bias_generator_stack
] = None
ROUTER_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_bias_generator_stack
] = None
ROUTER_BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_bias_generator_stack
] = None
ROUTER_BIAS_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _router_bias_generator_stack
] = None

_router_diagonal_generator = _router.child("Router Diagonal Generator Options")
# If `ROUTER_DIAGONAL_OPTION_FLAG` is False, router diagonal parameters are ignored.
ROUTER_DIAGONAL_OPTION_FLAG: Annotated[bool, _router_diagonal_generator] = False
ROUTER_DIAGONAL_OPTION: Annotated[
    type[DynamicDiagonalConfig] | None, _router_diagonal_generator
] = None
_router_diagonal_generator_stack = _router_diagonal_generator.child(
    "Router Diagonal Generator Stack Options"
)
# If False, router diagonal generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
ROUTER_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _router_diagonal_generator_stack
] = False
ROUTER_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _router_diagonal_generator_stack
] = None
ROUTER_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_diagonal_generator_stack
] = None
ROUTER_DIAGONAL_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_diagonal_generator_stack
] = None
ROUTER_DIAGONAL_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _router_diagonal_generator_stack
] = None
ROUTER_DIAGONAL_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_diagonal_generator_stack
] = None
ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_diagonal_generator_stack
] = None
ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _router_diagonal_generator_stack
] = False
ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _router_diagonal_generator_stack
] = None
ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_diagonal_generator_stack
] = None
ROUTER_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_diagonal_generator_stack
] = None
ROUTER_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_diagonal_generator_stack
] = None
ROUTER_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_diagonal_generator_stack
] = None
ROUTER_DIAGONAL_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _router_diagonal_generator_stack
] = None

_router_mask = _router.child("Router Mask Options")
# If `ROUTER_MASK_OPTION_FLAG` is False, router mask parameters are ignored.
ROUTER_MASK_OPTION_FLAG: Annotated[bool, _router_mask] = False
ROUTER_ROW_MASK_OPTION: Annotated[type[AxisMaskConfig] | None, _router_mask] = None
ROUTER_MASK_THRESHOLD: Annotated[float, _router_mask] = MASK_THRESHOLD
ROUTER_MASK_FLOOR: Annotated[float, _router_mask] = MASK_FLOOR
ROUTER_MASK_TRANSITION_WIDTH: Annotated[float, _router_mask] = MASK_TRANSITION_WIDTH
ROUTER_MASK_SURROGATE_SCALE: Annotated[float, _router_mask] = MASK_SURROGATE_SCALE
ROUTER_MASK_DIMENSION_OPTION: Annotated[MaskDimensionOptions, _router_mask] = (
    MASK_DIMENSION_OPTION
)
_router_mask_stack = _router_mask.child("Router Mask Stack Options")
# If False, router mask generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
ROUTER_MASK_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[bool, _router_mask_stack] = (
    False
)
ROUTER_MASK_GENERATOR_STACK_HIDDEN_DIM: Annotated[int | None, _router_mask_stack] = None
ROUTER_MASK_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_mask_stack
] = None
ROUTER_MASK_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_mask_stack
] = None
ROUTER_MASK_GENERATOR_STACK_NUM_LAYERS: Annotated[int | None, _router_mask_stack] = None
ROUTER_MASK_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_mask_stack
] = None
ROUTER_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_mask_stack
] = None
ROUTER_MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _router_mask_stack] = (
    False
)
ROUTER_MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _router_mask_stack
] = None
ROUTER_MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_mask_stack
] = None
ROUTER_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_mask_stack
] = None
ROUTER_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_mask_stack
] = None
ROUTER_MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_mask_stack
] = None
ROUTER_MASK_GENERATOR_STACK_BIAS_FLAG: Annotated[bool | None, _router_mask_stack] = None

RUNTIME_VALUE_CONSTRAINTS: Annotated[RuntimeValueConstraints, _router_mask_stack] = (
    positive_runtime_fields(
        "HIDDEN_DIM",
        "SOFT_MOE_SLOTS_PER_EXPERT",
        "SOFT_MOE_SCORING_HIDDEN_DIM",
        "SOFT_MOE_SCORING_NUM_LAYERS",
    )
)


# ==============================================================================
_main_adaptive_generation = Section(("Main Adaptive Generation",))
# ==============================================================================
WEIGHT_INPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_NUM_EXPERTS: Annotated[int | None, _main_adaptive_generation] = None
BIAS_MIXTURE_NUM_EXPERTS: Annotated[int | None, _main_adaptive_generation] = None
WEIGHT_MIXTURE_TOP_K: Annotated[int | None, _main_adaptive_generation] = None
BIAS_MIXTURE_TOP_K: Annotated[int | None, _main_adaptive_generation] = None
WEIGHT_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool | None, _main_adaptive_generation
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _main_adaptive_generation
] = False
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _main_adaptive_generation
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _main_adaptive_generation
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _main_adaptive_generation
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _main_adaptive_generation
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _main_adaptive_generation
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _main_adaptive_generation
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _main_adaptive_generation
] = False
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _main_adaptive_generation
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _main_adaptive_generation
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _main_adaptive_generation
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _main_adaptive_generation
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _main_adaptive_generation
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _main_adaptive_generation
] = False
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _main_adaptive_generation
] = False
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _main_adaptive_generation
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _main_adaptive_generation
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _main_adaptive_generation
] = False
WEIGHT_COEFFICIENT_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _main_adaptive_generation
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _main_adaptive_generation
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _main_adaptive_generation
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _main_adaptive_generation
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _main_adaptive_generation
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _main_adaptive_generation
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _main_adaptive_generation
] = False
WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _main_adaptive_generation
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _main_adaptive_generation
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _main_adaptive_generation
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _main_adaptive_generation
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _main_adaptive_generation
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _main_adaptive_generation
] = False
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _main_adaptive_generation
] = False
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _main_adaptive_generation
] = False
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _main_adaptive_generation
] = False
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _main_adaptive_generation
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _main_adaptive_generation
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _main_adaptive_generation
] = None

_main_grouping = Section(("Main Grouping",))
GROUPING_SCOPE: Annotated[
    AdaptiveParameterGroupingScopeOptions | None, _main_grouping
] = None
GROUP_COUNT: Annotated[int | None, _main_grouping] = None
CHUNK_SIZE: Annotated[int | None, _main_grouping] = None
GROUPING_SEQUENCE_LENGTH: Annotated[int | None, _main_grouping] = None
GROUPING_INPUT_ORDER: Annotated[
    AdaptiveParameterInputOrderOptions | None, _main_grouping
] = None
GROUPING_METHOD: Annotated[type[GroupingConfig] | None, _main_grouping] = None
GROUPING_SUMMARY_NORMALIZATION: Annotated[
    SummaryNormalizationOptions | None, _main_grouping
] = None
GROUPING_MODEL_CONFIG: Annotated[ConfigBase | None, _main_grouping] = None
GROUPING_ATTENTION_HIDDEN_DIM: Annotated[int | None, _main_grouping] = None
GROUPING_RMS_NORM_EPSILON: Annotated[float | None, _main_grouping] = None


# ==============================================================================
_router_adaptive_generation = Section(("Router Adaptive Generation",))
# ==============================================================================
ROUTER_WEIGHT_INPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_NUM_EXPERTS: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_NUM_EXPERTS: Annotated[int | None, _router_adaptive_generation] = (
    None
)
ROUTER_WEIGHT_MIXTURE_TOP_K: Annotated[int | None, _router_adaptive_generation] = None
ROUTER_BIAS_MIXTURE_TOP_K: Annotated[int | None, _router_adaptive_generation] = None
ROUTER_WEIGHT_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _router_adaptive_generation
] = False
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _router_adaptive_generation
] = False
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _router_adaptive_generation
] = False
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _router_adaptive_generation
] = False
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _router_adaptive_generation
] = False
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _router_adaptive_generation
] = False
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _router_adaptive_generation
] = False
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _router_adaptive_generation
] = False
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _router_adaptive_generation
] = False
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _router_adaptive_generation
] = False
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _router_adaptive_generation
] = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _router_adaptive_generation
] = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _router_adaptive_generation
] = None

_router_grouping = Section(("Router Grouping",))
ROUTER_GROUPING_SCOPE: Annotated[
    AdaptiveParameterGroupingScopeOptions | None, _router_grouping
] = None
ROUTER_GROUP_COUNT: Annotated[int | None, _router_grouping] = None
ROUTER_CHUNK_SIZE: Annotated[int | None, _router_grouping] = None
ROUTER_GROUPING_SEQUENCE_LENGTH: Annotated[int | None, _router_grouping] = None
ROUTER_GROUPING_INPUT_ORDER: Annotated[
    AdaptiveParameterInputOrderOptions | None, _router_grouping
] = None
ROUTER_GROUPING_METHOD: Annotated[type[GroupingConfig] | None, _router_grouping] = None
ROUTER_GROUPING_SUMMARY_NORMALIZATION: Annotated[
    SummaryNormalizationOptions | None, _router_grouping
] = None
ROUTER_GROUPING_MODEL_CONFIG: Annotated[ConfigBase | None, _router_grouping] = None
ROUTER_GROUPING_ATTENTION_HIDDEN_DIM: Annotated[int | None, _router_grouping] = None
ROUTER_GROUPING_RMS_NORM_EPSILON: Annotated[float | None, _router_grouping] = None


_attn_grouping = Section(("Attn Grouping",))
ATTN_GROUPING_SCOPE: Annotated[
    AdaptiveParameterGroupingScopeOptions | None, _attn_grouping
] = None
ATTN_GROUP_COUNT: Annotated[int | None, _attn_grouping] = None
ATTN_CHUNK_SIZE: Annotated[int | None, _attn_grouping] = None
ATTN_GROUPING_SEQUENCE_LENGTH: Annotated[int | None, _attn_grouping] = None
ATTN_GROUPING_INPUT_ORDER: Annotated[
    AdaptiveParameterInputOrderOptions | None, _attn_grouping
] = None
ATTN_GROUPING_METHOD: Annotated[type[GroupingConfig] | None, _attn_grouping] = None
ATTN_GROUPING_SUMMARY_NORMALIZATION: Annotated[
    SummaryNormalizationOptions | None, _attn_grouping
] = None
ATTN_GROUPING_MODEL_CONFIG: Annotated[ConfigBase | None, _attn_grouping] = None
ATTN_GROUPING_ATTENTION_HIDDEN_DIM: Annotated[int | None, _attn_grouping] = None
ATTN_GROUPING_RMS_NORM_EPSILON: Annotated[float | None, _attn_grouping] = None


_ff_grouping = Section(("Ff Grouping",))
FF_GROUPING_SCOPE: Annotated[
    AdaptiveParameterGroupingScopeOptions | None, _ff_grouping
] = None
FF_GROUP_COUNT: Annotated[int | None, _ff_grouping] = None
FF_CHUNK_SIZE: Annotated[int | None, _ff_grouping] = None
FF_GROUPING_SEQUENCE_LENGTH: Annotated[int | None, _ff_grouping] = None
FF_GROUPING_INPUT_ORDER: Annotated[
    AdaptiveParameterInputOrderOptions | None, _ff_grouping
] = None
FF_GROUPING_METHOD: Annotated[type[GroupingConfig] | None, _ff_grouping] = None
FF_GROUPING_SUMMARY_NORMALIZATION: Annotated[
    SummaryNormalizationOptions | None, _ff_grouping
] = None
FF_GROUPING_MODEL_CONFIG: Annotated[ConfigBase | None, _ff_grouping] = None
FF_GROUPING_ATTENTION_HIDDEN_DIM: Annotated[int | None, _ff_grouping] = None
FF_GROUPING_RMS_NORM_EPSILON: Annotated[float | None, _ff_grouping] = None

declare_soft_moe_runtime_defaults(globals())
_bert_embedding.declare_runtime_defaults(globals())


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

_component_defaults.declare_component_defaults(globals())

# Whole-word reconstruction with the hierarchical byte decoder.
HIERARCHICAL_MASKED_WORD_FLAG: Annotated[bool, _sequence_expansion] = False

_recurrent_compression.declare_sequence_options(globals())
