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
    WeightNormalizationConfig,  # noqa: F401
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.config import ConfigBase
from emperor.embedding.absolute import (
    AbsolutePositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,  # noqa: F401
)
from emperor.experts import SoftMoEConfig
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
from models import _attention_source_memory, _component_defaults, _recurrent_compression

_general = Section(("Global",))
INPUT_DIM: Annotated[int, _general] = 28782
HIDDEN_DIM: Annotated[int, _general] = 32
OUTPUT_DIM: Annotated[int, _general] = 28782
BATCH_SIZE: Annotated[int, _general] = 64
LEARNING_RATE: Annotated[float, _general] = 1e-3
NUM_EPOCHS: Annotated[int, _general] = 10
CONFIG_OVERRIDE_SKIP_KEYS: Annotated[set[str], _general] = {"HALTING_OUTPUT_DIM"}
SEQUENCE_LENGTH: Annotated[int, _general] = 35

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
DATA_NUM_WORKERS: Annotated[int, _trainer] = 4
RUN_TEST_AFTER_FIT: Annotated[bool, _trainer] = True
SEED: Annotated[int | None, _trainer] = None
CALLBACK_EARLY_STOPPING_PATIENCE: Annotated[int, _trainer] = 0
CALLBACK_EARLY_STOPPING_METRIC: Annotated[str, _trainer] = "validation/loss"
CALLBACK_EARLY_STOPPING_MODE: Annotated[MetricDirection | None, _trainer] = (
    None  # Optional min/max direction for the monitored metric.
)
CALLBACK_EARLY_STOPPING_MIN_DELTA: Annotated[float, _trainer] = 0.0
CALLBACK_EARLY_STOPPING_STRICT: Annotated[bool, _trainer] = True
CALLBACK_EARLY_STOPPING_CHECK_FINITE: Annotated[bool, _trainer] = True
CALLBACK_CHECKPOINT_FLAG: Annotated[bool, _trainer] = False

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
    LayerNormPositionOptions.BEFORE
)
NORMALIZATION: Annotated[NormalizationOptions, _layer_stack] = (
    NormalizationOptions.RMS_NORM
)
DECODER_OUTPUT_NORMALIZATION: Annotated[NormalizationOptions, _layer_stack] = (
    NormalizationOptions.LAYER_NORM
)
STACK_LAST_LAYER_BIAS_OPTION: Annotated[LastLayerBiasOptions, _layer_stack] = (
    LastLayerBiasOptions.DEFAULT
)
STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool, _layer_stack] = True
STACK_BIAS_FLAG: Annotated[bool, _layer_stack] = True
_embedding_pipeline = Section(("Embedding Pipeline",))
EMBEDDING_LAYER_NORM_FLAG: Annotated[bool, _embedding_pipeline] = False
EMBEDDING_NORMALIZATION: Annotated[NormalizationOptions, _embedding_pipeline] = (
    NormalizationOptions.LAYER_NORM
)
EMBEDDING_DROPOUT_PROBABILITY: Annotated[float, _embedding_pipeline] = (
    STACK_DROPOUT_PROBABILITY
)

_language_modeling_head = Section(("Language-Modeling Head",))
LM_HEAD_BIAS_FLAG: Annotated[bool, _language_modeling_head] = False
LM_HEAD_WEIGHT_TYING_FLAG: Annotated[bool, _language_modeling_head] = True

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
SUBMODULE_STACK_BIAS_FLAG: Annotated[bool, _submodule_stack] = STACK_BIAS_FLAG

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
# If `WEIGHT_OPTION_FLAG` is False, the weight-specific parameters below are ignored.
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
# If `BIAS_OPTION_FLAG` is False, the bias-specific parameters below are ignored.
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
# If `DIAGONAL_OPTION_FLAG` is False, the diagonal-specific parameters below are
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
# If `MASK_OPTION_FLAG` is False, the mask-specific parameters below are ignored.
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

_residual = Section(("Residual Options",))
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

_attention_projection_adaptive_parameter = Section(
    ("Attention Projection Adaptive Parameter Options",)
)
_attention_projection_adaptive_weight = _attention_projection_adaptive_parameter.child(
    "Attention Projection Adaptive Weight Options"
)
ATTN_WEIGHT_OPTION_FLAG: Annotated[bool, _attention_projection_adaptive_weight] = (
    WEIGHT_OPTION_FLAG
)
ATTN_WEIGHT_OPTION: Annotated[
    type[DynamicWeightConfig] | None, _attention_projection_adaptive_weight
] = WEIGHT_OPTION
ATTN_GENERATOR_DEPTH: Annotated[int, _attention_projection_adaptive_weight] = (
    GENERATOR_DEPTH
)
ATTN_WEIGHT_DECAY_SCHEDULE: Annotated[
    WeightDecayScheduleOptions, _attention_projection_adaptive_weight
] = WEIGHT_DECAY_SCHEDULE
ATTN_WEIGHT_DECAY_RATE: Annotated[float, _attention_projection_adaptive_weight] = (
    WEIGHT_DECAY_RATE
)
ATTN_WEIGHT_DECAY_WARMUP_BATCHES: Annotated[
    int, _attention_projection_adaptive_weight
] = WEIGHT_DECAY_WARMUP_BATCHES
ATTN_WEIGHT_NORMALIZATION_OPTION: Annotated[
    WeightNormalizationOptions, _attention_projection_adaptive_weight
] = WEIGHT_NORMALIZATION_OPTION
ATTN_WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG: Annotated[
    bool, _attention_projection_adaptive_weight
] = WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG
ATTN_WEIGHT_NORMALIZATION_POSITION_OPTION: Annotated[
    WeightNormalizationPositionOptions, _attention_projection_adaptive_weight
] = WEIGHT_NORMALIZATION_POSITION_OPTION
ATTN_WEIGHT_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions, _attention_projection_adaptive_weight
] = WEIGHT_BANK_EXPANSION_FACTOR
_attention_projection_weight_generator_stack = (
    _attention_projection_adaptive_weight.child(
        "Attention Projection Weight Generator Stack Options"
    )
)
ATTN_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attention_projection_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG
ATTN_WEIGHT_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _attention_projection_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_HIDDEN_DIM
ATTN_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attention_projection_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION
ATTN_WEIGHT_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attention_projection_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_NORMALIZATION
ATTN_WEIGHT_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _attention_projection_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_NUM_LAYERS
ATTN_WEIGHT_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_ACTIVATION
ATTN_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attention_projection_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
ATTN_WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attention_projection_weight_generator_stack
] = False
ATTN_WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attention_projection_weight_generator_stack
] = None
ATTN_WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attention_projection_weight_generator_stack
] = None
ATTN_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attention_projection_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY
ATTN_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attention_projection_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
ATTN_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attention_projection_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
ATTN_WEIGHT_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _attention_projection_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_BIAS_FLAG

_attention_projection_adaptive_bias = _attention_projection_adaptive_weight.child(
    "Attention Projection Adaptive Bias Options"
)
ATTN_BIAS_OPTION_FLAG: Annotated[bool, _attention_projection_adaptive_bias] = (
    BIAS_OPTION_FLAG
)
ATTN_BIAS_OPTION: Annotated[
    type[DynamicBiasConfig] | None, _attention_projection_adaptive_bias
] = BIAS_OPTION
ATTN_BIAS_DECAY_SCHEDULE: Annotated[
    WeightDecayScheduleOptions, _attention_projection_adaptive_bias
] = BIAS_DECAY_SCHEDULE
ATTN_BIAS_DECAY_RATE: Annotated[float, _attention_projection_adaptive_bias] = (
    BIAS_DECAY_RATE
)
ATTN_BIAS_DECAY_WARMUP_BATCHES: Annotated[int, _attention_projection_adaptive_bias] = (
    BIAS_DECAY_WARMUP_BATCHES
)
ATTN_BIAS_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions, _attention_projection_adaptive_bias
] = BIAS_BANK_EXPANSION_FACTOR
_attention_projection_bias_generator_stack = _attention_projection_adaptive_bias.child(
    "Attention Projection Bias Generator Stack Options"
)
ATTN_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attention_projection_bias_generator_stack
] = BIAS_GENERATOR_STACK_INDEPENDENT_FLAG
ATTN_BIAS_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _attention_projection_bias_generator_stack
] = BIAS_GENERATOR_STACK_HIDDEN_DIM
ATTN_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attention_projection_bias_generator_stack
] = BIAS_GENERATOR_STACK_LAYER_NORM_POSITION
ATTN_BIAS_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attention_projection_bias_generator_stack
] = BIAS_GENERATOR_STACK_NORMALIZATION
ATTN_BIAS_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _attention_projection_bias_generator_stack
] = BIAS_GENERATOR_STACK_NUM_LAYERS
ATTN_BIAS_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_bias_generator_stack
] = BIAS_GENERATOR_STACK_ACTIVATION
ATTN_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attention_projection_bias_generator_stack
] = BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
ATTN_BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attention_projection_bias_generator_stack
] = False
ATTN_BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attention_projection_bias_generator_stack
] = None
ATTN_BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attention_projection_bias_generator_stack
] = None
ATTN_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attention_projection_bias_generator_stack
] = BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY
ATTN_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attention_projection_bias_generator_stack
] = BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
ATTN_BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attention_projection_bias_generator_stack
] = BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
ATTN_BIAS_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _attention_projection_bias_generator_stack
] = BIAS_GENERATOR_STACK_BIAS_FLAG

_attention_projection_adaptive_diagonal = _attention_projection_adaptive_weight.child(
    "Attention Projection Adaptive Diagonal Options"
)
ATTN_DIAGONAL_OPTION_FLAG: Annotated[bool, _attention_projection_adaptive_diagonal] = (
    DIAGONAL_OPTION_FLAG
)
ATTN_DIAGONAL_OPTION: Annotated[
    type[DynamicDiagonalConfig] | None, _attention_projection_adaptive_diagonal
] = DIAGONAL_OPTION
_attention_projection_diagonal_generator_stack = (
    _attention_projection_adaptive_diagonal.child(
        "Attention Projection Diagonal Generator Stack Options"
    )
)
ATTN_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attention_projection_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG
ATTN_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _attention_projection_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_HIDDEN_DIM
ATTN_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attention_projection_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION
ATTN_DIAGONAL_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attention_projection_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_NORMALIZATION
ATTN_DIAGONAL_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _attention_projection_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_NUM_LAYERS
ATTN_DIAGONAL_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_ACTIVATION
ATTN_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attention_projection_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
ATTN_DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attention_projection_diagonal_generator_stack
] = False
ATTN_DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attention_projection_diagonal_generator_stack
] = None
ATTN_DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attention_projection_diagonal_generator_stack
] = None
ATTN_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attention_projection_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY
ATTN_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attention_projection_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
ATTN_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attention_projection_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
ATTN_DIAGONAL_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _attention_projection_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_BIAS_FLAG

_attention_projection_adaptive_mask = _attention_projection_adaptive_weight.child(
    "Attention Projection Adaptive Mask Options"
)
ATTN_MASK_OPTION_FLAG: Annotated[bool, _attention_projection_adaptive_mask] = (
    MASK_OPTION_FLAG
)
ATTN_ROW_MASK_OPTION: Annotated[
    type[AxisMaskConfig] | None, _attention_projection_adaptive_mask
] = ROW_MASK_OPTION
ATTN_MASK_THRESHOLD: Annotated[float, _attention_projection_adaptive_mask] = (
    MASK_THRESHOLD
)
ATTN_MASK_FLOOR: Annotated[float, _attention_projection_adaptive_mask] = MASK_FLOOR
ATTN_MASK_TRANSITION_WIDTH: Annotated[float, _attention_projection_adaptive_mask] = (
    MASK_TRANSITION_WIDTH
)
ATTN_MASK_SURROGATE_SCALE: Annotated[float, _attention_projection_adaptive_mask] = (
    MASK_SURROGATE_SCALE
)
ATTN_MASK_DIMENSION_OPTION: Annotated[
    MaskDimensionOptions, _attention_projection_adaptive_mask
] = MASK_DIMENSION_OPTION
_attention_projection_mask_generator_stack = _attention_projection_adaptive_mask.child(
    "Attention Projection Mask Generator Stack Options"
)
ATTN_MASK_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attention_projection_mask_generator_stack
] = MASK_GENERATOR_STACK_INDEPENDENT_FLAG
ATTN_MASK_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _attention_projection_mask_generator_stack
] = MASK_GENERATOR_STACK_HIDDEN_DIM
ATTN_MASK_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attention_projection_mask_generator_stack
] = MASK_GENERATOR_STACK_LAYER_NORM_POSITION
ATTN_MASK_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attention_projection_mask_generator_stack
] = MASK_GENERATOR_STACK_NORMALIZATION
ATTN_MASK_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _attention_projection_mask_generator_stack
] = MASK_GENERATOR_STACK_NUM_LAYERS
ATTN_MASK_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attention_projection_mask_generator_stack
] = MASK_GENERATOR_STACK_ACTIVATION
ATTN_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attention_projection_mask_generator_stack
] = MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
ATTN_MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attention_projection_mask_generator_stack
] = False
ATTN_MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attention_projection_mask_generator_stack
] = None
ATTN_MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attention_projection_mask_generator_stack
] = None
ATTN_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attention_projection_mask_generator_stack
] = MASK_GENERATOR_STACK_DROPOUT_PROBABILITY
ATTN_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attention_projection_mask_generator_stack
] = MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
ATTN_MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attention_projection_mask_generator_stack
] = MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
ATTN_MASK_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _attention_projection_mask_generator_stack
] = MASK_GENERATOR_STACK_BIAS_FLAG

_attention_projection_gate = _attention_projection_adaptive_weight.child(
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

_attention_projection_halting = _attention_projection_adaptive_weight.child(
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

_attention_source_memory_section = _attention_projection_adaptive_weight.child(
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
_attention_projection_memory = _attention_projection_adaptive_weight.child(
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

_attention_projection_recurrent_layer = _attention_projection_adaptive_weight.child(
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
SOFT_MOE_CONFIG: Annotated[SoftMoEConfig | None, _feed_forward_stack] = None
FF_NUM_LAYERS: Annotated[int, _feed_forward_stack] = 1
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

_feed_forward_adaptive_parameter = Section(("Feed-Forward Adaptive Parameter Options",))
_feed_forward_adaptive_weight = _feed_forward_adaptive_parameter.child(
    "Feed-Forward Adaptive Weight Options"
)
FF_WEIGHT_OPTION_FLAG: Annotated[bool, _feed_forward_adaptive_weight] = (
    WEIGHT_OPTION_FLAG
)
FF_WEIGHT_OPTION: Annotated[
    type[DynamicWeightConfig] | None, _feed_forward_adaptive_weight
] = WEIGHT_OPTION
FF_GENERATOR_DEPTH: Annotated[int, _feed_forward_adaptive_weight] = GENERATOR_DEPTH
FF_WEIGHT_DECAY_SCHEDULE: Annotated[
    WeightDecayScheduleOptions, _feed_forward_adaptive_weight
] = WEIGHT_DECAY_SCHEDULE
FF_WEIGHT_DECAY_RATE: Annotated[float, _feed_forward_adaptive_weight] = (
    WEIGHT_DECAY_RATE
)
FF_WEIGHT_DECAY_WARMUP_BATCHES: Annotated[int, _feed_forward_adaptive_weight] = (
    WEIGHT_DECAY_WARMUP_BATCHES
)
FF_WEIGHT_NORMALIZATION_OPTION: Annotated[
    WeightNormalizationOptions, _feed_forward_adaptive_weight
] = WEIGHT_NORMALIZATION_OPTION
FF_WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG: Annotated[
    bool, _feed_forward_adaptive_weight
] = WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG
FF_WEIGHT_NORMALIZATION_POSITION_OPTION: Annotated[
    WeightNormalizationPositionOptions, _feed_forward_adaptive_weight
] = WEIGHT_NORMALIZATION_POSITION_OPTION
FF_WEIGHT_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions, _feed_forward_adaptive_weight
] = WEIGHT_BANK_EXPANSION_FACTOR
_feed_forward_weight_generator_stack = _feed_forward_adaptive_weight.child(
    "Feed-Forward Weight Generator Stack Options"
)
FF_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _feed_forward_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG
FF_WEIGHT_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _feed_forward_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_HIDDEN_DIM
FF_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _feed_forward_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION
FF_WEIGHT_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _feed_forward_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_NORMALIZATION
FF_WEIGHT_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _feed_forward_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_NUM_LAYERS
FF_WEIGHT_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _feed_forward_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_ACTIVATION
FF_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _feed_forward_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
FF_WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _feed_forward_weight_generator_stack
] = False
FF_WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _feed_forward_weight_generator_stack
] = None
FF_WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _feed_forward_weight_generator_stack
] = None
FF_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _feed_forward_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY
FF_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _feed_forward_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
FF_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _feed_forward_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
FF_WEIGHT_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _feed_forward_weight_generator_stack
] = WEIGHT_GENERATOR_STACK_BIAS_FLAG

_feed_forward_adaptive_bias = _feed_forward_adaptive_parameter.child(
    "Feed-Forward Adaptive Bias Options"
)
FF_BIAS_OPTION_FLAG: Annotated[bool, _feed_forward_adaptive_bias] = BIAS_OPTION_FLAG
FF_BIAS_OPTION: Annotated[
    type[DynamicBiasConfig] | None, _feed_forward_adaptive_bias
] = BIAS_OPTION
FF_BIAS_DECAY_SCHEDULE: Annotated[
    WeightDecayScheduleOptions, _feed_forward_adaptive_bias
] = BIAS_DECAY_SCHEDULE
FF_BIAS_DECAY_RATE: Annotated[float, _feed_forward_adaptive_bias] = BIAS_DECAY_RATE
FF_BIAS_DECAY_WARMUP_BATCHES: Annotated[int, _feed_forward_adaptive_bias] = (
    BIAS_DECAY_WARMUP_BATCHES
)
FF_BIAS_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions, _feed_forward_adaptive_bias
] = BIAS_BANK_EXPANSION_FACTOR
_feed_forward_bias_generator_stack = _feed_forward_adaptive_bias.child(
    "Feed-Forward Bias Generator Stack Options"
)
FF_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _feed_forward_bias_generator_stack
] = BIAS_GENERATOR_STACK_INDEPENDENT_FLAG
FF_BIAS_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _feed_forward_bias_generator_stack
] = BIAS_GENERATOR_STACK_HIDDEN_DIM
FF_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _feed_forward_bias_generator_stack
] = BIAS_GENERATOR_STACK_LAYER_NORM_POSITION
FF_BIAS_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _feed_forward_bias_generator_stack
] = BIAS_GENERATOR_STACK_NORMALIZATION
FF_BIAS_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _feed_forward_bias_generator_stack
] = BIAS_GENERATOR_STACK_NUM_LAYERS
FF_BIAS_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _feed_forward_bias_generator_stack
] = BIAS_GENERATOR_STACK_ACTIVATION
FF_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _feed_forward_bias_generator_stack
] = BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
FF_BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _feed_forward_bias_generator_stack
] = False
FF_BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _feed_forward_bias_generator_stack
] = None
FF_BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _feed_forward_bias_generator_stack
] = None
FF_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _feed_forward_bias_generator_stack
] = BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY
FF_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _feed_forward_bias_generator_stack
] = BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
FF_BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _feed_forward_bias_generator_stack
] = BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
FF_BIAS_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _feed_forward_bias_generator_stack
] = BIAS_GENERATOR_STACK_BIAS_FLAG

_feed_forward_adaptive_diagonal = _feed_forward_adaptive_parameter.child(
    "Feed-Forward Adaptive Diagonal Options"
)
FF_DIAGONAL_OPTION_FLAG: Annotated[bool, _feed_forward_adaptive_diagonal] = (
    DIAGONAL_OPTION_FLAG
)
FF_DIAGONAL_OPTION: Annotated[
    type[DynamicDiagonalConfig] | None, _feed_forward_adaptive_diagonal
] = DIAGONAL_OPTION
_feed_forward_diagonal_generator_stack = _feed_forward_adaptive_diagonal.child(
    "Feed-Forward Diagonal Generator Stack Options"
)
FF_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _feed_forward_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG
FF_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _feed_forward_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_HIDDEN_DIM
FF_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _feed_forward_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION
FF_DIAGONAL_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _feed_forward_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_NORMALIZATION
FF_DIAGONAL_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _feed_forward_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_NUM_LAYERS
FF_DIAGONAL_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _feed_forward_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_ACTIVATION
FF_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _feed_forward_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
FF_DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _feed_forward_diagonal_generator_stack
] = False
FF_DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _feed_forward_diagonal_generator_stack
] = None
FF_DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _feed_forward_diagonal_generator_stack
] = None
FF_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _feed_forward_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY
FF_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _feed_forward_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
FF_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _feed_forward_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
FF_DIAGONAL_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _feed_forward_diagonal_generator_stack
] = DIAGONAL_GENERATOR_STACK_BIAS_FLAG

_feed_forward_adaptive_mask = _feed_forward_adaptive_parameter.child(
    "Feed-Forward Adaptive Mask Options"
)
FF_MASK_OPTION_FLAG: Annotated[bool, _feed_forward_adaptive_mask] = MASK_OPTION_FLAG
FF_ROW_MASK_OPTION: Annotated[
    type[AxisMaskConfig] | None, _feed_forward_adaptive_mask
] = ROW_MASK_OPTION
FF_MASK_THRESHOLD: Annotated[float, _feed_forward_adaptive_mask] = MASK_THRESHOLD
FF_MASK_FLOOR: Annotated[float, _feed_forward_adaptive_mask] = MASK_FLOOR
FF_MASK_TRANSITION_WIDTH: Annotated[float, _feed_forward_adaptive_mask] = (
    MASK_TRANSITION_WIDTH
)
FF_MASK_SURROGATE_SCALE: Annotated[float, _feed_forward_adaptive_mask] = (
    MASK_SURROGATE_SCALE
)
FF_MASK_DIMENSION_OPTION: Annotated[
    MaskDimensionOptions, _feed_forward_adaptive_mask
] = MASK_DIMENSION_OPTION
_feed_forward_mask_generator_stack = _feed_forward_adaptive_mask.child(
    "Feed-Forward Mask Generator Stack Options"
)
FF_MASK_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _feed_forward_mask_generator_stack
] = MASK_GENERATOR_STACK_INDEPENDENT_FLAG
FF_MASK_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _feed_forward_mask_generator_stack
] = MASK_GENERATOR_STACK_HIDDEN_DIM
FF_MASK_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _feed_forward_mask_generator_stack
] = MASK_GENERATOR_STACK_LAYER_NORM_POSITION
FF_MASK_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _feed_forward_mask_generator_stack
] = MASK_GENERATOR_STACK_NORMALIZATION
FF_MASK_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _feed_forward_mask_generator_stack
] = MASK_GENERATOR_STACK_NUM_LAYERS
FF_MASK_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _feed_forward_mask_generator_stack
] = MASK_GENERATOR_STACK_ACTIVATION
FF_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _feed_forward_mask_generator_stack
] = MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
FF_MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _feed_forward_mask_generator_stack
] = False
FF_MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _feed_forward_mask_generator_stack
] = None
FF_MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _feed_forward_mask_generator_stack
] = None
FF_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _feed_forward_mask_generator_stack
] = MASK_GENERATOR_STACK_DROPOUT_PROBABILITY
FF_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _feed_forward_mask_generator_stack
] = MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
FF_MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _feed_forward_mask_generator_stack
] = MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
FF_MASK_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _feed_forward_mask_generator_stack
] = MASK_GENERATOR_STACK_BIAS_FLAG

_feed_forward_gate = _feed_forward_adaptive_parameter.child("Feed-Forward Gate Options")
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

_feed_forward_halting = _feed_forward_adaptive_parameter.child(
    "Feed-Forward Halting Options"
)
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

_feed_forward_memory = _feed_forward_adaptive_parameter.child(
    "Feed-Forward Memory Options"
)
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

_feed_forward_recurrent_layer = _feed_forward_adaptive_parameter.child(
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
BIAS_MIXTURE_TOP_K: Annotated[int | None, _main_adaptive_generation] = None
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
_ff_adaptive_generation = Section(("Ff Adaptive Generation",))
# ==============================================================================
FF_WEIGHT_INPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_NUM_EXPERTS: Annotated[int | None, _ff_adaptive_generation] = None
FF_BIAS_MIXTURE_NUM_EXPERTS: Annotated[int | None, _ff_adaptive_generation] = None
FF_BIAS_MIXTURE_TOP_K: Annotated[int | None, _ff_adaptive_generation] = None
FF_BIAS_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool | None, _ff_adaptive_generation
] = None
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _ff_adaptive_generation
] = False
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _ff_adaptive_generation
] = None
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _ff_adaptive_generation
] = False
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _ff_adaptive_generation
] = None
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _ff_adaptive_generation
] = None
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _ff_adaptive_generation
] = None
FF_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _ff_adaptive_generation
] = False
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _ff_adaptive_generation
] = False
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _ff_adaptive_generation
] = None
FF_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _ff_adaptive_generation
] = None
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _ff_adaptive_generation
] = False
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _ff_adaptive_generation
] = None
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _ff_adaptive_generation
] = False
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _ff_adaptive_generation
] = None
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _ff_adaptive_generation
] = None
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _ff_adaptive_generation
] = None
FF_WEIGHT_COEFFICIENT_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _ff_adaptive_generation
] = False
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _ff_adaptive_generation
] = False
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _ff_adaptive_generation
] = None
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _ff_adaptive_generation
] = None
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _ff_adaptive_generation
] = None
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _ff_adaptive_generation
] = None
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _ff_adaptive_generation
] = False
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _ff_adaptive_generation
] = None
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _ff_adaptive_generation
] = False
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _ff_adaptive_generation
] = None
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _ff_adaptive_generation
] = None
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _ff_adaptive_generation
] = None
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _ff_adaptive_generation
] = None
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _ff_adaptive_generation
] = None
FF_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _ff_adaptive_generation
] = None
FF_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _ff_adaptive_generation
] = None

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


# ==============================================================================
_attn_adaptive_generation = Section(("Attn Adaptive Generation",))
# ==============================================================================
ATTN_WEIGHT_INPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_NUM_EXPERTS: Annotated[int | None, _attn_adaptive_generation] = None
ATTN_BIAS_MIXTURE_NUM_EXPERTS: Annotated[int | None, _attn_adaptive_generation] = None
ATTN_BIAS_MIXTURE_TOP_K: Annotated[int | None, _attn_adaptive_generation] = None
ATTN_BIAS_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attn_adaptive_generation
] = False
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attn_adaptive_generation
] = False
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attn_adaptive_generation
] = False
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attn_adaptive_generation
] = False
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attn_adaptive_generation
] = False
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attn_adaptive_generation
] = False
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_COEFFICIENT_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attn_adaptive_generation
] = False
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _attn_adaptive_generation
] = False
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attn_adaptive_generation
] = None
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _attn_adaptive_generation
] = None
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attn_adaptive_generation
] = None
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attn_adaptive_generation
] = None
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attn_adaptive_generation
] = False
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attn_adaptive_generation
] = None
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _attn_adaptive_generation
] = False
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _attn_adaptive_generation
] = None
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attn_adaptive_generation
] = None
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attn_adaptive_generation
] = None
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attn_adaptive_generation
] = None
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _attn_adaptive_generation
] = None
ATTN_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _attn_adaptive_generation
] = None
ATTN_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _attn_adaptive_generation
] = None

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


# Full hierarchical byte language modeling (independent of backbone width).
HIERARCHICAL_LANGUAGE_MODEL_FLAG: Annotated[bool, _attn_grouping] = False
HIERARCHICAL_BYTE_EMBEDDING_DIM: Annotated[int, _attn_grouping] = 32
HIERARCHICAL_BYTE_ENCODER_NUM_LAYERS: Annotated[int, _attn_grouping] = 1
HIERARCHICAL_BYTE_DECODER_NUM_LAYERS: Annotated[int, _attn_grouping] = 1
HIERARCHICAL_BYTE_NUM_HEADS: Annotated[int, _attn_grouping] = 4
HIERARCHICAL_BYTE_FEED_FORWARD_DIM: Annotated[int, _attn_grouping] = 64
HIERARCHICAL_MAX_TOKEN_BYTES: Annotated[int, _attn_grouping] = 64


_workbench_config_schema_boundary = Section(("Workbench Config Schema Boundary",))
# Construction defaults remain available to flat CLI adapters and presets. The
# grouped builder interface exposes residual and adaptive parameter settings.
_PUBLIC_CONFIG_KEYS: Annotated[set[str], _workbench_config_schema_boundary] = {
    "RESIDUAL_COEFFICIENT_ACTIVATION",
    "SOFT_MOE_CONFIG",
    "GENERATOR_DEPTH",
    "ROW_MASK_OPTION",
    "GROUP_COUNT",
    "CHUNK_SIZE",
    "INPUT_DIM",
    "OUTPUT_DIM",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "NUM_EPOCHS",
    "SEQUENCE_LENGTH",
    "EMBEDDING_LAYER_NORM_FLAG",
    "EMBEDDING_NORMALIZATION",
    "EMBEDDING_DROPOUT_PROBABILITY",
    "LM_HEAD_BIAS_FLAG",
    "LM_HEAD_WEIGHT_TYING_FLAG",
}

RUNTIME_VALUE_CONSTRAINTS: Annotated[
    RuntimeValueConstraints, _workbench_config_schema_boundary
] = positive_runtime_fields("HIDDEN_DIM")
_PUBLIC_CONFIG_PREFIXES: Annotated[
    tuple[str, ...], _workbench_config_schema_boundary
] = (
    "ATTN_SOURCE_MEMORY_",
    "HIERARCHICAL_",
    "ADAPTIVE_GENERATOR_STACK_",
    "WEIGHT_",
    "BIAS_",
    "DIAGONAL_",
    "MASK_",
    "GROUPING_",
    "ATTN_ADAPTIVE_GENERATOR_STACK_",
    "ATTN_WEIGHT_",
    "ATTN_BIAS_",
    "ATTN_DIAGONAL_",
    "ATTN_MASK_",
    "ATTN_GROUPING_",
    "FF_ADAPTIVE_GENERATOR_STACK_",
    "FF_WEIGHT_",
    "FF_BIAS_",
    "FF_DIAGONAL_",
    "FF_MASK_",
    "FF_GROUPING_",
    "ROUTER_WEIGHT_",
    "ROUTER_BIAS_",
    "ROUTER_GROUPING_",
    "ROUTER_DIAGONAL_",
    "ROUTER_MASK_",
    "TRAINER_",
    "CALLBACK_",
    "DATA_",
    "RUN_",
    "MONITOR_",
    "RESIDUAL_STACK_",
)
_PUBLIC_CONFIG_SUFFIXES: Annotated[
    tuple[str, ...], _workbench_config_schema_boundary
] = (
    "MEMORY_NUM_MEMORY_SLOTS",
    "_GROUP_COUNT",
    "_CHUNK_SIZE",
    "_GENERATOR_DEPTH",
    "_ROW_MASK_OPTION",
    "_RESIDUAL_CONNECTION_OPTION",
    "_RESIDUAL_BLOCK_SIZE",
    "_RESIDUAL_RMS_NORM_EPSILON",
    "_RESIDUAL_MODEL_FLAG",
)
CONFIG_SCHEMA_SKIP_KEYS: Annotated[set[str], _workbench_config_schema_boundary] = {
    key
    for key in globals()
    if key.isupper()
    and not key.startswith("_")
    and key != "CONFIG_SCHEMA_SKIP_KEYS"
    and key not in _PUBLIC_CONFIG_KEYS
    and not key.startswith(_PUBLIC_CONFIG_PREFIXES)
    and not key.endswith(_PUBLIC_CONFIG_SUFFIXES)
}

declare_soft_moe_runtime_defaults(globals())


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

_recurrent_compression.declare_sequence_options(globals(), include_residual=True)
