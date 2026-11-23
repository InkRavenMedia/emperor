from typing import Annotated

from emperor.augmentations.adaptive_parameters import (  # noqa: F401
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
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
    StandardDynamicDiagonalConfig,
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
from emperor.embedding.absolute import (  # noqa: F401
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
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
    AdditiveResidualConfig,
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
from model_runtime.packages import runtime_values, soft_moe
from model_runtime.packages.configuration_metadata import Section
from model_runtime.packages.metric_direction import MetricDirection
from models import _component_defaults, _recurrent_compression
from models.mlp_mixer._settings import ImageReadout

_general = Section(("Global",))
BATCH_SIZE: Annotated[int, _general] = 64
INPUT_DIM: Annotated[int, _general] = 224 * 224 * 3
HIDDEN_DIM: Annotated[int, _general] = 32
RUNTIME_VALUE_CONSTRAINTS: Annotated[
    runtime_values.RuntimeValueConstraints, _general
] = runtime_values.positive_runtime_fields("HIDDEN_DIM")
OUTPUT_DIM: Annotated[int, _general] = 10
LEARNING_RATE: Annotated[float, _general] = 1e-3
NUM_EPOCHS: Annotated[int, _general] = 10

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
CALLBACK_EARLY_STOPPING_METRIC: Annotated[str, _callback] = "validation/accuracy"
CALLBACK_EARLY_STOPPING_MODE: Annotated[MetricDirection | None, _callback] = (
    None  # Optional min/max direction for the monitored metric.
)
CALLBACK_EARLY_STOPPING_MIN_DELTA: Annotated[float, _callback] = 0.0
CALLBACK_EARLY_STOPPING_STRICT: Annotated[bool, _callback] = True
CALLBACK_EARLY_STOPPING_CHECK_FINITE: Annotated[bool, _callback] = True
CALLBACK_CHECKPOINT_FLAG: Annotated[bool, _callback] = False

_layer_stack = Section(("Layer Stack Options",))
# - hidden_dim comes from the global HIDDEN_DIM field above.
# - Token and channel mixer stacks inherit applicable activation/bias defaults.
LAYER_NORM_POSITION: Annotated[LayerNormPositionOptions, _layer_stack] = (
    LayerNormPositionOptions.BEFORE
)
NORMALIZATION: Annotated[NormalizationOptions, _layer_stack] = (
    NormalizationOptions.RMS_NORM
)
ENCODER_OUTPUT_NORMALIZATION: Annotated[NormalizationOptions, _layer_stack] = (
    NormalizationOptions.LAYER_NORM
)
STACK_NUM_LAYERS: Annotated[int, _layer_stack] = 8
STACK_ACTIVATION: Annotated[ActivationOptions, _layer_stack] = ActivationOptions.GELU
STACK_DROPOUT_PROBABILITY: Annotated[float, _layer_stack] = 0.0
STACK_LAST_LAYER_BIAS_OPTION: Annotated[LastLayerBiasOptions, _layer_stack] = (
    LastLayerBiasOptions.DEFAULT
)
STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool, _layer_stack] = True
STACK_BIAS_FLAG: Annotated[bool, _layer_stack] = True

_mixer_block = _layer_stack.child("Mixer Block Options")
MIXER_RESIDUAL_CONNECTION_OPTION: Annotated[type[ResidualConfig], _mixer_block] = (
    AdditiveResidualConfig
)
MIXER_RESIDUAL_MODEL_FLAG: Annotated[bool, _mixer_block] = False
MIXER_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _mixer_block] = None
MIXER_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _mixer_block] = None

_submodule_stack = Section(("Layer Stack Submodule Options",))
SUBMODULE_STACK_HIDDEN_DIM: Annotated[int, _submodule_stack] = HIDDEN_DIM
SUBMODULE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _submodule_stack
] = LayerNormPositionOptions.DISABLED
SUBMODULE_STACK_NORMALIZATION: Annotated[NormalizationOptions, _submodule_stack] = (
    NormalizationOptions.RMS_NORM
)
SUBMODULE_STACK_NUM_LAYERS: Annotated[int, _submodule_stack] = 1
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
# - If `STACK_GATE_FLAG` is False, the gate-specific parameters below are ignored.
STACK_GATE_FLAG: Annotated[bool, _gate] = False
GATE_OPTION: Annotated[LayerGateOptions | None, _gate] = LayerGateOptions.MULTIPLIER
GATE_ACTIVATION: Annotated[ActivationOptions | None, _gate] = ActivationOptions.SIGMOID
_gate_stack = _gate.child("Gate Stack Options")
# - If False, gate model stack options inherit the layer stack submodule options.
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
# - If `STACK_HALTING_FLAG` is False, the halting-specific parameters below are ignored.
STACK_HALTING_FLAG: Annotated[bool, _halting] = False
HALTING_OPTION: Annotated[type[HaltingConfig], _halting] = StickBreakingConfig
HALTING_THRESHOLD: Annotated[float, _halting] = 0.999
HALTING_DROPOUT: Annotated[float, _halting] = 0.0
HALTING_HIDDEN_STATE_MODE: Annotated[HaltingHiddenStateModeOptions, _halting] = (
    HaltingHiddenStateModeOptions.RAW
)
_halting_stack = _halting.child("Halting Stack Options")
# - If False, halting model stack options inherit the layer stack submodule options.
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
# - If `MEMORY_FLAG` is False, the memory-specific parameters below are ignored.
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
# - If False, memory model stack options inherit the layer stack submodule options.
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
# - If `RECURRENT_FLAG` is False, the recurrent-specific parameters below are ignored.
RECURRENT_FLAG: Annotated[bool, _recurrent_layer] = False
RECURRENT_MAX_STEPS: Annotated[int, _recurrent_layer] = 2
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
# - If False, recurrent gate stack options inherit gate/submodule stack options.
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
RECURRENT_HALTING_THRESHOLD: Annotated[float, _recurrent_halting] = 0.999
RECURRENT_HALTING_DROPOUT: Annotated[float, _recurrent_halting] = 0.0
RECURRENT_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _recurrent_halting
] = HaltingHiddenStateModeOptions.RAW
_recurrent_halting_stack = _recurrent_halting.child("Recurrent Halting Stack Options")
# - If False, recurrent halting stack options inherit halting/submodule stack options.
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
_recurrent_memory = _recurrent_layer.child("Recurrent Memory Options")
RECURRENT_MEMORY_FLAG: Annotated[bool, _recurrent_memory] = False

_mlp_mixer = Section(("Mlp Mixer Options",))
_output_layer = _mlp_mixer.child("Output Layer Options")
OUTPUT_BIAS_FLAG: Annotated[bool, _output_layer] = True
# Use mean to average valid patch states, or cls to prepend and read a learned class token.
IMAGE_READOUT: Annotated[ImageReadout, _output_layer] = ImageReadout.MEAN

_image_patch = _mlp_mixer.child("Image Patch Options")
IMAGE_PATCH_SIZE: Annotated[int, _image_patch] = 16
INPUT_CHANNELS: Annotated[int, _image_patch] = 3
IMAGE_HEIGHT: Annotated[int, _image_patch] = 224
PATCH_DROPOUT_PROBABILITY: Annotated[float, _image_patch] = 0.0
PATCH_BIAS_FLAG: Annotated[bool, _image_patch] = True

_token_mixer_stack = _mlp_mixer.child("Token Mixer Stack Options")
# - STACK_ACTIVATION and STACK_BIAS_FLAG supply inherited branch defaults.
# Temporary memory tokens mixed alongside the original token positions.
TOKEN_MIXER_SOFT_MOE_CONFIG: Annotated[SoftMoEConfig | None, _token_mixer_stack] = None
TOKEN_MIXER_SOURCE_MEMORY_FLAG: Annotated[bool, _token_mixer_stack] = False
TOKEN_MIXER_SOURCE_MEMORY_OPTION: Annotated[
    type[DynamicMemoryConfig], _token_mixer_stack
] = GatedResidualDynamicMemoryConfig
TOKEN_MIXER_SOURCE_MEMORY_NUM_MEMORY_SLOTS: Annotated[int, _token_mixer_stack] = 4
TOKEN_MIXER_SOURCE_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: Annotated[
    float | None, _token_mixer_stack
] = None
TOKEN_MIXER_SOURCE_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: Annotated[
    int | None, _token_mixer_stack
] = None

TOKEN_MIXER_STACK_HIDDEN_DIM: Annotated[int, _token_mixer_stack] = 64
TOKEN_MIXER_NUM_LAYERS: Annotated[int, _token_mixer_stack] = 2
TOKEN_MIXER_STACK_ACTIVATION: Annotated[ActivationOptions, _token_mixer_stack] = (
    STACK_ACTIVATION
)
TOKEN_MIXER_STACK_DROPOUT_PROBABILITY: Annotated[float, _token_mixer_stack] = 0.0
TOKEN_MIXER_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _token_mixer_stack
] = LayerNormPositionOptions.DISABLED
TOKEN_MIXER_STACK_NORMALIZATION: Annotated[NormalizationOptions, _token_mixer_stack] = (
    NormalizationOptions.RMS_NORM
)
TOKEN_MIXER_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _token_mixer_stack
] = None
TOKEN_MIXER_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _token_mixer_stack] = False
TOKEN_MIXER_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _token_mixer_stack] = None
TOKEN_MIXER_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _token_mixer_stack
] = None
TOKEN_MIXER_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions, _token_mixer_stack
] = LastLayerBiasOptions.DEFAULT
TOKEN_MIXER_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool, _token_mixer_stack
] = False

_channel_mixer_stack = _mlp_mixer.child("Channel Mixer Stack Options")
# - STACK_ACTIVATION and STACK_BIAS_FLAG supply inherited branch defaults.
SOFT_MOE_CONFIG: Annotated[SoftMoEConfig | None, _channel_mixer_stack] = None
CHANNEL_MIXER_STACK_HIDDEN_DIM: Annotated[int, _channel_mixer_stack] = 128
CHANNEL_MIXER_NUM_LAYERS: Annotated[int, _channel_mixer_stack] = 2
CHANNEL_MIXER_STACK_ACTIVATION: Annotated[ActivationOptions, _channel_mixer_stack] = (
    STACK_ACTIVATION
)
CHANNEL_MIXER_STACK_DROPOUT_PROBABILITY: Annotated[float, _channel_mixer_stack] = 0.0
CHANNEL_MIXER_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _channel_mixer_stack
] = LayerNormPositionOptions.DISABLED
CHANNEL_MIXER_STACK_NORMALIZATION: Annotated[
    NormalizationOptions, _channel_mixer_stack
] = NormalizationOptions.RMS_NORM
CHANNEL_MIXER_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _channel_mixer_stack
] = None
CHANNEL_MIXER_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _channel_mixer_stack] = False
CHANNEL_MIXER_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _channel_mixer_stack] = (
    None
)
CHANNEL_MIXER_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _channel_mixer_stack
] = None
CHANNEL_MIXER_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions, _channel_mixer_stack
] = LastLayerBiasOptions.DEFAULT
CHANNEL_MIXER_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool, _channel_mixer_stack
] = False

_token_mixer_gate = Section(("Token Mixer Gate Options",))
# - If `TOKEN_MIXER_STACK_GATE_FLAG` is False, the gate-specific parameters below are ignored.
TOKEN_MIXER_STACK_GATE_FLAG: Annotated[bool, _token_mixer_gate] = False
TOKEN_MIXER_GATE_OPTION: Annotated[LayerGateOptions | None, _token_mixer_gate] = (
    LayerGateOptions.MULTIPLIER
)
TOKEN_MIXER_GATE_ACTIVATION: Annotated[ActivationOptions | None, _token_mixer_gate] = (
    ActivationOptions.SIGMOID
)
_token_mixer_gate_stack = _token_mixer_gate.child("Token Mixer Gate Stack Options")
# - If False, gate stack options inherit the token mixer stack options.
TOKEN_MIXER_GATE_STACK_INDEPENDENT_FLAG: Annotated[bool, _token_mixer_gate_stack] = (
    False
)
TOKEN_MIXER_GATE_STACK_HIDDEN_DIM: Annotated[int | None, _token_mixer_gate_stack] = None
TOKEN_MIXER_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _token_mixer_gate_stack
] = None
TOKEN_MIXER_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _token_mixer_gate_stack
] = None
TOKEN_MIXER_GATE_STACK_NUM_LAYERS: Annotated[int | None, _token_mixer_gate_stack] = None
TOKEN_MIXER_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _token_mixer_gate_stack
] = ActivationOptions.TANH
TOKEN_MIXER_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _token_mixer_gate_stack
] = None
TOKEN_MIXER_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _token_mixer_gate_stack] = (
    False
)
TOKEN_MIXER_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _token_mixer_gate_stack
] = None
TOKEN_MIXER_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _token_mixer_gate_stack
] = None
TOKEN_MIXER_GATE_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _token_mixer_gate_stack
] = None
TOKEN_MIXER_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _token_mixer_gate_stack
] = None
TOKEN_MIXER_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _token_mixer_gate_stack
] = True
TOKEN_MIXER_GATE_STACK_BIAS_FLAG: Annotated[bool | None, _token_mixer_gate_stack] = True

_token_mixer_halting = Section(("Token Mixer Halting Options",))
# - If `TOKEN_MIXER_STACK_HALTING_FLAG` is False, the halting-specific parameters below are ignored.
TOKEN_MIXER_STACK_HALTING_FLAG: Annotated[bool, _token_mixer_halting] = False
TOKEN_MIXER_HALTING_OPTION: Annotated[type[HaltingConfig], _token_mixer_halting] = (
    StickBreakingConfig
)
TOKEN_MIXER_HALTING_THRESHOLD: Annotated[float, _token_mixer_halting] = 0.999
TOKEN_MIXER_HALTING_DROPOUT: Annotated[float, _token_mixer_halting] = 0.0
TOKEN_MIXER_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _token_mixer_halting
] = HaltingHiddenStateModeOptions.RAW
_token_mixer_halting_stack = _token_mixer_halting.child(
    "Token Mixer Halting Stack Options"
)
# - If False, halting stack options inherit the token mixer stack options.
TOKEN_MIXER_HALTING_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _token_mixer_halting_stack
] = False
TOKEN_MIXER_HALTING_STACK_HIDDEN_DIM: Annotated[
    int | None, _token_mixer_halting_stack
] = None
TOKEN_MIXER_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _token_mixer_halting_stack
] = LayerNormPositionOptions.DISABLED
TOKEN_MIXER_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _token_mixer_halting_stack
] = None
TOKEN_MIXER_HALTING_STACK_NUM_LAYERS: Annotated[
    int | None, _token_mixer_halting_stack
] = None
TOKEN_MIXER_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _token_mixer_halting_stack
] = None
TOKEN_MIXER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _token_mixer_halting_stack
] = None
TOKEN_MIXER_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _token_mixer_halting_stack
] = False
TOKEN_MIXER_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _token_mixer_halting_stack
] = None
TOKEN_MIXER_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _token_mixer_halting_stack
] = None
TOKEN_MIXER_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _token_mixer_halting_stack
] = None
TOKEN_MIXER_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _token_mixer_halting_stack
] = LastLayerBiasOptions.DISABLED
TOKEN_MIXER_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _token_mixer_halting_stack
] = None
TOKEN_MIXER_HALTING_STACK_BIAS_FLAG: Annotated[
    bool | None, _token_mixer_halting_stack
] = None

_token_mixer_memory = Section(("Token Mixer Memory Options",))
# - If `TOKEN_MIXER_MEMORY_FLAG` is False, the memory-specific parameters below are ignored.
TOKEN_MIXER_MEMORY_FLAG: Annotated[bool, _token_mixer_memory] = False
TOKEN_MIXER_MEMORY_OPTION: Annotated[type[DynamicMemoryConfig], _token_mixer_memory] = (
    GatedResidualDynamicMemoryConfig
)
TOKEN_MIXER_MEMORY_POSITION_OPTION: Annotated[
    MemoryPositionOptions, _token_mixer_memory
] = MemoryPositionOptions.AFTER_AFFINE
TOKEN_MIXER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: Annotated[
    float | None, _token_mixer_memory
] = None
TOKEN_MIXER_MEMORY_NUM_MEMORY_SLOTS: Annotated[int, _token_mixer_memory] = 4
TOKEN_MIXER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: Annotated[
    int | None, _token_mixer_memory
] = None
_token_mixer_memory_stack = _token_mixer_memory.child(
    "Token Mixer Memory Stack Options"
)
# - If False, memory stack options inherit the token mixer stack options.
TOKEN_MIXER_MEMORY_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _token_mixer_memory_stack
] = False
TOKEN_MIXER_MEMORY_STACK_HIDDEN_DIM: Annotated[
    int | None, _token_mixer_memory_stack
] = None
TOKEN_MIXER_MEMORY_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _token_mixer_memory_stack
] = None
TOKEN_MIXER_MEMORY_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _token_mixer_memory_stack
] = None
TOKEN_MIXER_MEMORY_STACK_NUM_LAYERS: Annotated[
    int | None, _token_mixer_memory_stack
] = None
TOKEN_MIXER_MEMORY_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _token_mixer_memory_stack
] = None
TOKEN_MIXER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _token_mixer_memory_stack
] = None
TOKEN_MIXER_MEMORY_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _token_mixer_memory_stack
] = False
TOKEN_MIXER_MEMORY_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _token_mixer_memory_stack
] = None
TOKEN_MIXER_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _token_mixer_memory_stack
] = None
TOKEN_MIXER_MEMORY_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _token_mixer_memory_stack
] = None
TOKEN_MIXER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _token_mixer_memory_stack
] = None
TOKEN_MIXER_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _token_mixer_memory_stack
] = None
TOKEN_MIXER_MEMORY_STACK_BIAS_FLAG: Annotated[
    bool | None, _token_mixer_memory_stack
] = None

_token_mixer_recurrent_layer = Section(("Token Mixer Recurrent Layer Options",))
# - If `TOKEN_MIXER_RECURRENT_FLAG` is False, the recurrent-specific parameters below are ignored.
TOKEN_MIXER_RECURRENT_FLAG: Annotated[bool, _token_mixer_recurrent_layer] = False
TOKEN_MIXER_RECURRENT_MAX_STEPS: Annotated[int, _token_mixer_recurrent_layer] = 2
TOKEN_MIXER_RECURRENT_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _token_mixer_recurrent_layer
] = LayerNormPositionOptions.DISABLED
TOKEN_MIXER_RECURRENT_NORMALIZATION: Annotated[
    NormalizationOptions, _token_mixer_recurrent_layer
] = NormalizationOptions.LAYER_NORM
TOKEN_MIXER_RECURRENT_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _token_mixer_recurrent_layer
] = None
TOKEN_MIXER_RECURRENT_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _token_mixer_recurrent_layer
] = False
TOKEN_MIXER_RECURRENT_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _token_mixer_recurrent_layer
] = None
TOKEN_MIXER_RECURRENT_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _token_mixer_recurrent_layer
] = None
_token_mixer_recurrent_gate = _token_mixer_recurrent_layer.child(
    "Token Mixer Recurrent Gate Options"
)
TOKEN_MIXER_RECURRENT_STACK_GATE_FLAG: Annotated[bool, _token_mixer_recurrent_gate] = (
    False
)
TOKEN_MIXER_RECURRENT_GATE_OPTION: Annotated[
    LayerGateOptions | None, _token_mixer_recurrent_gate
] = LayerGateOptions.MULTIPLIER
TOKEN_MIXER_RECURRENT_GATE_ACTIVATION: Annotated[
    ActivationOptions | None, _token_mixer_recurrent_gate
] = ActivationOptions.SIGMOID
_token_mixer_recurrent_gate_stack = _token_mixer_recurrent_gate.child(
    "Token Mixer Recurrent Gate Stack Options"
)
# - If False, recurrent gate stack options inherit the token mixer stack options.
TOKEN_MIXER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _token_mixer_recurrent_gate_stack
] = False
TOKEN_MIXER_RECURRENT_GATE_STACK_HIDDEN_DIM: Annotated[
    int | None, _token_mixer_recurrent_gate_stack
] = None
TOKEN_MIXER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _token_mixer_recurrent_gate_stack
] = None
TOKEN_MIXER_RECURRENT_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _token_mixer_recurrent_gate_stack
] = None
TOKEN_MIXER_RECURRENT_GATE_STACK_NUM_LAYERS: Annotated[
    int | None, _token_mixer_recurrent_gate_stack
] = None
TOKEN_MIXER_RECURRENT_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _token_mixer_recurrent_gate_stack
] = None
TOKEN_MIXER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _token_mixer_recurrent_gate_stack
] = None
TOKEN_MIXER_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _token_mixer_recurrent_gate_stack
] = False
TOKEN_MIXER_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _token_mixer_recurrent_gate_stack
] = None
TOKEN_MIXER_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _token_mixer_recurrent_gate_stack
] = None
TOKEN_MIXER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _token_mixer_recurrent_gate_stack
] = None
TOKEN_MIXER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _token_mixer_recurrent_gate_stack
] = None
TOKEN_MIXER_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _token_mixer_recurrent_gate_stack
] = None
TOKEN_MIXER_RECURRENT_GATE_STACK_BIAS_FLAG: Annotated[
    bool | None, _token_mixer_recurrent_gate_stack
] = None
_token_mixer_recurrent_halting = _token_mixer_recurrent_layer.child(
    "Token Mixer Recurrent Halting Options"
)
TOKEN_MIXER_RECURRENT_STACK_HALTING_FLAG: Annotated[
    bool, _token_mixer_recurrent_halting
] = False
TOKEN_MIXER_RECURRENT_HALTING_OPTION: Annotated[
    type[HaltingConfig], _token_mixer_recurrent_halting
] = StickBreakingConfig
TOKEN_MIXER_RECURRENT_HALTING_THRESHOLD: Annotated[
    float, _token_mixer_recurrent_halting
] = 0.999
TOKEN_MIXER_RECURRENT_HALTING_DROPOUT: Annotated[
    float, _token_mixer_recurrent_halting
] = 0.0
TOKEN_MIXER_RECURRENT_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _token_mixer_recurrent_halting
] = HaltingHiddenStateModeOptions.RAW
_token_mixer_recurrent_halting_stack = _token_mixer_recurrent_halting.child(
    "Token Mixer Recurrent Halting Stack Options"
)
# - If False, recurrent halting stack options inherit the token mixer stack options.
TOKEN_MIXER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _token_mixer_recurrent_halting_stack
] = False
TOKEN_MIXER_RECURRENT_HALTING_STACK_HIDDEN_DIM: Annotated[
    int | None, _token_mixer_recurrent_halting_stack
] = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _token_mixer_recurrent_halting_stack
] = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _token_mixer_recurrent_halting_stack
] = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_NUM_LAYERS: Annotated[
    int | None, _token_mixer_recurrent_halting_stack
] = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _token_mixer_recurrent_halting_stack
] = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _token_mixer_recurrent_halting_stack
] = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _token_mixer_recurrent_halting_stack
] = False
TOKEN_MIXER_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _token_mixer_recurrent_halting_stack
] = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _token_mixer_recurrent_halting_stack
] = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _token_mixer_recurrent_halting_stack
] = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _token_mixer_recurrent_halting_stack
] = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _token_mixer_recurrent_halting_stack
] = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_BIAS_FLAG: Annotated[
    bool | None, _token_mixer_recurrent_halting_stack
] = None

_channel_mixer_gate = Section(("Channel Mixer Gate Options",))
# - If `CHANNEL_MIXER_STACK_GATE_FLAG` is False, the gate-specific parameters below are ignored.
CHANNEL_MIXER_STACK_GATE_FLAG: Annotated[bool, _channel_mixer_gate] = False
CHANNEL_MIXER_GATE_OPTION: Annotated[LayerGateOptions | None, _channel_mixer_gate] = (
    LayerGateOptions.MULTIPLIER
)
CHANNEL_MIXER_GATE_ACTIVATION: Annotated[
    ActivationOptions | None, _channel_mixer_gate
] = ActivationOptions.SIGMOID
_channel_mixer_gate_stack = _channel_mixer_gate.child(
    "Channel Mixer Gate Stack Options"
)
# - If False, gate stack options inherit the channel mixer stack options.
CHANNEL_MIXER_GATE_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _channel_mixer_gate_stack
] = False
CHANNEL_MIXER_GATE_STACK_HIDDEN_DIM: Annotated[
    int | None, _channel_mixer_gate_stack
] = None
CHANNEL_MIXER_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _channel_mixer_gate_stack
] = None
CHANNEL_MIXER_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _channel_mixer_gate_stack
] = None
CHANNEL_MIXER_GATE_STACK_NUM_LAYERS: Annotated[
    int | None, _channel_mixer_gate_stack
] = None
CHANNEL_MIXER_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _channel_mixer_gate_stack
] = ActivationOptions.TANH
CHANNEL_MIXER_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _channel_mixer_gate_stack
] = None
CHANNEL_MIXER_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _channel_mixer_gate_stack
] = False
CHANNEL_MIXER_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _channel_mixer_gate_stack
] = None
CHANNEL_MIXER_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _channel_mixer_gate_stack
] = None
CHANNEL_MIXER_GATE_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _channel_mixer_gate_stack
] = None
CHANNEL_MIXER_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _channel_mixer_gate_stack
] = None
CHANNEL_MIXER_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _channel_mixer_gate_stack
] = True
CHANNEL_MIXER_GATE_STACK_BIAS_FLAG: Annotated[
    bool | None, _channel_mixer_gate_stack
] = True

_channel_mixer_halting = Section(("Channel Mixer Halting Options",))
# - If `CHANNEL_MIXER_STACK_HALTING_FLAG` is False, the halting-specific parameters below are ignored.
CHANNEL_MIXER_STACK_HALTING_FLAG: Annotated[bool, _channel_mixer_halting] = False
CHANNEL_MIXER_HALTING_OPTION: Annotated[type[HaltingConfig], _channel_mixer_halting] = (
    StickBreakingConfig
)
CHANNEL_MIXER_HALTING_THRESHOLD: Annotated[float, _channel_mixer_halting] = 0.999
CHANNEL_MIXER_HALTING_DROPOUT: Annotated[float, _channel_mixer_halting] = 0.0
CHANNEL_MIXER_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _channel_mixer_halting
] = HaltingHiddenStateModeOptions.RAW
_channel_mixer_halting_stack = _channel_mixer_halting.child(
    "Channel Mixer Halting Stack Options"
)
# - If False, halting stack options inherit the channel mixer stack options.
CHANNEL_MIXER_HALTING_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _channel_mixer_halting_stack
] = False
CHANNEL_MIXER_HALTING_STACK_HIDDEN_DIM: Annotated[
    int | None, _channel_mixer_halting_stack
] = None
CHANNEL_MIXER_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _channel_mixer_halting_stack
] = LayerNormPositionOptions.DISABLED
CHANNEL_MIXER_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _channel_mixer_halting_stack
] = None
CHANNEL_MIXER_HALTING_STACK_NUM_LAYERS: Annotated[
    int | None, _channel_mixer_halting_stack
] = None
CHANNEL_MIXER_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _channel_mixer_halting_stack
] = None
CHANNEL_MIXER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _channel_mixer_halting_stack
] = None
CHANNEL_MIXER_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _channel_mixer_halting_stack
] = False
CHANNEL_MIXER_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _channel_mixer_halting_stack
] = None
CHANNEL_MIXER_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _channel_mixer_halting_stack
] = None
CHANNEL_MIXER_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _channel_mixer_halting_stack
] = None
CHANNEL_MIXER_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _channel_mixer_halting_stack
] = LastLayerBiasOptions.DISABLED
CHANNEL_MIXER_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _channel_mixer_halting_stack
] = None
CHANNEL_MIXER_HALTING_STACK_BIAS_FLAG: Annotated[
    bool | None, _channel_mixer_halting_stack
] = None

_channel_mixer_memory = Section(("Channel Mixer Memory Options",))
# - If `CHANNEL_MIXER_MEMORY_FLAG` is False, the memory-specific parameters below are ignored.
CHANNEL_MIXER_MEMORY_FLAG: Annotated[bool, _channel_mixer_memory] = False
CHANNEL_MIXER_MEMORY_OPTION: Annotated[
    type[DynamicMemoryConfig], _channel_mixer_memory
] = GatedResidualDynamicMemoryConfig
CHANNEL_MIXER_MEMORY_POSITION_OPTION: Annotated[
    MemoryPositionOptions, _channel_mixer_memory
] = MemoryPositionOptions.AFTER_AFFINE
CHANNEL_MIXER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: Annotated[
    float | None, _channel_mixer_memory
] = None
CHANNEL_MIXER_MEMORY_NUM_MEMORY_SLOTS: Annotated[int, _channel_mixer_memory] = 4
CHANNEL_MIXER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: Annotated[
    int | None, _channel_mixer_memory
] = None
_channel_mixer_memory_stack = _channel_mixer_memory.child(
    "Channel Mixer Memory Stack Options"
)
# - If False, memory stack options inherit the channel mixer stack options.
CHANNEL_MIXER_MEMORY_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _channel_mixer_memory_stack
] = False
CHANNEL_MIXER_MEMORY_STACK_HIDDEN_DIM: Annotated[
    int | None, _channel_mixer_memory_stack
] = None
CHANNEL_MIXER_MEMORY_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _channel_mixer_memory_stack
] = None
CHANNEL_MIXER_MEMORY_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _channel_mixer_memory_stack
] = None
CHANNEL_MIXER_MEMORY_STACK_NUM_LAYERS: Annotated[
    int | None, _channel_mixer_memory_stack
] = None
CHANNEL_MIXER_MEMORY_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _channel_mixer_memory_stack
] = None
CHANNEL_MIXER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _channel_mixer_memory_stack
] = None
CHANNEL_MIXER_MEMORY_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _channel_mixer_memory_stack
] = False
CHANNEL_MIXER_MEMORY_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _channel_mixer_memory_stack
] = None
CHANNEL_MIXER_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _channel_mixer_memory_stack
] = None
CHANNEL_MIXER_MEMORY_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _channel_mixer_memory_stack
] = None
CHANNEL_MIXER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _channel_mixer_memory_stack
] = None
CHANNEL_MIXER_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _channel_mixer_memory_stack
] = None
CHANNEL_MIXER_MEMORY_STACK_BIAS_FLAG: Annotated[
    bool | None, _channel_mixer_memory_stack
] = None

_channel_mixer_recurrent_layer = Section(("Channel Mixer Recurrent Layer Options",))
# - If `CHANNEL_MIXER_RECURRENT_FLAG` is False, the recurrent-specific parameters below are ignored.
CHANNEL_MIXER_RECURRENT_FLAG: Annotated[bool, _channel_mixer_recurrent_layer] = False
CHANNEL_MIXER_RECURRENT_MAX_STEPS: Annotated[int, _channel_mixer_recurrent_layer] = 2
CHANNEL_MIXER_RECURRENT_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _channel_mixer_recurrent_layer
] = LayerNormPositionOptions.DISABLED
CHANNEL_MIXER_RECURRENT_NORMALIZATION: Annotated[
    NormalizationOptions, _channel_mixer_recurrent_layer
] = NormalizationOptions.LAYER_NORM
CHANNEL_MIXER_RECURRENT_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _channel_mixer_recurrent_layer
] = None
CHANNEL_MIXER_RECURRENT_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _channel_mixer_recurrent_layer
] = False
CHANNEL_MIXER_RECURRENT_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _channel_mixer_recurrent_layer
] = None
CHANNEL_MIXER_RECURRENT_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _channel_mixer_recurrent_layer
] = None
_channel_mixer_recurrent_gate = _channel_mixer_recurrent_layer.child(
    "Channel Mixer Recurrent Gate Options"
)
CHANNEL_MIXER_RECURRENT_STACK_GATE_FLAG: Annotated[
    bool, _channel_mixer_recurrent_gate
] = False
CHANNEL_MIXER_RECURRENT_GATE_OPTION: Annotated[
    LayerGateOptions | None, _channel_mixer_recurrent_gate
] = LayerGateOptions.MULTIPLIER
CHANNEL_MIXER_RECURRENT_GATE_ACTIVATION: Annotated[
    ActivationOptions | None, _channel_mixer_recurrent_gate
] = ActivationOptions.SIGMOID
_channel_mixer_recurrent_gate_stack = _channel_mixer_recurrent_gate.child(
    "Channel Mixer Recurrent Gate Stack Options"
)
# - If False, recurrent gate stack options inherit the channel mixer stack options.
CHANNEL_MIXER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _channel_mixer_recurrent_gate_stack
] = False
CHANNEL_MIXER_RECURRENT_GATE_STACK_HIDDEN_DIM: Annotated[
    int | None, _channel_mixer_recurrent_gate_stack
] = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _channel_mixer_recurrent_gate_stack
] = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _channel_mixer_recurrent_gate_stack
] = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_NUM_LAYERS: Annotated[
    int | None, _channel_mixer_recurrent_gate_stack
] = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _channel_mixer_recurrent_gate_stack
] = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _channel_mixer_recurrent_gate_stack
] = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _channel_mixer_recurrent_gate_stack
] = False
CHANNEL_MIXER_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _channel_mixer_recurrent_gate_stack
] = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _channel_mixer_recurrent_gate_stack
] = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _channel_mixer_recurrent_gate_stack
] = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _channel_mixer_recurrent_gate_stack
] = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _channel_mixer_recurrent_gate_stack
] = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_BIAS_FLAG: Annotated[
    bool | None, _channel_mixer_recurrent_gate_stack
] = None
_channel_mixer_recurrent_halting = _channel_mixer_recurrent_layer.child(
    "Channel Mixer Recurrent Halting Options"
)
CHANNEL_MIXER_RECURRENT_STACK_HALTING_FLAG: Annotated[
    bool, _channel_mixer_recurrent_halting
] = False
CHANNEL_MIXER_RECURRENT_HALTING_OPTION: Annotated[
    type[HaltingConfig], _channel_mixer_recurrent_halting
] = StickBreakingConfig
CHANNEL_MIXER_RECURRENT_HALTING_THRESHOLD: Annotated[
    float, _channel_mixer_recurrent_halting
] = 0.999
CHANNEL_MIXER_RECURRENT_HALTING_DROPOUT: Annotated[
    float, _channel_mixer_recurrent_halting
] = 0.0
CHANNEL_MIXER_RECURRENT_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _channel_mixer_recurrent_halting
] = HaltingHiddenStateModeOptions.RAW
_channel_mixer_recurrent_halting_stack = _channel_mixer_recurrent_halting.child(
    "Channel Mixer Recurrent Halting Stack Options"
)
# - If False, recurrent halting stack options inherit the channel mixer stack options.
CHANNEL_MIXER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _channel_mixer_recurrent_halting_stack
] = False
CHANNEL_MIXER_RECURRENT_HALTING_STACK_HIDDEN_DIM: Annotated[
    int | None, _channel_mixer_recurrent_halting_stack
] = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _channel_mixer_recurrent_halting_stack
] = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _channel_mixer_recurrent_halting_stack
] = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_NUM_LAYERS: Annotated[
    int | None, _channel_mixer_recurrent_halting_stack
] = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _channel_mixer_recurrent_halting_stack
] = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _channel_mixer_recurrent_halting_stack
] = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _channel_mixer_recurrent_halting_stack
] = False
CHANNEL_MIXER_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _channel_mixer_recurrent_halting_stack
] = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _channel_mixer_recurrent_halting_stack
] = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _channel_mixer_recurrent_halting_stack
] = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _channel_mixer_recurrent_halting_stack
] = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _channel_mixer_recurrent_halting_stack
] = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_BIAS_FLAG: Annotated[
    bool | None, _channel_mixer_recurrent_halting_stack
] = None

_adaptive_expert_parameters_these_options_apply_inside_experts_only_routers = Section(
    ("Adaptive Expert Parameters. These Options Apply Inside Experts Only; Routers,",)
)
# patch projection, and classification remain ordinary linear layers.
ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 32
ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = LayerNormPositionOptions.BEFORE
ADAPTIVE_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = NormalizationOptions.RMS_NORM
ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 2
ADAPTIVE_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = ActivationOptions.GELU
ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
ADAPTIVE_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
ADAPTIVE_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
ADAPTIVE_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 0.0
ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = LastLayerBiasOptions.DEFAULT
ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
ADAPTIVE_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = True

WEIGHT_OPTION_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
WEIGHT_OPTION: Annotated[
    type[DynamicWeightConfig] | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
GENERATOR_DEPTH: Annotated[
    int, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 3
WEIGHT_DECAY_SCHEDULE: Annotated[
    WeightDecayScheduleOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = WeightDecayScheduleOptions.DISABLED
WEIGHT_DECAY_RATE: Annotated[
    float, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 0.0
WEIGHT_DECAY_WARMUP_BATCHES: Annotated[
    int, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 0
WEIGHT_NORMALIZATION_OPTION: Annotated[
    WeightNormalizationOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = WeightNormalizationOptions.DISABLED
WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
WEIGHT_NORMALIZATION_POSITION_OPTION: Annotated[
    WeightNormalizationPositionOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
WEIGHT_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = BankExpansionFactorOptions.FACTOR_OF_THREE
WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
WEIGHT_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
WEIGHT_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
WEIGHT_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
WEIGHT_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
WEIGHT_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None

BIAS_OPTION_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
BIAS_OPTION: Annotated[
    type[DynamicBiasConfig] | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
BIAS_DECAY_SCHEDULE: Annotated[
    WeightDecayScheduleOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = WeightDecayScheduleOptions.DISABLED
BIAS_DECAY_RATE: Annotated[
    float, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 0.0
BIAS_DECAY_WARMUP_BATCHES: Annotated[
    int, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 0
BIAS_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = BankExpansionFactorOptions.FACTOR_OF_TWO
BIAS_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
BIAS_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
BIAS_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
BIAS_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
BIAS_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
BIAS_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
BIAS_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None

DIAGONAL_OPTION_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = True
DIAGONAL_OPTION: Annotated[
    type[DynamicDiagonalConfig] | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = StandardDynamicDiagonalConfig
DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
DIAGONAL_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
DIAGONAL_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
DIAGONAL_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
DIAGONAL_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
DIAGONAL_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None

MASK_OPTION_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
ROW_MASK_OPTION: Annotated[
    type[AxisMaskConfig] | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
MASK_THRESHOLD: Annotated[
    float, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 0.5
MASK_FLOOR: Annotated[
    float, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 0.0
MASK_TRANSITION_WIDTH: Annotated[
    float, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 0.1
MASK_SURROGATE_SCALE: Annotated[
    float, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 10.0
MASK_DIMENSION_OPTION: Annotated[
    MaskDimensionOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = MaskDimensionOptions.COLUMN
MASK_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
MASK_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
MASK_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
MASK_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
MASK_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
MASK_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = False
MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
MASK_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None
MASK_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = None

# Expert routing
TOP_K: Annotated[
    int, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 2
NUM_EXPERTS: Annotated[
    int, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 4
CAPACITY_FACTOR: Annotated[
    float, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = 0.0
DROPPED_TOKEN_BEHAVIOR: Annotated[
    DroppedTokenOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = DroppedTokenOptions.IDENTITY
WEIGHTED_PARAMETERS_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = True
COMPUTE_EXPERT_MIXTURE_FLAG: Annotated[
    bool, _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers
] = True
WEIGHTING_POSITION_OPTION: Annotated[
    ExpertWeightingPositionOptions,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = ExpertWeightingPositionOptions.AFTER_EXPERTS
ROUTING_INITIALIZATION_MODE: Annotated[
    RoutingInitializationMode,
    _adaptive_expert_parameters_these_options_apply_inside_experts_only_routers,
] = RoutingInitializationMode.LAYER

_sampler = Section(("Sampler",))
SAMPLER_THRESHOLD: Annotated[float, _sampler] = 0.0
SAMPLER_FILTER_ABOVE_THRESHOLD: Annotated[bool, _sampler] = False
SAMPLER_NUM_TOPK_SAMPLES: Annotated[int, _sampler] = 0
SAMPLER_NORMALIZE_PROBABILITIES_FLAG: Annotated[bool, _sampler] = True
SAMPLER_NOISY_TOPK_FLAG: Annotated[bool, _sampler] = False
SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: Annotated[float, _sampler] = 0.0
SAMPLER_SWITCH_LOSS_WEIGHT: Annotated[float, _sampler] = 0.0
SAMPLER_ZERO_CENTRED_LOSS_WEIGHT: Annotated[float, _sampler] = 0.0
SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT: Annotated[float, _sampler] = 0.0

_router = Section(("Router",))
ROUTER_NOISY_TOPK_FLAG: Annotated[bool, _router] = False
ROUTER_STACK_HIDDEN_DIM: Annotated[int, _router] = 32
ROUTER_STACK_NUM_LAYERS: Annotated[int, _router] = 2
ROUTER_STACK_ACTIVATION: Annotated[ActivationOptions, _router] = ActivationOptions.GELU
ROUTER_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _router
] = None
ROUTER_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _router] = False
ROUTER_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _router] = None
ROUTER_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _router] = None
ROUTER_STACK_DROPOUT_PROBABILITY: Annotated[float, _router] = 0.0
ROUTER_STACK_LAYER_NORM_POSITION: Annotated[LayerNormPositionOptions, _router] = (
    LayerNormPositionOptions.DISABLED
)
ROUTER_STACK_NORMALIZATION: Annotated[NormalizationOptions, _router] = (
    NormalizationOptions.RMS_NORM
)
ROUTER_STACK_LAST_LAYER_BIAS_OPTION: Annotated[LastLayerBiasOptions, _router] = (
    LastLayerBiasOptions.DEFAULT
)
ROUTER_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool, _router] = False
ROUTER_BIAS_FLAG: Annotated[bool, _router] = STACK_BIAS_FLAG

_experts = Section(("Experts",))
EXPERT_STACK_HIDDEN_DIM: Annotated[int, _experts] = 32
EXPERT_STACK_NUM_LAYERS: Annotated[int, _experts] = 1
EXPERT_STACK_ACTIVATION: Annotated[ActivationOptions, _experts] = ActivationOptions.GELU
EXPERT_STACK_DROPOUT_PROBABILITY: Annotated[float, _experts] = 0.0
EXPERT_STACK_LAYER_NORM_POSITION: Annotated[LayerNormPositionOptions, _experts] = (
    LayerNormPositionOptions.DISABLED
)
EXPERT_STACK_NORMALIZATION: Annotated[NormalizationOptions, _experts] = (
    NormalizationOptions.RMS_NORM
)
EXPERT_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _experts
] = None
EXPERT_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _experts] = False
EXPERT_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _experts] = None
EXPERT_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _experts] = None
EXPERT_STACK_LAST_LAYER_BIAS_OPTION: Annotated[LastLayerBiasOptions, _experts] = (
    LastLayerBiasOptions.DEFAULT
)
EXPERT_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool, _experts] = False
EXPERT_BIAS_FLAG: Annotated[bool, _experts] = True

EXPERT_STACK_GATE_FLAG: Annotated[bool, _experts] = False
EXPERT_GATE_OPTION: Annotated[LayerGateOptions | None, _experts] = (
    LayerGateOptions.MULTIPLIER
)
EXPERT_GATE_ACTIVATION: Annotated[ActivationOptions | None, _experts] = (
    ActivationOptions.SIGMOID
)
EXPERT_GATE_STACK_INDEPENDENT_FLAG: Annotated[bool, _experts] = False
EXPERT_GATE_STACK_HIDDEN_DIM: Annotated[int | None, _experts] = None
EXPERT_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _experts
] = None
EXPERT_GATE_STACK_NORMALIZATION: Annotated[NormalizationOptions | None, _experts] = None
EXPERT_GATE_STACK_NUM_LAYERS: Annotated[int | None, _experts] = None
EXPERT_GATE_STACK_ACTIVATION: Annotated[ActivationOptions | None, _experts] = (
    ActivationOptions.TANH
)
EXPERT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _experts
] = None
EXPERT_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _experts] = False
EXPERT_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _experts] = None
EXPERT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _experts] = None
EXPERT_GATE_STACK_DROPOUT_PROBABILITY: Annotated[float | None, _experts] = None
EXPERT_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _experts
] = None
EXPERT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool | None, _experts] = (
    True
)
EXPERT_GATE_STACK_BIAS_FLAG: Annotated[bool | None, _experts] = True

EXPERT_STACK_HALTING_FLAG: Annotated[bool, _experts] = False
EXPERT_HALTING_OPTION: Annotated[type[HaltingConfig], _experts] = StickBreakingConfig
EXPERT_HALTING_THRESHOLD: Annotated[float, _experts] = 0.999
EXPERT_HALTING_DROPOUT: Annotated[float, _experts] = 0.0
EXPERT_HALTING_HIDDEN_STATE_MODE: Annotated[HaltingHiddenStateModeOptions, _experts] = (
    HaltingHiddenStateModeOptions.RAW
)
EXPERT_HALTING_STACK_INDEPENDENT_FLAG: Annotated[bool, _experts] = False
EXPERT_HALTING_STACK_HIDDEN_DIM: Annotated[int | None, _experts] = None
EXPERT_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _experts
] = None
EXPERT_HALTING_STACK_NORMALIZATION: Annotated[NormalizationOptions | None, _experts] = (
    None
)
EXPERT_HALTING_STACK_NUM_LAYERS: Annotated[int | None, _experts] = None
EXPERT_HALTING_STACK_ACTIVATION: Annotated[ActivationOptions | None, _experts] = None
EXPERT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _experts
] = None
EXPERT_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _experts] = False
EXPERT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _experts] = None
EXPERT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _experts] = None
EXPERT_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[float | None, _experts] = None
EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _experts
] = None
EXPERT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _experts
] = None
EXPERT_HALTING_STACK_BIAS_FLAG: Annotated[bool | None, _experts] = None

EXPERT_MEMORY_FLAG: Annotated[bool, _experts] = False
EXPERT_MEMORY_OPTION: Annotated[type[DynamicMemoryConfig], _experts] = (
    GatedResidualDynamicMemoryConfig
)
EXPERT_MEMORY_POSITION_OPTION: Annotated[MemoryPositionOptions, _experts] = (
    MemoryPositionOptions.AFTER_AFFINE
)
EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: Annotated[float | None, _experts] = None
EXPERT_MEMORY_NUM_MEMORY_SLOTS: Annotated[int, _experts] = 4
EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: Annotated[int | None, _experts] = None
EXPERT_MEMORY_STACK_INDEPENDENT_FLAG: Annotated[bool, _experts] = False
EXPERT_MEMORY_STACK_HIDDEN_DIM: Annotated[int | None, _experts] = None
EXPERT_MEMORY_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _experts
] = None
EXPERT_MEMORY_STACK_NORMALIZATION: Annotated[NormalizationOptions | None, _experts] = (
    None
)
EXPERT_MEMORY_STACK_NUM_LAYERS: Annotated[int | None, _experts] = None
EXPERT_MEMORY_STACK_ACTIVATION: Annotated[ActivationOptions | None, _experts] = None
EXPERT_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _experts
] = None
EXPERT_MEMORY_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _experts] = False
EXPERT_MEMORY_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _experts] = None
EXPERT_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _experts] = None
EXPERT_MEMORY_STACK_DROPOUT_PROBABILITY: Annotated[float | None, _experts] = None
EXPERT_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _experts
] = None
EXPERT_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _experts
] = None
EXPERT_MEMORY_STACK_BIAS_FLAG: Annotated[bool | None, _experts] = None

EXPERT_RECURRENT_FLAG: Annotated[bool, _experts] = False
EXPERT_RECURRENT_MAX_STEPS: Annotated[int, _experts] = 3
EXPERT_RECURRENT_LAYER_NORM_POSITION: Annotated[LayerNormPositionOptions, _experts] = (
    LayerNormPositionOptions.DISABLED
)
EXPERT_RECURRENT_NORMALIZATION: Annotated[NormalizationOptions, _experts] = (
    NormalizationOptions.LAYER_NORM
)
EXPERT_RECURRENT_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _experts
] = None
EXPERT_RECURRENT_RESIDUAL_MODEL_FLAG: Annotated[bool, _experts] = False
EXPERT_RECURRENT_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _experts] = None
EXPERT_RECURRENT_RESIDUAL_RMS_NORM_EPSILON: Annotated[float | None, _experts] = None
EXPERT_RECURRENT_STACK_GATE_FLAG: Annotated[bool, _experts] = False
EXPERT_RECURRENT_GATE_OPTION: Annotated[LayerGateOptions | None, _experts] = (
    LayerGateOptions.MULTIPLIER
)
EXPERT_RECURRENT_GATE_ACTIVATION: Annotated[ActivationOptions | None, _experts] = (
    ActivationOptions.SIGMOID
)
EXPERT_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: Annotated[bool, _experts] = False
EXPERT_RECURRENT_GATE_STACK_HIDDEN_DIM: Annotated[int | None, _experts] = None
EXPERT_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _experts
] = None
EXPERT_RECURRENT_GATE_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _experts
] = None
EXPERT_RECURRENT_GATE_STACK_NUM_LAYERS: Annotated[int | None, _experts] = None
EXPERT_RECURRENT_GATE_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _experts
] = None
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _experts
] = None
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _experts] = False
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _experts] = None
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _experts
] = None
EXPERT_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: Annotated[float | None, _experts] = (
    None
)
EXPERT_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _experts
] = None
EXPERT_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _experts
] = None
EXPERT_RECURRENT_GATE_STACK_BIAS_FLAG: Annotated[bool | None, _experts] = None
EXPERT_RECURRENT_STACK_HALTING_FLAG: Annotated[bool, _experts] = False
EXPERT_RECURRENT_HALTING_OPTION: Annotated[type[HaltingConfig], _experts] = (
    StickBreakingConfig
)
EXPERT_RECURRENT_HALTING_THRESHOLD: Annotated[float, _experts] = 0.999
EXPERT_RECURRENT_HALTING_DROPOUT: Annotated[float, _experts] = 0.0
EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _experts
] = HaltingHiddenStateModeOptions.RAW
EXPERT_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: Annotated[bool, _experts] = False
EXPERT_RECURRENT_HALTING_STACK_HIDDEN_DIM: Annotated[int | None, _experts] = None
EXPERT_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _experts
] = None
EXPERT_RECURRENT_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _experts
] = None
EXPERT_RECURRENT_HALTING_STACK_NUM_LAYERS: Annotated[int | None, _experts] = None
EXPERT_RECURRENT_HALTING_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _experts
] = None
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _experts
] = None
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _experts] = False
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _experts] = (
    None
)
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _experts
] = None
EXPERT_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _experts
] = None
EXPERT_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _experts
] = None
EXPERT_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _experts
] = None
EXPERT_RECURRENT_HALTING_STACK_BIAS_FLAG: Annotated[bool | None, _experts] = None


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


_channel_mixer_grouping = Section(("Channel Mixer Grouping",))
CHANNEL_MIXER_GROUPING_SCOPE: Annotated[
    AdaptiveParameterGroupingScopeOptions | None, _channel_mixer_grouping
] = None
CHANNEL_MIXER_GROUP_COUNT: Annotated[int | None, _channel_mixer_grouping] = None
CHANNEL_MIXER_CHUNK_SIZE: Annotated[int | None, _channel_mixer_grouping] = None
CHANNEL_MIXER_GROUPING_SEQUENCE_LENGTH: Annotated[
    int | None, _channel_mixer_grouping
] = None
CHANNEL_MIXER_GROUPING_INPUT_ORDER: Annotated[
    AdaptiveParameterInputOrderOptions | None, _channel_mixer_grouping
] = None
CHANNEL_MIXER_GROUPING_METHOD: Annotated[
    type[GroupingConfig] | None, _channel_mixer_grouping
] = None
CHANNEL_MIXER_GROUPING_SUMMARY_NORMALIZATION: Annotated[
    SummaryNormalizationOptions | None, _channel_mixer_grouping
] = None
CHANNEL_MIXER_GROUPING_MODEL_CONFIG: Annotated[
    ConfigBase | None, _channel_mixer_grouping
] = None
CHANNEL_MIXER_GROUPING_ATTENTION_HIDDEN_DIM: Annotated[
    int | None, _channel_mixer_grouping
] = None
CHANNEL_MIXER_GROUPING_RMS_NORM_EPSILON: Annotated[
    float | None, _channel_mixer_grouping
] = None

soft_moe.declare_soft_moe_runtime_defaults(globals())


_optional_sequence_expansion = Section(("Optional Sequence Expansion",))
SEQUENCE_EXPANSION_FLAG: Annotated[bool, _optional_sequence_expansion] = False
SEQUENCE_EXPANSION_CONFIG: Annotated[
    SequenceExpansionConfig | None, _optional_sequence_expansion
] = None
SEQUENCE_EXPANSION_MAX_NEW_TOKENS: Annotated[
    int | None, _optional_sequence_expansion
] = None
SEQUENCE_EXPANSION_MODEL_CONFIG: Annotated[
    ConfigBase | None, _optional_sequence_expansion
] = None
SEQUENCE_EXPANSION_POSITIONAL_EMBEDDING_CONFIG: Annotated[
    TextLearnedPositionalEmbeddingConfig
    | TextSinusoidalPositionalEmbeddingConfig
    | None,
    _optional_sequence_expansion,
] = None

_hierarchical_masked_word = Section(("Hierarchical Masked-Word Options",))
# Selected by the text dataset.
HIERARCHICAL_MASKED_WORD_FLAG: Annotated[bool, _hierarchical_masked_word] = False
SEQUENCE_LENGTH: Annotated[int, _hierarchical_masked_word] = 100
HIERARCHICAL_BYTE_EMBEDDING_DIM: Annotated[int, _hierarchical_masked_word] = 32
HIERARCHICAL_BYTE_ENCODER_NUM_LAYERS: Annotated[int, _hierarchical_masked_word] = 1
HIERARCHICAL_BYTE_FEED_FORWARD_DIM: Annotated[int, _hierarchical_masked_word] = 64
HIERARCHICAL_BYTE_NUM_HEADS: Annotated[int, _hierarchical_masked_word] = 4
HIERARCHICAL_MAX_TOKEN_BYTES: Annotated[int, _hierarchical_masked_word] = 64
EMBEDDING_DROPOUT_PROBABILITY: Annotated[float, _hierarchical_masked_word] = 0.0

_token_mixer_source_memory = Section(("Token Mixer Source Memory Options",))
# Optional chunk-compressed source memory (None retains per-token memory).
TOKEN_MIXER_SOURCE_MEMORY_CHUNK_SIZE: Annotated[
    int | None, _token_mixer_source_memory
] = None
TOKEN_MIXER_SOURCE_MEMORY_GROUPING_METHOD: Annotated[
    type[GroupingConfig], _token_mixer_source_memory
] = SumGroupingConfig
TOKEN_MIXER_SOURCE_MEMORY_MODEL_CONFIG: Annotated[
    ConfigBase | None, _token_mixer_source_memory
] = None

# Recurrent Layer Options
## Recurrent Halting Options
# Defaults preserve the image model's behavior.
RECURRENT_MIN_STEPS: Annotated[int, _recurrent_halting] = 1
RECURRENT_PONDER_COST_WEIGHT: Annotated[float, _recurrent_halting] = 1.0

_component_defaults.declare_component_defaults(globals())

_recurrent_compression.declare_compression_options(globals())
