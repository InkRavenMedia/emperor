from typing import Annotated

from emperor.augmentations.adaptive_parameters import (
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
from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,  # noqa: F401
    AttentionResidualConfig,  # noqa: F401
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
    WeightedBlendResidualConfig,  # noqa: F401
    WeightedResidualConfig,  # noqa: F401
)
from emperor.memory import (
    AttentionDynamicMemoryConfig,  # noqa: F401
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,  # noqa: F401
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,  # noqa: F401
)
from model_runtime.packages.configuration_metadata import Section
from model_runtime.packages.metric_direction import MetricDirection
from model_runtime.packages.runtime_values import (
    RuntimeValueConstraints,
    positive_runtime_fields,
)
from models import _component_defaults

# Package-local defaults keep each model package independently loadable.
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

_general = Section(("Global",))
BATCH_SIZE: Annotated[int, _general] = 128
NUM_EPOCHS: Annotated[int, _general] = 10
INPUT_DIM: Annotated[int, _general] = 28**2
HIDDEN_DIM: Annotated[int, _general] = 32
RUNTIME_VALUE_CONSTRAINTS: Annotated[RuntimeValueConstraints, _general] = (
    positive_runtime_fields("HIDDEN_DIM")
)
OUTPUT_DIM: Annotated[int, _general] = 10
LEARNING_RATE: Annotated[float, _general] = 1e-3

_layer_stack = Section(("Layer Stack Options",))
# - hidden_dim comes from the global HIDDEN_DIM field above.
LAYER_NORM_POSITION: Annotated[LayerNormPositionOptions, _layer_stack] = (
    LayerNormPositionOptions.BEFORE
)
NORMALIZATION: Annotated[NormalizationOptions, _layer_stack] = (
    NormalizationOptions.RMS_NORM
)
STACK_NUM_LAYERS: Annotated[int, _layer_stack] = 5
STACK_ACTIVATION: Annotated[ActivationOptions, _layer_stack] = ActivationOptions.GELU
STACK_DROPOUT_PROBABILITY: Annotated[float, _layer_stack] = 0.0
STACK_LAST_LAYER_BIAS_OPTION: Annotated[LastLayerBiasOptions, _layer_stack] = (
    LastLayerBiasOptions.DEFAULT
)
STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool, _layer_stack] = True
STACK_BIAS_FLAG: Annotated[bool, _layer_stack] = True

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
] = LAYER_NORM_POSITION
ADAPTIVE_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions, _adaptive_generator_stack
] = NORMALIZATION
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
ADAPTIVE_GENERATOR_STACK_BIAS_FLAG: Annotated[bool, _adaptive_generator_stack] = (
    STACK_BIAS_FLAG
)

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
# If `STACK_GATE_FLAG` is False, the gate-specific parameters below are ignored.
STACK_GATE_FLAG: Annotated[bool, _gate] = False
GATE_OPTION: Annotated[LayerGateOptions | None, _gate] = LayerGateOptions.MULTIPLIER
GATE_ACTIVATION: Annotated[ActivationOptions | None, _gate] = ActivationOptions.SIGMOID
_gate_stack = _gate.child("Gate Stack Options")
# If False, gate model stack options inherit the layer stack submodule options.
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
# If `STACK_HALTING_FLAG` is False, the halting-specific parameters below are ignored.
STACK_HALTING_FLAG: Annotated[bool, _halting] = False
HALTING_OPTION: Annotated[type[HaltingConfig], _halting] = StickBreakingConfig
HALTING_THRESHOLD: Annotated[float, _halting] = 0.999
HALTING_DROPOUT: Annotated[float, _halting] = 0.0
HALTING_HIDDEN_STATE_MODE: Annotated[HaltingHiddenStateModeOptions, _halting] = (
    HaltingHiddenStateModeOptions.RAW
)
_halting_stack = _halting.child("Halting Stack Options")
# If False, halting model stack options inherit the layer stack submodule options.
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
# If `MEMORY_FLAG` is False, the memory-specific parameters below are ignored.
MEMORY_FLAG: Annotated[bool, _memory] = False
MEMORY_OPTION: Annotated[type[DynamicMemoryConfig], _memory] = (
    GatedResidualDynamicMemoryConfig
)
MEMORY_POSITION_OPTION: Annotated[MemoryPositionOptions, _memory] = (
    MemoryPositionOptions.AFTER_AFFINE
)
MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: Annotated[float | None, _memory] = None
MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: Annotated[int | None, _memory] = None
_memory_stack = _memory.child("Memory Stack Options")
# If False, memory model stack options inherit the layer stack submodule options.
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
# If `RECURRENT_FLAG` is False, the recurrent-specific parameters below are ignored.
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
# If False, recurrent gate stack options inherit gate/submodule stack options.
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
# If False, recurrent halting stack options inherit halting/submodule stack options.
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


_input_boundary_model = Section(("Input Boundary Model Options",))
# Input boundary dynamic weight options.
INPUT_LAYER_WEIGHT_OPTION: Annotated[
    type[DynamicWeightConfig] | None, _input_boundary_model
] = None
INPUT_LAYER_GENERATOR_DEPTH: Annotated[int, _input_boundary_model] = GENERATOR_DEPTH
INPUT_LAYER_WEIGHT_DECAY_SCHEDULE: Annotated[
    WeightDecayScheduleOptions, _input_boundary_model
] = WEIGHT_DECAY_SCHEDULE
INPUT_LAYER_WEIGHT_DECAY_RATE: Annotated[float, _input_boundary_model] = (
    WEIGHT_DECAY_RATE
)
INPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES: Annotated[int, _input_boundary_model] = (
    WEIGHT_DECAY_WARMUP_BATCHES
)
INPUT_LAYER_WEIGHT_NORMALIZATION_OPTION: Annotated[
    WeightNormalizationOptions, _input_boundary_model
] = WEIGHT_NORMALIZATION_OPTION
INPUT_LAYER_WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG: Annotated[
    bool, _input_boundary_model
] = WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG
INPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION: Annotated[
    WeightNormalizationPositionOptions, _input_boundary_model
] = WEIGHT_NORMALIZATION_POSITION_OPTION
INPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions, _input_boundary_model
] = WEIGHT_BANK_EXPANSION_FACTOR
# Input boundary dynamic bias options.
INPUT_LAYER_BIAS_OPTION: Annotated[
    type[DynamicBiasConfig] | None, _input_boundary_model
] = None
INPUT_LAYER_BIAS_DECAY_SCHEDULE: Annotated[
    WeightDecayScheduleOptions, _input_boundary_model
] = BIAS_DECAY_SCHEDULE
INPUT_LAYER_BIAS_DECAY_RATE: Annotated[float, _input_boundary_model] = BIAS_DECAY_RATE
INPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES: Annotated[int, _input_boundary_model] = (
    BIAS_DECAY_WARMUP_BATCHES
)
INPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions, _input_boundary_model
] = BIAS_BANK_EXPANSION_FACTOR
# Input boundary dynamic diagonal options.
INPUT_LAYER_DIAGONAL_OPTION: Annotated[
    type[DynamicDiagonalConfig] | None, _input_boundary_model
] = None
# Input boundary dynamic mask options.
INPUT_LAYER_ROW_MASK_OPTION: Annotated[
    type[AxisMaskConfig] | None, _input_boundary_model
] = None
INPUT_LAYER_MASK_THRESHOLD: Annotated[float, _input_boundary_model] = MASK_THRESHOLD
INPUT_LAYER_MASK_FLOOR: Annotated[float, _input_boundary_model] = MASK_FLOOR
INPUT_LAYER_MASK_TRANSITION_WIDTH: Annotated[float, _input_boundary_model] = (
    MASK_TRANSITION_WIDTH
)
INPUT_LAYER_MASK_SURROGATE_SCALE: Annotated[float, _input_boundary_model] = (
    MASK_SURROGATE_SCALE
)
INPUT_LAYER_MASK_DIMENSION_OPTION: Annotated[
    MaskDimensionOptions, _input_boundary_model
] = MASK_DIMENSION_OPTION

_output_boundary_model = Section(("Output Boundary Model Options",))
# Output boundary dynamic weight options.
OUTPUT_LAYER_WEIGHT_OPTION: Annotated[
    type[DynamicWeightConfig] | None, _output_boundary_model
] = None
OUTPUT_LAYER_GENERATOR_DEPTH: Annotated[int, _output_boundary_model] = GENERATOR_DEPTH
OUTPUT_LAYER_WEIGHT_DECAY_SCHEDULE: Annotated[
    WeightDecayScheduleOptions, _output_boundary_model
] = WEIGHT_DECAY_SCHEDULE
OUTPUT_LAYER_WEIGHT_DECAY_RATE: Annotated[float, _output_boundary_model] = (
    WEIGHT_DECAY_RATE
)
OUTPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES: Annotated[int, _output_boundary_model] = (
    WEIGHT_DECAY_WARMUP_BATCHES
)
OUTPUT_LAYER_WEIGHT_NORMALIZATION_OPTION: Annotated[
    WeightNormalizationOptions, _output_boundary_model
] = WEIGHT_NORMALIZATION_OPTION
OUTPUT_LAYER_WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG: Annotated[
    bool, _output_boundary_model
] = WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG
OUTPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION: Annotated[
    WeightNormalizationPositionOptions, _output_boundary_model
] = WEIGHT_NORMALIZATION_POSITION_OPTION  # noqa: E501
OUTPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions, _output_boundary_model
] = WEIGHT_BANK_EXPANSION_FACTOR
# Output boundary dynamic bias options.
OUTPUT_LAYER_BIAS_OPTION: Annotated[
    type[DynamicBiasConfig] | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_DECAY_SCHEDULE: Annotated[
    WeightDecayScheduleOptions, _output_boundary_model
] = BIAS_DECAY_SCHEDULE
OUTPUT_LAYER_BIAS_DECAY_RATE: Annotated[float, _output_boundary_model] = BIAS_DECAY_RATE
OUTPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES: Annotated[int, _output_boundary_model] = (
    BIAS_DECAY_WARMUP_BATCHES
)
OUTPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR: Annotated[
    BankExpansionFactorOptions, _output_boundary_model
] = BIAS_BANK_EXPANSION_FACTOR
# Output boundary dynamic diagonal options.
OUTPUT_LAYER_DIAGONAL_OPTION: Annotated[
    type[DynamicDiagonalConfig] | None, _output_boundary_model
] = None
# Output boundary dynamic mask options.
OUTPUT_LAYER_ROW_MASK_OPTION: Annotated[
    type[AxisMaskConfig] | None, _output_boundary_model
] = None
OUTPUT_LAYER_MASK_THRESHOLD: Annotated[float, _output_boundary_model] = MASK_THRESHOLD
OUTPUT_LAYER_MASK_FLOOR: Annotated[float, _output_boundary_model] = MASK_FLOOR
OUTPUT_LAYER_MASK_TRANSITION_WIDTH: Annotated[float, _output_boundary_model] = (
    MASK_TRANSITION_WIDTH
)
OUTPUT_LAYER_MASK_SURROGATE_SCALE: Annotated[float, _output_boundary_model] = (
    MASK_SURROGATE_SCALE
)
OUTPUT_LAYER_MASK_DIMENSION_OPTION: Annotated[
    MaskDimensionOptions, _output_boundary_model
] = MASK_DIMENSION_OPTION

# Contextual rank modulation and complete parameter sets hidden
WEIGHT_INPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_NUM_EXPERTS: Annotated[int | None, _output_boundary_model] = None
BIAS_MIXTURE_NUM_EXPERTS: Annotated[int | None, _output_boundary_model] = None
WEIGHT_MIXTURE_TOP_K: Annotated[int | None, _output_boundary_model] = None
BIAS_MIXTURE_TOP_K: Annotated[int | None, _output_boundary_model] = None
WEIGHT_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
BIAS_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
WEIGHT_COEFFICIENT_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None

# Contextual rank modulation and complete parameter sets INPUT_LAYER_
INPUT_LAYER_WEIGHT_INPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_NUM_EXPERTS: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_NUM_EXPERTS: Annotated[int | None, _output_boundary_model] = (
    None
)
INPUT_LAYER_WEIGHT_MIXTURE_TOP_K: Annotated[int | None, _output_boundary_model] = None
INPUT_LAYER_BIAS_MIXTURE_TOP_K: Annotated[int | None, _output_boundary_model] = None
INPUT_LAYER_WEIGHT_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
INPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
INPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None

# Contextual rank modulation and complete parameter sets OUTPUT_LAYER_
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_SOURCE: Annotated[
    LowRankFactorSourceOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_NUM_EXPERTS: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_NUM_EXPERTS: Annotated[int | None, _output_boundary_model] = (
    None
)
OUTPUT_LAYER_WEIGHT_MIXTURE_TOP_K: Annotated[int | None, _output_boundary_model] = None
OUTPUT_LAYER_BIAS_MIXTURE_TOP_K: Annotated[int | None, _output_boundary_model] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_COEFFICIENT_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: Annotated[
    bool, _output_boundary_model
] = False
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: Annotated[
    NormalizationOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: Annotated[
    ActivationOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _output_boundary_model
] = False
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[
    float | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
OUTPUT_LAYER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None
OUTPUT_LAYER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: Annotated[
    bool | None, _output_boundary_model
] = None


_main_adaptive_parameter_grouping = Section(("Main Adaptive Parameter Grouping",))
# Choose group_count or chunk_size; internal padding never changes real row counts.
GROUPING_SCOPE: Annotated[
    AdaptiveParameterGroupingScopeOptions | None, _main_adaptive_parameter_grouping
] = None
GROUP_COUNT: Annotated[int | None, _main_adaptive_parameter_grouping] = None
CHUNK_SIZE: Annotated[int | None, _main_adaptive_parameter_grouping] = None
GROUPING_SEQUENCE_LENGTH: Annotated[int | None, _main_adaptive_parameter_grouping] = (
    None
)
GROUPING_INPUT_ORDER: Annotated[
    AdaptiveParameterInputOrderOptions | None, _main_adaptive_parameter_grouping
] = None
GROUPING_METHOD: Annotated[
    type[GroupingConfig] | None, _main_adaptive_parameter_grouping
] = None
GROUPING_SUMMARY_NORMALIZATION: Annotated[
    SummaryNormalizationOptions | None, _main_adaptive_parameter_grouping
] = None
GROUPING_MODEL_CONFIG: Annotated[
    ConfigBase | None, _main_adaptive_parameter_grouping
] = None
GROUPING_ATTENTION_HIDDEN_DIM: Annotated[
    int | None, _main_adaptive_parameter_grouping
] = None
GROUPING_RMS_NORM_EPSILON: Annotated[
    float | None, _main_adaptive_parameter_grouping
] = None


_input_layer_adaptive_parameter_grouping = Section(
    ("Input Layer Adaptive Parameter Grouping",)
)
# Choose group_count or chunk_size; internal padding never changes real row counts.
INPUT_LAYER_GROUPING_SCOPE: Annotated[
    AdaptiveParameterGroupingScopeOptions | None,
    _input_layer_adaptive_parameter_grouping,
] = None
INPUT_LAYER_GROUP_COUNT: Annotated[
    int | None, _input_layer_adaptive_parameter_grouping
] = None
INPUT_LAYER_CHUNK_SIZE: Annotated[
    int | None, _input_layer_adaptive_parameter_grouping
] = None
INPUT_LAYER_GROUPING_SEQUENCE_LENGTH: Annotated[
    int | None, _input_layer_adaptive_parameter_grouping
] = None
INPUT_LAYER_GROUPING_INPUT_ORDER: Annotated[
    AdaptiveParameterInputOrderOptions | None, _input_layer_adaptive_parameter_grouping
] = None
INPUT_LAYER_GROUPING_METHOD: Annotated[
    type[GroupingConfig] | None, _input_layer_adaptive_parameter_grouping
] = None
INPUT_LAYER_GROUPING_SUMMARY_NORMALIZATION: Annotated[
    SummaryNormalizationOptions | None, _input_layer_adaptive_parameter_grouping
] = None
INPUT_LAYER_GROUPING_MODEL_CONFIG: Annotated[
    ConfigBase | None, _input_layer_adaptive_parameter_grouping
] = None
INPUT_LAYER_GROUPING_ATTENTION_HIDDEN_DIM: Annotated[
    int | None, _input_layer_adaptive_parameter_grouping
] = None
INPUT_LAYER_GROUPING_RMS_NORM_EPSILON: Annotated[
    float | None, _input_layer_adaptive_parameter_grouping
] = None


_output_layer_adaptive_parameter_grouping = Section(
    ("Output Layer Adaptive Parameter Grouping",)
)
# Choose group_count or chunk_size; internal padding never changes real row counts.
OUTPUT_LAYER_GROUPING_SCOPE: Annotated[
    AdaptiveParameterGroupingScopeOptions | None,
    _output_layer_adaptive_parameter_grouping,
] = None
OUTPUT_LAYER_GROUP_COUNT: Annotated[
    int | None, _output_layer_adaptive_parameter_grouping
] = None
OUTPUT_LAYER_CHUNK_SIZE: Annotated[
    int | None, _output_layer_adaptive_parameter_grouping
] = None
OUTPUT_LAYER_GROUPING_SEQUENCE_LENGTH: Annotated[
    int | None, _output_layer_adaptive_parameter_grouping
] = None
OUTPUT_LAYER_GROUPING_INPUT_ORDER: Annotated[
    AdaptiveParameterInputOrderOptions | None, _output_layer_adaptive_parameter_grouping
] = None
OUTPUT_LAYER_GROUPING_METHOD: Annotated[
    type[GroupingConfig] | None, _output_layer_adaptive_parameter_grouping
] = None
OUTPUT_LAYER_GROUPING_SUMMARY_NORMALIZATION: Annotated[
    SummaryNormalizationOptions | None, _output_layer_adaptive_parameter_grouping
] = None
OUTPUT_LAYER_GROUPING_MODEL_CONFIG: Annotated[
    ConfigBase | None, _output_layer_adaptive_parameter_grouping
] = None
OUTPUT_LAYER_GROUPING_ATTENTION_HIDDEN_DIM: Annotated[
    int | None, _output_layer_adaptive_parameter_grouping
] = None
OUTPUT_LAYER_GROUPING_RMS_NORM_EPSILON: Annotated[
    float | None, _output_layer_adaptive_parameter_grouping
] = None

_component_defaults.declare_component_defaults(globals())
