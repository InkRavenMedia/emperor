from typing import Annotated

# ruff: noqa: E402
from emperor.layers import (
    AdditiveResidualConfig,  # noqa: F401
    AttentionResidualConfig,  # noqa: F401
    NormalizationOptions,
    ResidualConfig,
    WeightedBlendResidualConfig,  # noqa: F401
    WeightedResidualConfig,  # noqa: F401
)
from model_runtime.packages.configuration_metadata import Section
from model_runtime.packages.metric_direction import MetricDirection
from model_runtime.packages.runtime_values import (
    RuntimeValueConstraints,
    positive_runtime_fields,
)
from models import _recurrent_variants

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
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
)
from emperor.memory import (
    AttentionDynamicMemoryConfig,  # noqa: F401
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,  # noqa: F401
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,  # noqa: F401
)

_general = Section(("Global",))
INPUT_DIM: Annotated[int, _general] = 28**2
HIDDEN_DIM: Annotated[int, _general] = 32
RUNTIME_VALUE_CONSTRAINTS: Annotated[RuntimeValueConstraints, _general] = (
    positive_runtime_fields("HIDDEN_DIM")
)
OUTPUT_DIM: Annotated[int, _general] = 10
BATCH_SIZE: Annotated[int, _general] = 128
LEARNING_RATE: Annotated[float, _general] = 1e-3
NUM_EPOCHS: Annotated[int, _general] = 30

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
HALTING_OUTPUT_DIM: Annotated[int, _halting] = 2
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

_mixture_of_experts_model = Section(("Mixture Of Experts Model Options",))
TOP_K: Annotated[int, _mixture_of_experts_model] = 2
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
    LAYER_NORM_POSITION
)
ROUTER_STACK_NORMALIZATION: Annotated[NormalizationOptions, _router_stack] = (
    NORMALIZATION
)
ROUTER_STACK_LAST_LAYER_BIAS_OPTION: Annotated[LastLayerBiasOptions, _router_stack] = (
    LastLayerBiasOptions.DEFAULT
)
ROUTER_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[bool, _router_stack] = False
ROUTER_BIAS_FLAG: Annotated[bool, _router_stack] = STACK_BIAS_FLAG

_neuron_wrapper = Section(("Neuron Wrapper Options",))
from emperor.neuron import (
    NeuronClusterOptimizerSyncCallback,
    TerminalRangeOptions,
    TerminalRoutingTreeDepthOptions,
)

CALLBACK_NEURON_CLUSTER_OPTIMIZER_SYNC: Annotated[
    NeuronClusterOptimizerSyncCallback, _neuron_wrapper
] = NeuronClusterOptimizerSyncCallback()


_cluster_geometry = _neuron_wrapper.child("Cluster Geometry Options")
CLUSTER_X_AXIS_TOTAL_NEURONS: Annotated[int, _cluster_geometry] = 10
CLUSTER_Y_AXIS_TOTAL_NEURONS: Annotated[int, _cluster_geometry] = 10
CLUSTER_Z_AXIS_TOTAL_NEURONS: Annotated[int, _cluster_geometry] = 1
CLUSTER_INITIAL_X_AXIS_TOTAL_NEURONS: Annotated[int, _cluster_geometry] = 3
CLUSTER_INITIAL_Y_AXIS_TOTAL_NEURONS: Annotated[int, _cluster_geometry] = 3
CLUSTER_INITIAL_Z_AXIS_TOTAL_NEURONS: Annotated[int, _cluster_geometry] = 1
CLUSTER_MAX_STEPS: Annotated[int, _cluster_geometry] = 4
CLUSTER_BEAM_WIDTH: Annotated[int | None, _cluster_geometry] = None
CLUSTER_GROWTH_THRESHOLD: Annotated[int | None, _cluster_geometry] = 250
CLUSTER_GROWTH_COOLDOWN_STEPS: Annotated[int | None, _cluster_geometry] = None
CLUSTER_MAX_TOTAL_GROWTHS: Annotated[int | None, _cluster_geometry] = None
CLUSTER_GROWTH_WARMUP_STEPS: Annotated[int | None, _cluster_geometry] = None
CLUSTER_PRUNING_THRESHOLD: Annotated[int | None, _cluster_geometry] = None
CLUSTER_ESCAPE_DRIVEN_GROWTH_FLAG: Annotated[bool, _cluster_geometry] = False
CLUSTER_MITOSIS_INITIALIZATION_FLAG: Annotated[bool, _cluster_geometry] = False

_cluster_terminal = _neuron_wrapper.child("Cluster Terminal Options")
CLUSTER_TERMINAL_XY_AXIS_RANGE: Annotated[TerminalRangeOptions, _cluster_terminal] = (
    TerminalRangeOptions.ONE
)
CLUSTER_TERMINAL_Z_AXIS_RANGE: Annotated[TerminalRangeOptions, _cluster_terminal] = (
    TerminalRangeOptions.ONE
)
CLUSTER_TERMINAL_TOP_K: Annotated[int, _cluster_terminal] = 1
CLUSTER_TERMINAL_ROUTING_TREE_DEPTH: Annotated[
    TerminalRoutingTreeDepthOptions | None, _cluster_terminal
] = None
CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_1_BRANCH_COUNT: Annotated[
    int | None, _cluster_terminal
] = None
CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_1_TOP_K: Annotated[
    int | None, _cluster_terminal
] = None
CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_2_BRANCH_COUNT: Annotated[
    int | None, _cluster_terminal
] = None
CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_2_TOP_K: Annotated[
    int | None, _cluster_terminal
] = None

_cluster_terminal_router = _cluster_terminal.child("Cluster Terminal Router Options")
CLUSTER_TERMINAL_ROUTER_NUM_LAYERS: Annotated[int, _cluster_terminal_router] = 1
CLUSTER_TERMINAL_ROUTER_HIDDEN_DIM: Annotated[int, _cluster_terminal_router] = (
    HIDDEN_DIM
)
CLUSTER_TERMINAL_ROUTER_ACTIVATION: Annotated[
    ActivationOptions, _cluster_terminal_router
] = ActivationOptions.DISABLED
CLUSTER_TERMINAL_ROUTER_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _cluster_terminal_router
] = LayerNormPositionOptions.DISABLED
CLUSTER_TERMINAL_ROUTER_NORMALIZATION: Annotated[
    NormalizationOptions, _cluster_terminal_router
] = NormalizationOptions.RMS_NORM
CLUSTER_TERMINAL_ROUTER_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _cluster_terminal_router
] = None
CLUSTER_TERMINAL_ROUTER_RESIDUAL_MODEL_FLAG: Annotated[
    bool, _cluster_terminal_router
] = False
CLUSTER_TERMINAL_ROUTER_RESIDUAL_BLOCK_SIZE: Annotated[
    int | None, _cluster_terminal_router
] = None
CLUSTER_TERMINAL_ROUTER_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _cluster_terminal_router
] = None
CLUSTER_TERMINAL_ROUTER_DROPOUT_PROBABILITY: Annotated[
    float, _cluster_terminal_router
] = 0.0
CLUSTER_TERMINAL_ROUTER_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions, _cluster_terminal_router
] = LastLayerBiasOptions.DEFAULT
CLUSTER_TERMINAL_ROUTER_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool, _cluster_terminal_router
] = False
CLUSTER_TERMINAL_ROUTER_BIAS_FLAG: Annotated[bool, _cluster_terminal_router] = True

_cluster_terminal_sampler = _cluster_terminal.child("Cluster Terminal Sampler Options")
CLUSTER_TERMINAL_SAMPLER_THRESHOLD: Annotated[float, _cluster_terminal_sampler] = 0.0
CLUSTER_TERMINAL_SAMPLER_FILTER_ABOVE_THRESHOLD: Annotated[
    bool, _cluster_terminal_sampler
] = False
CLUSTER_TERMINAL_SAMPLER_NUM_TOPK_SAMPLES: Annotated[int, _cluster_terminal_sampler] = 0
CLUSTER_TERMINAL_SAMPLER_NORMALIZE_PROBABILITIES_FLAG: Annotated[
    bool, _cluster_terminal_sampler
] = False
CLUSTER_TERMINAL_SAMPLER_NOISY_TOPK_FLAG: Annotated[bool, _cluster_terminal_sampler] = (
    False
)
CLUSTER_TERMINAL_SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: Annotated[
    float, _cluster_terminal_sampler
] = 0.0
CLUSTER_TERMINAL_SAMPLER_SWITCH_LOSS_WEIGHT: Annotated[
    float, _cluster_terminal_sampler
] = 0.0
CLUSTER_TERMINAL_SAMPLER_ZERO_CENTRED_LOSS_WEIGHT: Annotated[
    float, _cluster_terminal_sampler
] = 0.0
CLUSTER_TERMINAL_SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT: Annotated[
    float, _cluster_terminal_sampler
] = 0.0

_cluster_halting = _neuron_wrapper.child("Cluster Halting Options")
CLUSTER_HALTING_FLAG: Annotated[bool, _cluster_halting] = True
CLUSTER_HALTING_OPTION: Annotated[type[HaltingConfig], _cluster_halting] = (
    StickBreakingConfig
)
CLUSTER_HALTING_THRESHOLD: Annotated[float, _cluster_halting] = 0.95
CLUSTER_HALTING_DROPOUT: Annotated[float, _cluster_halting] = 0.0
CLUSTER_HALTING_HIDDEN_STATE_MODE: Annotated[
    HaltingHiddenStateModeOptions, _cluster_halting
] = HaltingHiddenStateModeOptions.RAW
CLUSTER_HALTING_STACK_HIDDEN_DIM: Annotated[int, _cluster_halting] = HIDDEN_DIM
CLUSTER_HALTING_OUTPUT_DIM: Annotated[int, _cluster_halting] = 2
CLUSTER_HALTING_STACK_LAYER_NORM_POSITION: Annotated[
    LayerNormPositionOptions, _cluster_halting
] = LayerNormPositionOptions.DISABLED
CLUSTER_HALTING_STACK_NORMALIZATION: Annotated[
    NormalizationOptions, _cluster_halting
] = NormalizationOptions.RMS_NORM
CLUSTER_HALTING_STACK_NUM_LAYERS: Annotated[int, _cluster_halting] = 1
CLUSTER_HALTING_STACK_ACTIVATION: Annotated[ActivationOptions, _cluster_halting] = (
    ActivationOptions.DISABLED
)
CLUSTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: Annotated[
    type[ResidualConfig] | None, _cluster_halting
] = None
CLUSTER_HALTING_STACK_RESIDUAL_MODEL_FLAG: Annotated[bool, _cluster_halting] = False
CLUSTER_HALTING_STACK_RESIDUAL_BLOCK_SIZE: Annotated[int | None, _cluster_halting] = (
    None
)
CLUSTER_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: Annotated[
    float | None, _cluster_halting
] = None
CLUSTER_HALTING_STACK_DROPOUT_PROBABILITY: Annotated[float, _cluster_halting] = 0.0
CLUSTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION: Annotated[
    LastLayerBiasOptions, _cluster_halting
] = LastLayerBiasOptions.DISABLED
CLUSTER_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: Annotated[
    bool, _cluster_halting
] = False
CLUSTER_HALTING_STACK_BIAS_FLAG: Annotated[bool, _cluster_halting] = True

_recurrent_variants.declare_variant_options(
    globals(), _recurrent_variants.RecurrentResidualOptions
)
