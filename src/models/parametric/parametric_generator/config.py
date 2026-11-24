from typing import Annotated

from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,  # noqa: F401
    AttentionResidualConfig,  # noqa: F401
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
    WeightedBlendResidualConfig,  # noqa: F401
    WeightedResidualConfig,  # noqa: F401
)
from emperor.parametric import ClipParameterOptions, GeneratorBiasMixtureConfig
from model_runtime.packages.configuration_metadata import Section
from model_runtime.packages.metric_direction import MetricDirection
from model_runtime.packages.runtime_values import (
    RuntimeValueConstraints,
    positive_runtime_fields,
)
from models import _recurrent_variants

_trainer = Section(("Trainer",))
TRAINER_ACCELERATOR: Annotated[str, _trainer] = "auto"
TRAINER_DEVICES: Annotated[str | int, _trainer] = "auto"
TRAINER_GRADIENT_CLIP_VAL: Annotated[float, _trainer] = 0.0
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
BATCH_SIZE: Annotated[int, _general] = 64
LEARNING_RATE: Annotated[float, _general] = 1e-3
NUM_EPOCHS: Annotated[int, _general] = 2

_model = Section(("Model",))
INPUT_DIM: Annotated[int, _model] = 28**2
HIDDEN_DIM: Annotated[int, _model] = 32
RUNTIME_VALUE_CONSTRAINTS: Annotated[RuntimeValueConstraints, _model] = (
    positive_runtime_fields("HIDDEN_DIM")
)
OUTPUT_DIM: Annotated[int, _model] = 10

_layer_stack = Section(("Layer Stack Options",))
# - hidden_dim comes from the global HIDDEN_DIM field above.
STACK_NUM_LAYERS: Annotated[int, _layer_stack] = 1
STACK_ACTIVATION: Annotated[ActivationOptions, _layer_stack] = ActivationOptions.GELU
STACK_DROPOUT_PROBABILITY: Annotated[float, _layer_stack] = 0.0

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
# - If False, residual stack options inherit layer stack options.
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

_parametric_generator_mixture = Section(("Parametric Generator Mixture Options",))
ADAPTIVE_MIXTURE_TOP_K: Annotated[int, _parametric_generator_mixture] = 1
ADAPTIVE_MIXTURE_NUM_EXPERTS: Annotated[int, _parametric_generator_mixture] = 2
ADAPTIVE_MIXTURE_WEIGHTED_PARAMETERS_FLAG: Annotated[
    bool, _parametric_generator_mixture
] = False
ADAPTIVE_MIXTURE_CLIP_PARAMETER_OPTION: Annotated[
    ClipParameterOptions, _parametric_generator_mixture
] = ClipParameterOptions.DISABLED
ADAPTIVE_MIXTURE_CLIP_RANGE: Annotated[float, _parametric_generator_mixture] = 5.0
ADAPTIVE_BIAS_OPTION: Annotated[
    type[GeneratorBiasMixtureConfig] | None, _parametric_generator_mixture
] = None

_sampler_model = Section(("Sampler Model Options",))
SAMPLER_THRESHOLD: Annotated[float, _sampler_model] = 0.0
SAMPLER_FILTER_ABOVE_THRESHOLD: Annotated[bool, _sampler_model] = False
SAMPLER_NUM_TOPK_SAMPLES: Annotated[int, _sampler_model] = 0
SAMPLER_NORMALIZE_PROBABILITIES_FLAG: Annotated[bool, _sampler_model] = False
SAMPLER_NOISY_TOPK_FLAG: Annotated[bool, _sampler_model] = False
SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: Annotated[float, _sampler_model] = 0.0
SAMPLER_SWITCH_LOSS_WEIGHT: Annotated[float, _sampler_model] = 0.0
SAMPLER_ZERO_CENTRED_LOSS_WEIGHT: Annotated[float, _sampler_model] = 0.0
SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT: Annotated[float, _sampler_model] = 0.0

_generator_expert_stack = Section(("Generator Expert Stack Options",))
GENERATOR_STACK_NUM_LAYERS: Annotated[int, _generator_expert_stack] = 2
GENERATOR_STACK_HIDDEN_DIM: Annotated[int, _generator_expert_stack] = 32
GENERATOR_STACK_ACTIVATION: Annotated[ActivationOptions, _generator_expert_stack] = (
    ActivationOptions.GELU
)
GENERATOR_STACK_DROPOUT_PROBABILITY: Annotated[float, _generator_expert_stack] = 0.0

# Hyperparameter search space

_recurrent_layer = Section(("Recurrent Layer Options",))
RECURRENT_FLAG: Annotated[bool, _recurrent_layer] = False

_recurrent_variants.declare_variant_options(
    globals(), _recurrent_variants.ParametricRecurrentOptions
)
