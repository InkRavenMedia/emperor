from dataclasses import dataclass, field

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
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from models.neuron.linear._hidden import SETTINGS_VALIDATION
from models.neuron.linear._residual import ResidualStackOptions


@dataclass(frozen=True, slots=True)
class MainStackOptions:
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
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.stack(self)


@dataclass(frozen=True, slots=True)
class ControllerStackOptions:
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
    dropout_probability: float
    bias_flag: bool
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.positive_integer("stack_hidden_dim", self.hidden_dim)
        SETTINGS_VALIDATION.stack(self)


@dataclass(frozen=True, slots=True)
class GateOptions:
    enabled: bool
    option: LayerGateOptions | None
    activation: ActivationOptions | None
    stack: ControllerStackOptions
    shared_config: GateConfig | None = None

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.gate(self.enabled, self.option)


@dataclass(frozen=True, slots=True)
class HaltingOptions:
    enabled: bool
    threshold: float
    dropout_probability: float
    hidden_state_mode: HaltingHiddenStateModeOptions
    stack: ControllerStackOptions

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.halting_threshold("halting_threshold", self.threshold)
        SETTINGS_VALIDATION.dropout("halting_dropout", self.dropout_probability)


@dataclass(frozen=True, slots=True)
class MemoryOptions:
    enabled: bool
    implementation: type[DynamicMemoryConfig]
    position: MemoryPositionOptions
    test_time_training_learning_rate: float | None
    test_time_training_num_inner_steps: int | None
    stack: ControllerStackOptions

    def __post_init__(self) -> None:
        if self.test_time_training_learning_rate is not None:
            SETTINGS_VALIDATION.positive_float(
                "memory_test_time_training_learning_rate",
                self.test_time_training_learning_rate,
            )
        if self.test_time_training_num_inner_steps is not None:
            SETTINGS_VALIDATION.positive_integer(
                "memory_test_time_training_num_inner_steps",
                self.test_time_training_num_inner_steps,
            )


@dataclass(frozen=True, slots=True)
class RecurrenceOptions:
    enabled: bool
    max_steps: int
    initial_iterations: int = field(default=2, kw_only=True)
    gradient_transition_count: int | None = field(default=None, kw_only=True)
    no_gradient_transition_count: int | None = field(default=None, kw_only=True)
    iteration_increment: int = field(default=1, kw_only=True)
    forward_calls_before_iteration_increment: int = field(default=1, kw_only=True)
    smooth_iteration_growth_flag: bool = field(default=False, kw_only=True)
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    gate: GateOptions
    halting: HaltingOptions

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.positive_integer("recurrent_max_steps", self.max_steps)


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    batch_size: int
    learning_rate: float
    input_dim: int
    hidden_dim: int
    output_dim: int
    stack: MainStackOptions
    gate: GateOptions
    halting: HaltingOptions
    memory: MemoryOptions
    recurrence: RecurrenceOptions
    halting_option: type[HaltingConfig] = StickBreakingConfig
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.dimensions(self)
        SETTINGS_VALIDATION.shared_gate(self.gate.enabled, self.gate.shared_config)
