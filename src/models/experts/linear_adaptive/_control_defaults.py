from dataclasses import dataclass, field

import models.experts.linear_adaptive.config as config
from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
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
from models._expert_control_defaults import ExpertControlDefaults
from models.experts.linear_adaptive._residual import (
    ResidualStackOptions,
    ResidualStackSource,
    resolve_residual_stack_options,
)
from models.experts.linear_adaptive.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
)


@dataclass(frozen=True, kw_only=True)
class ControlDefaultValues:
    residual_coefficient_activation: ActivationOptions = (
        config.RESIDUAL_COEFFICIENT_ACTIVATION
    )
    residual_stack_independent_flag: bool = config.RESIDUAL_STACK_INDEPENDENT_FLAG
    residual_stack_hidden_dim: int | None = config.RESIDUAL_STACK_HIDDEN_DIM
    residual_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.RESIDUAL_STACK_LAYER_NORM_POSITION
    )
    residual_stack_normalization: NormalizationOptions | None = field(
        default=(config.RESIDUAL_STACK_NORMALIZATION), kw_only=True
    )
    residual_stack_num_layers: int | None = config.RESIDUAL_STACK_NUM_LAYERS
    residual_stack_activation: ActivationOptions | None = (
        config.RESIDUAL_STACK_ACTIVATION
    )
    residual_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION
    )
    residual_stack_residual_model_flag: bool = config.RESIDUAL_STACK_RESIDUAL_MODEL_FLAG
    residual_stack_residual_block_size: int | None = (
        config.RESIDUAL_STACK_RESIDUAL_BLOCK_SIZE
    )
    residual_stack_residual_rms_norm_epsilon: float | None = (
        config.RESIDUAL_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    residual_stack_dropout_probability: float | None = (
        config.RESIDUAL_STACK_DROPOUT_PROBABILITY
    )
    residual_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION
    )
    residual_stack_apply_output_postprocessing_flag: bool | None = (
        config.RESIDUAL_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    residual_stack_bias_flag: bool | None = config.RESIDUAL_STACK_BIAS_FLAG
    expert_stack_gate_flag: bool = config.EXPERT_STACK_GATE_FLAG
    expert_gate_option: LayerGateOptions | None = config.EXPERT_GATE_OPTION
    expert_gate_activation: ActivationOptions | None = config.EXPERT_GATE_ACTIVATION
    expert_gate_stack_independent_flag: bool = config.EXPERT_GATE_STACK_INDEPENDENT_FLAG
    expert_gate_stack_hidden_dim: int | None = config.EXPERT_GATE_STACK_HIDDEN_DIM
    expert_gate_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.EXPERT_GATE_STACK_LAYER_NORM_POSITION
    )
    expert_gate_stack_normalization: NormalizationOptions | None = field(
        default=(config.EXPERT_GATE_STACK_NORMALIZATION), kw_only=True
    )
    expert_gate_stack_num_layers: int | None = config.EXPERT_GATE_STACK_NUM_LAYERS
    expert_gate_stack_activation: ActivationOptions | None = (
        config.EXPERT_GATE_STACK_ACTIVATION
    )
    expert_gate_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.EXPERT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
    )
    expert_gate_stack_residual_model_flag: bool = (
        config.EXPERT_GATE_STACK_RESIDUAL_MODEL_FLAG
    )
    expert_gate_stack_residual_block_size: int | None = (
        config.EXPERT_GATE_STACK_RESIDUAL_BLOCK_SIZE
    )
    expert_gate_stack_residual_rms_norm_epsilon: float | None = (
        config.EXPERT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    expert_gate_stack_dropout_probability: float | None = (
        config.EXPERT_GATE_STACK_DROPOUT_PROBABILITY
    )
    expert_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.EXPERT_GATE_STACK_LAST_LAYER_BIAS_OPTION
    )
    expert_gate_stack_apply_output_postprocessing_flag: bool | None = (
        config.EXPERT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    expert_gate_stack_bias_flag: bool | None = config.EXPERT_GATE_STACK_BIAS_FLAG
    expert_stack_halting_flag: bool = config.EXPERT_STACK_HALTING_FLAG
    expert_halting_threshold: float = config.EXPERT_HALTING_THRESHOLD
    expert_halting_dropout: float = config.EXPERT_HALTING_DROPOUT
    expert_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        config.EXPERT_HALTING_HIDDEN_STATE_MODE
    )
    expert_halting_output_dim: int = config.EXPERT_HALTING_OUTPUT_DIM
    expert_halting_stack_independent_flag: bool = (
        config.EXPERT_HALTING_STACK_INDEPENDENT_FLAG
    )
    expert_halting_stack_hidden_dim: int | None = config.EXPERT_HALTING_STACK_HIDDEN_DIM
    expert_halting_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.EXPERT_HALTING_STACK_LAYER_NORM_POSITION
    )
    expert_halting_stack_normalization: NormalizationOptions | None = field(
        default=(config.EXPERT_HALTING_STACK_NORMALIZATION), kw_only=True
    )
    expert_halting_stack_num_layers: int | None = config.EXPERT_HALTING_STACK_NUM_LAYERS
    expert_halting_stack_activation: ActivationOptions | None = (
        config.EXPERT_HALTING_STACK_ACTIVATION
    )
    expert_halting_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.EXPERT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
    )
    expert_halting_stack_residual_model_flag: bool = (
        config.EXPERT_HALTING_STACK_RESIDUAL_MODEL_FLAG
    )
    expert_halting_stack_residual_block_size: int | None = (
        config.EXPERT_HALTING_STACK_RESIDUAL_BLOCK_SIZE
    )
    expert_halting_stack_residual_rms_norm_epsilon: float | None = (
        config.EXPERT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    expert_halting_stack_dropout_probability: float | None = (
        config.EXPERT_HALTING_STACK_DROPOUT_PROBABILITY
    )
    expert_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
    )
    expert_halting_stack_apply_output_postprocessing_flag: bool | None = (
        config.EXPERT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    expert_halting_stack_bias_flag: bool | None = config.EXPERT_HALTING_STACK_BIAS_FLAG
    expert_memory_flag: bool = config.EXPERT_MEMORY_FLAG
    expert_memory_option: type[DynamicMemoryConfig] = config.EXPERT_MEMORY_OPTION
    expert_memory_position_option: MemoryPositionOptions = (
        config.EXPERT_MEMORY_POSITION_OPTION
    )
    expert_memory_test_time_training_learning_rate: float | None = (
        config.EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
    )
    expert_memory_test_time_training_num_inner_steps: int | None = (
        config.EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
    )
    expert_memory_stack_independent_flag: bool = (
        config.EXPERT_MEMORY_STACK_INDEPENDENT_FLAG
    )
    expert_memory_stack_hidden_dim: int | None = config.EXPERT_MEMORY_STACK_HIDDEN_DIM
    expert_memory_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.EXPERT_MEMORY_STACK_LAYER_NORM_POSITION
    )
    expert_memory_stack_normalization: NormalizationOptions | None = field(
        default=(config.EXPERT_MEMORY_STACK_NORMALIZATION), kw_only=True
    )
    expert_memory_stack_num_layers: int | None = config.EXPERT_MEMORY_STACK_NUM_LAYERS
    expert_memory_stack_activation: ActivationOptions | None = (
        config.EXPERT_MEMORY_STACK_ACTIVATION
    )
    expert_memory_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.EXPERT_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
    )
    expert_memory_stack_residual_model_flag: bool = (
        config.EXPERT_MEMORY_STACK_RESIDUAL_MODEL_FLAG
    )
    expert_memory_stack_residual_block_size: int | None = (
        config.EXPERT_MEMORY_STACK_RESIDUAL_BLOCK_SIZE
    )
    expert_memory_stack_residual_rms_norm_epsilon: float | None = (
        config.EXPERT_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    expert_memory_stack_dropout_probability: float | None = (
        config.EXPERT_MEMORY_STACK_DROPOUT_PROBABILITY
    )
    expert_memory_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.EXPERT_MEMORY_STACK_LAST_LAYER_BIAS_OPTION
    )
    expert_memory_stack_apply_output_postprocessing_flag: bool | None = (
        config.EXPERT_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    expert_memory_stack_bias_flag: bool | None = config.EXPERT_MEMORY_STACK_BIAS_FLAG
    expert_recurrent_flag: bool = config.EXPERT_RECURRENT_FLAG
    expert_recurrent_max_steps: int = config.EXPERT_RECURRENT_MAX_STEPS
    expert_recurrent_layer_norm_position: LayerNormPositionOptions = (
        config.EXPERT_RECURRENT_LAYER_NORM_POSITION
    )
    expert_recurrent_normalization: NormalizationOptions = field(
        default=(config.EXPERT_RECURRENT_NORMALIZATION), kw_only=True
    )
    expert_recurrent_stack_gate_flag: bool = config.EXPERT_RECURRENT_STACK_GATE_FLAG
    expert_recurrent_gate_option: LayerGateOptions | None = (
        config.EXPERT_RECURRENT_GATE_OPTION
    )
    expert_recurrent_gate_activation: ActivationOptions | None = (
        config.EXPERT_RECURRENT_GATE_ACTIVATION
    )
    expert_recurrent_gate_stack_independent_flag: bool = (
        config.EXPERT_RECURRENT_GATE_STACK_INDEPENDENT_FLAG
    )
    expert_recurrent_gate_stack_hidden_dim: int | None = (
        config.EXPERT_RECURRENT_GATE_STACK_HIDDEN_DIM
    )
    expert_recurrent_gate_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.EXPERT_RECURRENT_GATE_STACK_LAYER_NORM_POSITION
    )
    expert_recurrent_gate_stack_normalization: NormalizationOptions | None = field(
        default=(config.EXPERT_RECURRENT_GATE_STACK_NORMALIZATION), kw_only=True
    )
    expert_recurrent_gate_stack_num_layers: int | None = (
        config.EXPERT_RECURRENT_GATE_STACK_NUM_LAYERS
    )
    expert_recurrent_gate_stack_activation: ActivationOptions | None = (
        config.EXPERT_RECURRENT_GATE_STACK_ACTIVATION
    )
    expert_recurrent_gate_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
    expert_recurrent_gate_stack_residual_model_flag: bool = (
        config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG
    )
    expert_recurrent_gate_stack_residual_block_size: int | None = (
        config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE
    )
    expert_recurrent_gate_stack_residual_rms_norm_epsilon: float | None = (
        config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    expert_recurrent_gate_stack_dropout_probability: float | None = (
        config.EXPERT_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY
    )
    expert_recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.EXPERT_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
    )
    expert_recurrent_gate_stack_apply_output_postprocessing_flag: bool | None = (
        config.EXPERT_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    expert_recurrent_gate_stack_bias_flag: bool | None = (
        config.EXPERT_RECURRENT_GATE_STACK_BIAS_FLAG
    )
    expert_recurrent_stack_halting_flag: bool = (
        config.EXPERT_RECURRENT_STACK_HALTING_FLAG
    )
    expert_recurrent_halting_threshold: float = (
        config.EXPERT_RECURRENT_HALTING_THRESHOLD
    )
    expert_recurrent_halting_dropout: float = config.EXPERT_RECURRENT_HALTING_DROPOUT
    expert_recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        config.EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE
    )
    expert_recurrent_halting_stack_independent_flag: bool = (
        config.EXPERT_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG
    )
    expert_recurrent_halting_stack_hidden_dim: int | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_HIDDEN_DIM
    )
    expert_recurrent_halting_stack_layer_norm_position: (
        LayerNormPositionOptions | None
    ) = config.EXPERT_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION
    expert_recurrent_halting_stack_normalization: NormalizationOptions | None = field(
        default=config.EXPERT_RECURRENT_HALTING_STACK_NORMALIZATION, kw_only=True
    )
    expert_recurrent_halting_stack_num_layers: int | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_NUM_LAYERS
    )
    expert_recurrent_halting_stack_activation: ActivationOptions | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_ACTIVATION
    )
    expert_recurrent_halting_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
    expert_recurrent_halting_stack_residual_model_flag: bool = (
        config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG
    )
    expert_recurrent_halting_stack_residual_block_size: int | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE
    )
    expert_recurrent_halting_stack_residual_rms_norm_epsilon: float | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    expert_recurrent_halting_stack_dropout_probability: float | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY
    )
    expert_recurrent_halting_stack_last_layer_bias_option: (
        LastLayerBiasOptions | None
    ) = config.EXPERT_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
    expert_recurrent_halting_stack_apply_output_postprocessing_flag: bool | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    expert_recurrent_halting_stack_bias_flag: bool | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_BIAS_FLAG
    )
    router_noisy_topk_flag: bool = config.ROUTER_NOISY_TOPK_FLAG
    router_stack_hidden_dim: int = config.ROUTER_STACK_HIDDEN_DIM
    router_stack_num_layers: int = config.ROUTER_STACK_NUM_LAYERS
    router_stack_activation: ActivationOptions = config.ROUTER_STACK_ACTIVATION
    router_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION
    )
    router_stack_residual_model_flag: bool = config.ROUTER_STACK_RESIDUAL_MODEL_FLAG
    router_stack_residual_block_size: int | None = (
        config.ROUTER_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_stack_dropout_probability: float = config.ROUTER_STACK_DROPOUT_PROBABILITY
    router_stack_layer_norm_position: LayerNormPositionOptions = (
        config.ROUTER_STACK_LAYER_NORM_POSITION
    )
    router_stack_normalization: NormalizationOptions = field(
        default=(config.ROUTER_STACK_NORMALIZATION), kw_only=True
    )
    router_stack_last_layer_bias_option: LastLayerBiasOptions = (
        config.ROUTER_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_stack_apply_output_postprocessing_flag: bool = (
        config.ROUTER_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_bias_flag: bool = config.ROUTER_BIAS_FLAG
    router_stack_gate_flag: bool = config.ROUTER_STACK_GATE_FLAG
    router_gate_option: LayerGateOptions | None = config.ROUTER_GATE_OPTION
    router_gate_activation: ActivationOptions | None = config.ROUTER_GATE_ACTIVATION
    router_gate_stack_independent_flag: bool = config.ROUTER_GATE_STACK_INDEPENDENT_FLAG
    router_gate_stack_hidden_dim: int | None = config.ROUTER_GATE_STACK_HIDDEN_DIM
    router_gate_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.ROUTER_GATE_STACK_LAYER_NORM_POSITION
    )
    router_gate_stack_normalization: NormalizationOptions | None = field(
        default=(config.ROUTER_GATE_STACK_NORMALIZATION), kw_only=True
    )
    router_gate_stack_num_layers: int | None = config.ROUTER_GATE_STACK_NUM_LAYERS
    router_gate_stack_activation: ActivationOptions | None = (
        config.ROUTER_GATE_STACK_ACTIVATION
    )
    router_gate_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.ROUTER_GATE_STACK_RESIDUAL_CONNECTION_OPTION
    )
    router_gate_stack_residual_model_flag: bool = (
        config.ROUTER_GATE_STACK_RESIDUAL_MODEL_FLAG
    )
    router_gate_stack_residual_block_size: int | None = (
        config.ROUTER_GATE_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_gate_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_gate_stack_dropout_probability: float | None = (
        config.ROUTER_GATE_STACK_DROPOUT_PROBABILITY
    )
    router_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.ROUTER_GATE_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_gate_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_gate_stack_bias_flag: bool | None = config.ROUTER_GATE_STACK_BIAS_FLAG
    router_stack_halting_flag: bool = config.ROUTER_STACK_HALTING_FLAG
    router_halting_threshold: float = config.ROUTER_HALTING_THRESHOLD
    router_halting_dropout: float = config.ROUTER_HALTING_DROPOUT
    router_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        config.ROUTER_HALTING_HIDDEN_STATE_MODE
    )
    router_halting_output_dim: int = config.ROUTER_HALTING_OUTPUT_DIM
    router_halting_stack_independent_flag: bool = (
        config.ROUTER_HALTING_STACK_INDEPENDENT_FLAG
    )
    router_halting_stack_hidden_dim: int | None = config.ROUTER_HALTING_STACK_HIDDEN_DIM
    router_halting_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.ROUTER_HALTING_STACK_LAYER_NORM_POSITION
    )
    router_halting_stack_normalization: NormalizationOptions | None = field(
        default=(config.ROUTER_HALTING_STACK_NORMALIZATION), kw_only=True
    )
    router_halting_stack_num_layers: int | None = config.ROUTER_HALTING_STACK_NUM_LAYERS
    router_halting_stack_activation: ActivationOptions | None = (
        config.ROUTER_HALTING_STACK_ACTIVATION
    )
    router_halting_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.ROUTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
    )
    router_halting_stack_residual_model_flag: bool = (
        config.ROUTER_HALTING_STACK_RESIDUAL_MODEL_FLAG
    )
    router_halting_stack_residual_block_size: int | None = (
        config.ROUTER_HALTING_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_halting_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_halting_stack_dropout_probability: float | None = (
        config.ROUTER_HALTING_STACK_DROPOUT_PROBABILITY
    )
    router_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.ROUTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_halting_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_halting_stack_bias_flag: bool | None = config.ROUTER_HALTING_STACK_BIAS_FLAG
    router_memory_flag: bool = config.ROUTER_MEMORY_FLAG
    router_memory_option: type[DynamicMemoryConfig] = config.ROUTER_MEMORY_OPTION
    router_memory_position_option: MemoryPositionOptions = (
        config.ROUTER_MEMORY_POSITION_OPTION
    )
    router_memory_test_time_training_learning_rate: float | None = (
        config.ROUTER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
    )
    router_memory_test_time_training_num_inner_steps: int | None = (
        config.ROUTER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
    )
    router_memory_stack_independent_flag: bool = (
        config.ROUTER_MEMORY_STACK_INDEPENDENT_FLAG
    )
    router_memory_stack_hidden_dim: int | None = config.ROUTER_MEMORY_STACK_HIDDEN_DIM
    router_memory_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.ROUTER_MEMORY_STACK_LAYER_NORM_POSITION
    )
    router_memory_stack_normalization: NormalizationOptions | None = field(
        default=(config.ROUTER_MEMORY_STACK_NORMALIZATION), kw_only=True
    )
    router_memory_stack_num_layers: int | None = config.ROUTER_MEMORY_STACK_NUM_LAYERS
    router_memory_stack_activation: ActivationOptions | None = (
        config.ROUTER_MEMORY_STACK_ACTIVATION
    )
    router_memory_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.ROUTER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
    )
    router_memory_stack_residual_model_flag: bool = (
        config.ROUTER_MEMORY_STACK_RESIDUAL_MODEL_FLAG
    )
    router_memory_stack_residual_block_size: int | None = (
        config.ROUTER_MEMORY_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_memory_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_memory_stack_dropout_probability: float | None = (
        config.ROUTER_MEMORY_STACK_DROPOUT_PROBABILITY
    )
    router_memory_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.ROUTER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_memory_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_memory_stack_bias_flag: bool | None = config.ROUTER_MEMORY_STACK_BIAS_FLAG
    router_recurrent_flag: bool = config.ROUTER_RECURRENT_FLAG
    router_recurrent_max_steps: int = config.ROUTER_RECURRENT_MAX_STEPS
    router_recurrent_layer_norm_position: LayerNormPositionOptions = (
        config.ROUTER_RECURRENT_LAYER_NORM_POSITION
    )
    router_recurrent_normalization: NormalizationOptions = field(
        default=(config.ROUTER_RECURRENT_NORMALIZATION), kw_only=True
    )
    router_recurrent_stack_gate_flag: bool = config.ROUTER_RECURRENT_STACK_GATE_FLAG
    router_recurrent_gate_option: LayerGateOptions | None = (
        config.ROUTER_RECURRENT_GATE_OPTION
    )
    router_recurrent_gate_activation: ActivationOptions | None = (
        config.ROUTER_RECURRENT_GATE_ACTIVATION
    )
    router_recurrent_gate_stack_independent_flag: bool = (
        config.ROUTER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG
    )
    router_recurrent_gate_stack_hidden_dim: int | None = (
        config.ROUTER_RECURRENT_GATE_STACK_HIDDEN_DIM
    )
    router_recurrent_gate_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.ROUTER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION
    )
    router_recurrent_gate_stack_normalization: NormalizationOptions | None = field(
        default=(config.ROUTER_RECURRENT_GATE_STACK_NORMALIZATION), kw_only=True
    )
    router_recurrent_gate_stack_num_layers: int | None = (
        config.ROUTER_RECURRENT_GATE_STACK_NUM_LAYERS
    )
    router_recurrent_gate_stack_activation: ActivationOptions | None = (
        config.ROUTER_RECURRENT_GATE_STACK_ACTIVATION
    )
    router_recurrent_gate_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
    router_recurrent_gate_stack_residual_model_flag: bool = (
        config.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG
    )
    router_recurrent_gate_stack_residual_block_size: int | None = (
        config.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_recurrent_gate_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_recurrent_gate_stack_dropout_probability: float | None = (
        config.ROUTER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY
    )
    router_recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.ROUTER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_recurrent_gate_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_recurrent_gate_stack_bias_flag: bool | None = (
        config.ROUTER_RECURRENT_GATE_STACK_BIAS_FLAG
    )
    router_recurrent_stack_halting_flag: bool = (
        config.ROUTER_RECURRENT_STACK_HALTING_FLAG
    )
    router_recurrent_halting_threshold: float = (
        config.ROUTER_RECURRENT_HALTING_THRESHOLD
    )
    router_recurrent_halting_dropout: float = config.ROUTER_RECURRENT_HALTING_DROPOUT
    router_recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        config.ROUTER_RECURRENT_HALTING_HIDDEN_STATE_MODE
    )
    router_recurrent_halting_stack_independent_flag: bool = (
        config.ROUTER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG
    )
    router_recurrent_halting_stack_hidden_dim: int | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_HIDDEN_DIM
    )
    router_recurrent_halting_stack_layer_norm_position: (
        LayerNormPositionOptions | None
    ) = config.ROUTER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION
    router_recurrent_halting_stack_normalization: NormalizationOptions | None = field(
        default=config.ROUTER_RECURRENT_HALTING_STACK_NORMALIZATION, kw_only=True
    )
    router_recurrent_halting_stack_num_layers: int | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_NUM_LAYERS
    )
    router_recurrent_halting_stack_activation: ActivationOptions | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_ACTIVATION
    )
    router_recurrent_halting_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
    router_recurrent_halting_stack_residual_model_flag: bool = (
        config.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG
    )
    router_recurrent_halting_stack_residual_block_size: int | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_recurrent_halting_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_recurrent_halting_stack_dropout_probability: float | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY
    )
    router_recurrent_halting_stack_last_layer_bias_option: (
        LastLayerBiasOptions | None
    ) = config.ROUTER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
    router_recurrent_halting_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_recurrent_halting_stack_bias_flag: bool | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_BIAS_FLAG
    )
    stack_gate_flag: bool = config.STACK_GATE_FLAG
    gate_option: LayerGateOptions | None = config.GATE_OPTION
    gate_activation: ActivationOptions | None = config.GATE_ACTIVATION
    gate_stack_independent_flag: bool = config.GATE_STACK_INDEPENDENT_FLAG
    gate_stack_hidden_dim: int | None = config.GATE_STACK_HIDDEN_DIM
    gate_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.GATE_STACK_LAYER_NORM_POSITION
    )
    gate_stack_normalization: NormalizationOptions | None = field(
        default=(config.GATE_STACK_NORMALIZATION), kw_only=True
    )
    gate_stack_num_layers: int | None = config.GATE_STACK_NUM_LAYERS
    gate_stack_activation: ActivationOptions | None = config.GATE_STACK_ACTIVATION
    gate_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.GATE_STACK_RESIDUAL_CONNECTION_OPTION
    )
    gate_stack_residual_model_flag: bool = config.GATE_STACK_RESIDUAL_MODEL_FLAG
    gate_stack_residual_block_size: int | None = config.GATE_STACK_RESIDUAL_BLOCK_SIZE
    gate_stack_residual_rms_norm_epsilon: float | None = (
        config.GATE_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    gate_stack_dropout_probability: float | None = config.GATE_STACK_DROPOUT_PROBABILITY
    gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.GATE_STACK_LAST_LAYER_BIAS_OPTION
    )
    gate_stack_apply_output_postprocessing_flag: bool | None = (
        config.GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    gate_stack_bias_flag: bool | None = config.GATE_STACK_BIAS_FLAG
    stack_halting_flag: bool = config.STACK_HALTING_FLAG
    halting_threshold: float = config.HALTING_THRESHOLD
    halting_dropout: float = config.HALTING_DROPOUT
    halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        config.HALTING_HIDDEN_STATE_MODE
    )
    halting_output_dim: int = config.HALTING_OUTPUT_DIM
    halting_stack_independent_flag: bool = config.HALTING_STACK_INDEPENDENT_FLAG
    halting_stack_hidden_dim: int | None = config.HALTING_STACK_HIDDEN_DIM
    halting_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.HALTING_STACK_LAYER_NORM_POSITION
    )
    halting_stack_normalization: NormalizationOptions | None = field(
        default=(config.HALTING_STACK_NORMALIZATION), kw_only=True
    )
    halting_stack_num_layers: int | None = config.HALTING_STACK_NUM_LAYERS
    halting_stack_activation: ActivationOptions | None = config.HALTING_STACK_ACTIVATION
    halting_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION
    )
    halting_stack_residual_model_flag: bool = config.HALTING_STACK_RESIDUAL_MODEL_FLAG
    halting_stack_residual_block_size: int | None = (
        config.HALTING_STACK_RESIDUAL_BLOCK_SIZE
    )
    halting_stack_residual_rms_norm_epsilon: float | None = (
        config.HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    halting_stack_dropout_probability: float | None = (
        config.HALTING_STACK_DROPOUT_PROBABILITY
    )
    halting_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.HALTING_STACK_LAST_LAYER_BIAS_OPTION
    )
    halting_stack_apply_output_postprocessing_flag: bool | None = (
        config.HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    halting_stack_bias_flag: bool | None = config.HALTING_STACK_BIAS_FLAG
    memory_flag: bool = config.MEMORY_FLAG
    memory_option: type[DynamicMemoryConfig] = config.MEMORY_OPTION
    memory_position_option: MemoryPositionOptions = config.MEMORY_POSITION_OPTION
    memory_test_time_training_learning_rate: float | None = (
        config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
    )
    memory_test_time_training_num_inner_steps: int | None = (
        config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
    )
    memory_stack_independent_flag: bool = config.MEMORY_STACK_INDEPENDENT_FLAG
    memory_stack_hidden_dim: int | None = config.MEMORY_STACK_HIDDEN_DIM
    memory_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.MEMORY_STACK_LAYER_NORM_POSITION
    )
    memory_stack_normalization: NormalizationOptions | None = field(
        default=(config.MEMORY_STACK_NORMALIZATION), kw_only=True
    )
    memory_stack_num_layers: int | None = config.MEMORY_STACK_NUM_LAYERS
    memory_stack_activation: ActivationOptions | None = config.MEMORY_STACK_ACTIVATION
    memory_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
    )
    memory_stack_residual_model_flag: bool = config.MEMORY_STACK_RESIDUAL_MODEL_FLAG
    memory_stack_residual_block_size: int | None = (
        config.MEMORY_STACK_RESIDUAL_BLOCK_SIZE
    )
    memory_stack_residual_rms_norm_epsilon: float | None = (
        config.MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    memory_stack_dropout_probability: float | None = (
        config.MEMORY_STACK_DROPOUT_PROBABILITY
    )
    memory_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.MEMORY_STACK_LAST_LAYER_BIAS_OPTION
    )
    memory_stack_apply_output_postprocessing_flag: bool | None = (
        config.MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    memory_stack_bias_flag: bool | None = config.MEMORY_STACK_BIAS_FLAG
    recurrent_flag: bool = config.RECURRENT_FLAG
    recurrent_max_steps: int = config.RECURRENT_MAX_STEPS
    recurrent_initial_iterations: int | None = config.RECURRENT_INITIAL_ITERATIONS
    recurrent_gradient_transition_count: int | None = (
        config.RECURRENT_GRADIENT_TRANSITION_COUNT
    )
    recurrent_no_gradient_transition_count: int | None = (
        config.RECURRENT_NO_GRADIENT_TRANSITION_COUNT
    )
    recurrent_iteration_increment: int = config.RECURRENT_ITERATION_INCREMENT
    recurrent_forward_calls_before_iteration_increment: int = (
        config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT
    )
    recurrent_smooth_iteration_growth_flag: bool = (
        config.RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG
    )
    recurrent_layer_norm_position: LayerNormPositionOptions = (
        config.RECURRENT_LAYER_NORM_POSITION
    )
    recurrent_normalization: NormalizationOptions = field(
        default=(config.RECURRENT_NORMALIZATION), kw_only=True
    )
    recurrent_stack_gate_flag: bool = config.RECURRENT_STACK_GATE_FLAG
    recurrent_gate_option: LayerGateOptions | None = config.RECURRENT_GATE_OPTION
    recurrent_gate_activation: ActivationOptions | None = (
        config.RECURRENT_GATE_ACTIVATION
    )
    recurrent_gate_stack_independent_flag: bool = (
        config.RECURRENT_GATE_STACK_INDEPENDENT_FLAG
    )
    recurrent_gate_stack_hidden_dim: int | None = config.RECURRENT_GATE_STACK_HIDDEN_DIM
    recurrent_gate_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.RECURRENT_GATE_STACK_LAYER_NORM_POSITION
    )
    recurrent_gate_stack_normalization: NormalizationOptions | None = field(
        default=(config.RECURRENT_GATE_STACK_NORMALIZATION), kw_only=True
    )
    recurrent_gate_stack_num_layers: int | None = config.RECURRENT_GATE_STACK_NUM_LAYERS
    recurrent_gate_stack_activation: ActivationOptions | None = (
        config.RECURRENT_GATE_STACK_ACTIVATION
    )
    recurrent_gate_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
    )
    recurrent_gate_stack_residual_model_flag: bool = (
        config.RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG
    )
    recurrent_gate_stack_residual_block_size: int | None = (
        config.RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE
    )
    recurrent_gate_stack_residual_rms_norm_epsilon: float | None = (
        config.RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    recurrent_gate_stack_dropout_probability: float | None = (
        config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY
    )
    recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
    )
    recurrent_gate_stack_apply_output_postprocessing_flag: bool | None = (
        config.RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    recurrent_gate_stack_bias_flag: bool | None = config.RECURRENT_GATE_STACK_BIAS_FLAG
    recurrent_stack_halting_flag: bool = config.RECURRENT_STACK_HALTING_FLAG
    recurrent_halting_threshold: float = config.RECURRENT_HALTING_THRESHOLD
    recurrent_halting_dropout: float = config.RECURRENT_HALTING_DROPOUT
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        config.RECURRENT_HALTING_HIDDEN_STATE_MODE
    )
    recurrent_halting_stack_independent_flag: bool = (
        config.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG
    )
    recurrent_halting_stack_hidden_dim: int | None = (
        config.RECURRENT_HALTING_STACK_HIDDEN_DIM
    )
    recurrent_halting_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION
    )
    recurrent_halting_stack_normalization: NormalizationOptions | None = field(
        default=(config.RECURRENT_HALTING_STACK_NORMALIZATION), kw_only=True
    )
    recurrent_halting_stack_num_layers: int | None = (
        config.RECURRENT_HALTING_STACK_NUM_LAYERS
    )
    recurrent_halting_stack_activation: ActivationOptions | None = (
        config.RECURRENT_HALTING_STACK_ACTIVATION
    )
    recurrent_halting_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
    )
    recurrent_halting_stack_residual_model_flag: bool = (
        config.RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG
    )
    recurrent_halting_stack_residual_block_size: int | None = (
        config.RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE
    )
    recurrent_halting_stack_residual_rms_norm_epsilon: float | None = (
        config.RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    recurrent_halting_stack_dropout_probability: float | None = (
        config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY
    )
    recurrent_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
    )
    recurrent_halting_stack_apply_output_postprocessing_flag: bool | None = (
        config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    recurrent_halting_stack_bias_flag: bool | None = (
        config.RECURRENT_HALTING_STACK_BIAS_FLAG
    )
    shared_gate_config: GateConfig | None = None
    router_options: ExpertsRouterOptions | None = None
    router_stack_options: ExpertsSubmoduleStackOptions | None = None
    router_layer_controller_options: ExpertsLayerControllerOptions | None = None
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions | None = None
    layer_controller_options: ExpertsLayerControllerOptions | None = None
    dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None
    expert_layer_controller_options: ExpertsLayerControllerOptions | None = None
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions | None = None
    recurrent_controller_options: ExpertsRecurrentControllerOptions | None = None
    halting_option: type[HaltingConfig] = config.HALTING_OPTION
    expert_halting_option: type[HaltingConfig] = config.EXPERT_HALTING_OPTION
    router_halting_option: type[HaltingConfig] = config.ROUTER_HALTING_OPTION
    recurrent_halting_option: type[HaltingConfig] = config.RECURRENT_HALTING_OPTION
    expert_recurrent_halting_option: type[HaltingConfig] = (
        config.EXPERT_RECURRENT_HALTING_OPTION
    )
    router_recurrent_halting_option: type[HaltingConfig] = (
        config.ROUTER_RECURRENT_HALTING_OPTION
    )


@dataclass(frozen=True, slots=True)
class ControlDefaults:
    expert_layer_controller_options: ExpertsLayerControllerOptions
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions
    router_options: ExpertsRouterOptions
    router_stack_options: ExpertsSubmoduleStackOptions
    router_layer_controller_options: ExpertsLayerControllerOptions
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions
    layer_controller_options: ExpertsLayerControllerOptions
    dynamic_memory_options: ExpertsDynamicMemoryOptions
    recurrent_controller_options: ExpertsRecurrentControllerOptions
    residual_stack_options: ResidualStackOptions


class _Defaults(ExpertControlDefaults[ControlDefaults]):
    stack_type = ExpertsSubmoduleStackOptions
    source_type = ExpertsSubmoduleStackSource
    layer_type = ExpertsLayerControllerOptions
    memory_type = ExpertsDynamicMemoryOptions
    recurrent_type = ExpertsRecurrentControllerOptions
    router_type = ExpertsRouterOptions
    defaults_type = ControlDefaults
    resolve_residual = staticmethod(resolve_residual_stack_options)
    residual_source_type = ResidualStackSource


def resolve_control_defaults(
    values: ControlDefaultValues, submodule_stack_options: ExpertsSubmoduleStackOptions
) -> ControlDefaults:
    return _Defaults().resolve_control_defaults(values, submodule_stack_options)


__all__ = ["ControlDefaultValues", "ControlDefaults", "resolve_control_defaults"]
