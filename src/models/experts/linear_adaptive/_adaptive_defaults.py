from dataclasses import dataclass, field

import models.experts.linear_adaptive.config as config
from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from models._expert_adaptive_defaults import ExpertAdaptiveDefaults
from models.experts.linear_adaptive._generation import AdaptiveGeneratorStackSource
from models.experts.linear_adaptive.runtime_options import (
    AdaptiveBoundaryModelOptions,
    AdaptiveGeneratorStackOptions,
    ExpertsAdaptiveGeneratorStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
)


@dataclass(frozen=True, kw_only=True)
class AdaptiveDefaultValues:
    weight_option_flag: bool | None = None
    generator_depth: int = config.GENERATOR_DEPTH
    diagonal_option: type[DynamicDiagonalConfig] | None = config.DIAGONAL_OPTION
    diagonal_option_flag: bool | None = None
    bias_option: type[DynamicBiasConfig] | None = config.BIAS_OPTION
    bias_option_flag: bool | None = None
    weight_option: type[DynamicWeightConfig] | None = config.WEIGHT_OPTION
    weight_generator_stack_independent_flag: bool = (
        config.WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    weight_generator_stack_hidden_dim: int | None = (
        config.WEIGHT_GENERATOR_STACK_HIDDEN_DIM
    )
    weight_generator_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    weight_generator_stack_normalization: NormalizationOptions | None = field(
        default=(config.WEIGHT_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    weight_generator_stack_num_layers: int | None = (
        config.WEIGHT_GENERATOR_STACK_NUM_LAYERS
    )
    weight_generator_stack_activation: ActivationOptions | None = (
        config.WEIGHT_GENERATOR_STACK_ACTIVATION
    )
    weight_generator_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    )
    weight_generator_stack_residual_model_flag: bool = (
        config.WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    weight_generator_stack_residual_block_size: int | None = (
        config.WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    weight_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    weight_generator_stack_dropout_probability: float | None = (
        config.WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    weight_generator_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    weight_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    weight_generator_stack_bias_flag: bool | None = (
        config.WEIGHT_GENERATOR_STACK_BIAS_FLAG
    )
    weight_normalization_option: WeightNormalizationOptions = (
        config.WEIGHT_NORMALIZATION_OPTION
    )
    weight_normalization_learnable_scale_flag: bool = (
        config.WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG
    )
    weight_normalization_position_option: WeightNormalizationPositionOptions = (
        config.WEIGHT_NORMALIZATION_POSITION_OPTION
    )
    weight_decay_schedule: WeightDecayScheduleOptions = config.WEIGHT_DECAY_SCHEDULE
    weight_decay_rate: float = config.WEIGHT_DECAY_RATE
    weight_decay_warmup_batches: int = config.WEIGHT_DECAY_WARMUP_BATCHES
    weight_bank_expansion_factor: BankExpansionFactorOptions = (
        config.WEIGHT_BANK_EXPANSION_FACTOR
    )
    bias_decay_schedule: WeightDecayScheduleOptions = config.BIAS_DECAY_SCHEDULE
    bias_decay_rate: float = config.BIAS_DECAY_RATE
    bias_decay_warmup_batches: int = config.BIAS_DECAY_WARMUP_BATCHES
    bias_bank_expansion_factor: BankExpansionFactorOptions = (
        config.BIAS_BANK_EXPANSION_FACTOR
    )
    bias_generator_stack_independent_flag: bool = (
        config.BIAS_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    bias_generator_stack_hidden_dim: int | None = config.BIAS_GENERATOR_STACK_HIDDEN_DIM
    bias_generator_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.BIAS_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    bias_generator_stack_normalization: NormalizationOptions | None = field(
        default=(config.BIAS_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    bias_generator_stack_num_layers: int | None = config.BIAS_GENERATOR_STACK_NUM_LAYERS
    bias_generator_stack_activation: ActivationOptions | None = (
        config.BIAS_GENERATOR_STACK_ACTIVATION
    )
    bias_generator_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    )
    bias_generator_stack_residual_model_flag: bool = (
        config.BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    bias_generator_stack_residual_block_size: int | None = (
        config.BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    bias_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    bias_generator_stack_dropout_probability: float | None = (
        config.BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    bias_generator_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    bias_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    bias_generator_stack_bias_flag: bool | None = config.BIAS_GENERATOR_STACK_BIAS_FLAG
    diagonal_generator_stack_independent_flag: bool = (
        config.DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    diagonal_generator_stack_hidden_dim: int | None = (
        config.DIAGONAL_GENERATOR_STACK_HIDDEN_DIM
    )
    diagonal_generator_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    diagonal_generator_stack_normalization: NormalizationOptions | None = field(
        default=(config.DIAGONAL_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    diagonal_generator_stack_num_layers: int | None = (
        config.DIAGONAL_GENERATOR_STACK_NUM_LAYERS
    )
    diagonal_generator_stack_activation: ActivationOptions | None = (
        config.DIAGONAL_GENERATOR_STACK_ACTIVATION
    )
    diagonal_generator_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    )
    diagonal_generator_stack_residual_model_flag: bool = (
        config.DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    diagonal_generator_stack_residual_block_size: int | None = (
        config.DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    diagonal_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    diagonal_generator_stack_dropout_probability: float | None = (
        config.DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    diagonal_generator_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    diagonal_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    diagonal_generator_stack_bias_flag: bool | None = (
        config.DIAGONAL_GENERATOR_STACK_BIAS_FLAG
    )
    row_mask_option: type[AxisMaskConfig] | None = config.ROW_MASK_OPTION
    mask_option_flag: bool | None = None
    mask_dimension_option: MaskDimensionOptions = config.MASK_DIMENSION_OPTION
    mask_threshold: float = config.MASK_THRESHOLD
    mask_surrogate_scale: float = config.MASK_SURROGATE_SCALE
    mask_floor: float = config.MASK_FLOOR
    mask_transition_width: float = config.MASK_TRANSITION_WIDTH
    mask_generator_stack_independent_flag: bool = (
        config.MASK_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    mask_generator_stack_hidden_dim: int | None = config.MASK_GENERATOR_STACK_HIDDEN_DIM
    mask_generator_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.MASK_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    mask_generator_stack_normalization: NormalizationOptions | None = field(
        default=(config.MASK_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    mask_generator_stack_num_layers: int | None = config.MASK_GENERATOR_STACK_NUM_LAYERS
    mask_generator_stack_activation: ActivationOptions | None = (
        config.MASK_GENERATOR_STACK_ACTIVATION
    )
    mask_generator_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    )
    mask_generator_stack_residual_model_flag: bool = (
        config.MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    mask_generator_stack_residual_block_size: int | None = (
        config.MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    mask_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    mask_generator_stack_dropout_probability: float | None = (
        config.MASK_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    mask_generator_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    mask_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    mask_generator_stack_bias_flag: bool | None = config.MASK_GENERATOR_STACK_BIAS_FLAG
    adaptive_generator_stack_num_layers: int = (
        config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS
    )
    adaptive_generator_stack_hidden_dim: int = (
        config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM
    )
    adaptive_generator_stack_activation: ActivationOptions = (
        config.ADAPTIVE_GENERATOR_STACK_ACTIVATION
    )
    adaptive_generator_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    )
    adaptive_generator_stack_residual_model_flag: bool = (
        config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    adaptive_generator_stack_residual_block_size: int | None = (
        config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    adaptive_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    adaptive_generator_stack_dropout_probability: float = (
        config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions = (
        config.ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    adaptive_generator_stack_normalization: NormalizationOptions = field(
        default=(config.ADAPTIVE_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    adaptive_generator_stack_last_layer_bias_option: LastLayerBiasOptions = (
        config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    adaptive_generator_stack_apply_output_postprocessing_flag: bool = (
        config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    adaptive_generator_stack_bias_flag: bool = config.ADAPTIVE_GENERATOR_STACK_BIAS_FLAG
    input_layer_weight_option: type[DynamicWeightConfig] | None = (
        config.INPUT_LAYER_WEIGHT_OPTION
    )
    input_layer_generator_depth: int = config.INPUT_LAYER_GENERATOR_DEPTH
    input_layer_weight_decay_schedule: WeightDecayScheduleOptions = (
        config.INPUT_LAYER_WEIGHT_DECAY_SCHEDULE
    )
    input_layer_weight_decay_rate: float = config.INPUT_LAYER_WEIGHT_DECAY_RATE
    input_layer_weight_decay_warmup_batches: int = (
        config.INPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES
    )
    input_layer_weight_normalization_option: WeightNormalizationOptions = (
        config.INPUT_LAYER_WEIGHT_NORMALIZATION_OPTION
    )
    input_layer_weight_normalization_learnable_scale_flag: bool = (
        config.INPUT_LAYER_WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG
    )
    input_layer_weight_normalization_position_option: WeightNormalizationPositionOptions = config.INPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION
    input_layer_weight_bank_expansion_factor: BankExpansionFactorOptions = (
        config.INPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR
    )
    input_layer_bias_option: type[DynamicBiasConfig] | None = (
        config.INPUT_LAYER_BIAS_OPTION
    )
    input_layer_bias_decay_schedule: WeightDecayScheduleOptions = (
        config.INPUT_LAYER_BIAS_DECAY_SCHEDULE
    )
    input_layer_bias_decay_rate: float = config.INPUT_LAYER_BIAS_DECAY_RATE
    input_layer_bias_decay_warmup_batches: int = (
        config.INPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES
    )
    input_layer_bias_bank_expansion_factor: BankExpansionFactorOptions = (
        config.INPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR
    )
    input_layer_diagonal_option: type[DynamicDiagonalConfig] | None = (
        config.INPUT_LAYER_DIAGONAL_OPTION
    )
    input_layer_row_mask_option: type[AxisMaskConfig] | None = (
        config.INPUT_LAYER_ROW_MASK_OPTION
    )
    input_layer_mask_dimension_option: MaskDimensionOptions = (
        config.INPUT_LAYER_MASK_DIMENSION_OPTION
    )
    input_layer_mask_threshold: float = config.INPUT_LAYER_MASK_THRESHOLD
    input_layer_mask_surrogate_scale: float = config.INPUT_LAYER_MASK_SURROGATE_SCALE
    input_layer_mask_floor: float = config.INPUT_LAYER_MASK_FLOOR
    input_layer_mask_transition_width: float = config.INPUT_LAYER_MASK_TRANSITION_WIDTH
    output_layer_weight_option: type[DynamicWeightConfig] | None = (
        config.OUTPUT_LAYER_WEIGHT_OPTION
    )
    output_layer_generator_depth: int = config.OUTPUT_LAYER_GENERATOR_DEPTH
    output_layer_weight_decay_schedule: WeightDecayScheduleOptions = (
        config.OUTPUT_LAYER_WEIGHT_DECAY_SCHEDULE
    )
    output_layer_weight_decay_rate: float = config.OUTPUT_LAYER_WEIGHT_DECAY_RATE
    output_layer_weight_decay_warmup_batches: int = (
        config.OUTPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES
    )
    output_layer_weight_normalization_option: WeightNormalizationOptions = (
        config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_OPTION
    )
    output_layer_weight_normalization_learnable_scale_flag: bool = (
        config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG
    )
    output_layer_weight_normalization_position_option: WeightNormalizationPositionOptions = config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION
    output_layer_weight_bank_expansion_factor: BankExpansionFactorOptions = (
        config.OUTPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR
    )
    output_layer_bias_option: type[DynamicBiasConfig] | None = (
        config.OUTPUT_LAYER_BIAS_OPTION
    )
    output_layer_bias_decay_schedule: WeightDecayScheduleOptions = (
        config.OUTPUT_LAYER_BIAS_DECAY_SCHEDULE
    )
    output_layer_bias_decay_rate: float = config.OUTPUT_LAYER_BIAS_DECAY_RATE
    output_layer_bias_decay_warmup_batches: int = (
        config.OUTPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES
    )
    output_layer_bias_bank_expansion_factor: BankExpansionFactorOptions = (
        config.OUTPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR
    )
    output_layer_diagonal_option: type[DynamicDiagonalConfig] | None = (
        config.OUTPUT_LAYER_DIAGONAL_OPTION
    )
    output_layer_row_mask_option: type[AxisMaskConfig] | None = (
        config.OUTPUT_LAYER_ROW_MASK_OPTION
    )
    output_layer_mask_dimension_option: MaskDimensionOptions = (
        config.OUTPUT_LAYER_MASK_DIMENSION_OPTION
    )
    output_layer_mask_threshold: float = config.OUTPUT_LAYER_MASK_THRESHOLD
    output_layer_mask_surrogate_scale: float = config.OUTPUT_LAYER_MASK_SURROGATE_SCALE
    output_layer_mask_floor: float = config.OUTPUT_LAYER_MASK_FLOOR
    output_layer_mask_transition_width: float = (
        config.OUTPUT_LAYER_MASK_TRANSITION_WIDTH
    )
    router_weight_option_flag: bool | None = None
    router_weight_option: type[DynamicWeightConfig] | None = config.ROUTER_WEIGHT_OPTION
    router_generator_depth: int = config.ROUTER_GENERATOR_DEPTH
    router_weight_normalization_option: WeightNormalizationOptions = (
        config.ROUTER_WEIGHT_NORMALIZATION_OPTION
    )
    router_weight_normalization_learnable_scale_flag: bool = (
        config.ROUTER_WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG
    )
    router_weight_normalization_position_option: WeightNormalizationPositionOptions = (
        config.ROUTER_WEIGHT_NORMALIZATION_POSITION_OPTION
    )
    router_weight_decay_schedule: WeightDecayScheduleOptions = (
        config.ROUTER_WEIGHT_DECAY_SCHEDULE
    )
    router_weight_decay_rate: float = config.ROUTER_WEIGHT_DECAY_RATE
    router_weight_decay_warmup_batches: int = config.ROUTER_WEIGHT_DECAY_WARMUP_BATCHES
    router_weight_bank_expansion_factor: BankExpansionFactorOptions = (
        config.ROUTER_WEIGHT_BANK_EXPANSION_FACTOR
    )
    router_weight_generator_stack_independent_flag: bool = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    router_weight_generator_stack_hidden_dim: int | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_HIDDEN_DIM
    )
    router_weight_generator_stack_layer_norm_position: (
        LayerNormPositionOptions | None
    ) = config.ROUTER_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION
    router_weight_generator_stack_normalization: NormalizationOptions | None = field(
        default=config.ROUTER_WEIGHT_GENERATOR_STACK_NORMALIZATION, kw_only=True
    )
    router_weight_generator_stack_num_layers: int | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_NUM_LAYERS
    )
    router_weight_generator_stack_activation: ActivationOptions | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_ACTIVATION
    )
    router_weight_generator_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    router_weight_generator_stack_residual_model_flag: bool = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    router_weight_generator_stack_residual_block_size: int | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_weight_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_weight_generator_stack_dropout_probability: float | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    router_weight_generator_stack_last_layer_bias_option: (
        LastLayerBiasOptions | None
    ) = config.ROUTER_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    router_weight_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_weight_generator_stack_bias_flag: bool | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_BIAS_FLAG
    )
    router_bias_option_flag: bool | None = None
    router_bias_option: type[DynamicBiasConfig] | None = config.ROUTER_BIAS_OPTION
    router_bias_decay_schedule: WeightDecayScheduleOptions = (
        config.ROUTER_BIAS_DECAY_SCHEDULE
    )
    router_bias_decay_rate: float = config.ROUTER_BIAS_DECAY_RATE
    router_bias_decay_warmup_batches: int = config.ROUTER_BIAS_DECAY_WARMUP_BATCHES
    router_bias_bank_expansion_factor: BankExpansionFactorOptions = (
        config.ROUTER_BIAS_BANK_EXPANSION_FACTOR
    )
    router_bias_generator_stack_independent_flag: bool = (
        config.ROUTER_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    router_bias_generator_stack_hidden_dim: int | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_HIDDEN_DIM
    )
    router_bias_generator_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    router_bias_generator_stack_normalization: NormalizationOptions | None = field(
        default=(config.ROUTER_BIAS_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    router_bias_generator_stack_num_layers: int | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_NUM_LAYERS
    )
    router_bias_generator_stack_activation: ActivationOptions | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_ACTIVATION
    )
    router_bias_generator_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    router_bias_generator_stack_residual_model_flag: bool = (
        config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    router_bias_generator_stack_residual_block_size: int | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_bias_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_bias_generator_stack_dropout_probability: float | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    router_bias_generator_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_bias_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_bias_generator_stack_bias_flag: bool | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_BIAS_FLAG
    )
    router_diagonal_option_flag: bool | None = None
    router_diagonal_option: type[DynamicDiagonalConfig] | None = (
        config.ROUTER_DIAGONAL_OPTION
    )
    router_diagonal_generator_stack_independent_flag: bool = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    router_diagonal_generator_stack_hidden_dim: int | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM
    )
    router_diagonal_generator_stack_layer_norm_position: (
        LayerNormPositionOptions | None
    ) = config.ROUTER_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION
    router_diagonal_generator_stack_normalization: NormalizationOptions | None = field(
        default=config.ROUTER_DIAGONAL_GENERATOR_STACK_NORMALIZATION, kw_only=True
    )
    router_diagonal_generator_stack_num_layers: int | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_NUM_LAYERS
    )
    router_diagonal_generator_stack_activation: ActivationOptions | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_ACTIVATION
    )
    router_diagonal_generator_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    router_diagonal_generator_stack_residual_model_flag: bool = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    router_diagonal_generator_stack_residual_block_size: int | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_diagonal_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_diagonal_generator_stack_dropout_probability: float | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    router_diagonal_generator_stack_last_layer_bias_option: (
        LastLayerBiasOptions | None
    ) = config.ROUTER_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    router_diagonal_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_diagonal_generator_stack_bias_flag: bool | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_BIAS_FLAG
    )
    router_mask_option_flag: bool | None = None
    router_row_mask_option: type[AxisMaskConfig] | None = config.ROUTER_ROW_MASK_OPTION
    router_mask_dimension_option: MaskDimensionOptions = (
        config.ROUTER_MASK_DIMENSION_OPTION
    )
    router_mask_threshold: float = config.ROUTER_MASK_THRESHOLD
    router_mask_surrogate_scale: float = config.ROUTER_MASK_SURROGATE_SCALE
    router_mask_floor: float = config.ROUTER_MASK_FLOOR
    router_mask_transition_width: float = config.ROUTER_MASK_TRANSITION_WIDTH
    router_mask_generator_stack_independent_flag: bool = (
        config.ROUTER_MASK_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    router_mask_generator_stack_hidden_dim: int | None = (
        config.ROUTER_MASK_GENERATOR_STACK_HIDDEN_DIM
    )
    router_mask_generator_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.ROUTER_MASK_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    router_mask_generator_stack_normalization: NormalizationOptions | None = field(
        default=(config.ROUTER_MASK_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    router_mask_generator_stack_num_layers: int | None = (
        config.ROUTER_MASK_GENERATOR_STACK_NUM_LAYERS
    )
    router_mask_generator_stack_activation: ActivationOptions | None = (
        config.ROUTER_MASK_GENERATOR_STACK_ACTIVATION
    )
    router_mask_generator_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    router_mask_generator_stack_residual_model_flag: bool = (
        config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    router_mask_generator_stack_residual_block_size: int | None = (
        config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_mask_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_mask_generator_stack_dropout_probability: float | None = (
        config.ROUTER_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    router_mask_generator_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.ROUTER_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_mask_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_mask_generator_stack_bias_flag: bool | None = (
        config.ROUTER_MASK_GENERATOR_STACK_BIAS_FLAG
    )
    adaptive_generator_stack_options: (
        AdaptiveGeneratorStackOptions | ExpertsAdaptiveGeneratorStackOptions | None
    ) = None
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None
    input_boundary_options: AdaptiveBoundaryModelOptions | None = None
    output_boundary_options: AdaptiveBoundaryModelOptions | None = None
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None


@dataclass(frozen=True, slots=True)
class AdaptiveDefaults:
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions
    input_boundary_options: AdaptiveBoundaryModelOptions
    output_boundary_options: AdaptiveBoundaryModelOptions
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions


class _Defaults(ExpertAdaptiveDefaults[AdaptiveDefaults]):
    generator_type = AdaptiveGeneratorStackOptions
    source_type = AdaptiveGeneratorStackSource
    weight_type = HiddenAdaptiveWeightOptions
    bias_type = HiddenAdaptiveBiasOptions
    diagonal_type = HiddenAdaptiveDiagonalOptions
    mask_type = HiddenAdaptiveMaskOptions
    boundary_type = AdaptiveBoundaryModelOptions
    defaults_type = AdaptiveDefaults
    config = config
    inherit_foreign_generator_residual = True


def resolve_adaptive_defaults(values: AdaptiveDefaultValues) -> AdaptiveDefaults:
    return _Defaults().resolve_adaptive_defaults(values)


__all__ = ["AdaptiveDefaultValues", "AdaptiveDefaults", "resolve_adaptive_defaults"]
