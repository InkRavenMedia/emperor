from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Generic, TypeVar, cast

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GroupingConfig,
    LowRankFactorSourceOptions,
    MaskDimensionOptions,
    SummaryNormalizationOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.config import ConfigBase
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
from models import _learned_grouping
from models._adaptive_runtime_fields import (
    BiasFields,
    BiasValues,
    ControlFields,
    ControlValues,
    DiagonalFields,
    DiagonalValues,
    GenerationValues,
    MaskFields,
    MaskValues,
    OptionalStackFields,
    OptionalStackValues,
    ProjectionFields,
    ProjectionValues,
    RuntimeDefaultValues,
    RuntimeDimensionsFields,
    RuntimeDimensionsValues,
    RuntimeFieldFactory,
    RuntimeOverrideReader,
    StackFields,
    StackValues,
    WeightFields,
    WeightValues,
    raise_type_error,
)

_Values = TypeVar("_Values", bound=RuntimeDefaultValues)


class AdaptiveRuntimeDefaults(RuntimeFieldFactory, Generic[_Values]):
    config: ModuleType
    package: str
    runtime_override_reader_type: type[RuntimeOverrideReader] = RuntimeOverrideReader
    runtime_dimensions_values_type: type[RuntimeDimensionsValues] = (
        RuntimeDimensionsValues
    )
    runtime_dimensions_fields_type: type[RuntimeDimensionsFields] = (
        RuntimeDimensionsFields
    )
    stack_values_type: type[StackValues] = StackValues
    stack_fields_type: type[StackFields] = StackFields
    optional_stack_values_type: type[OptionalStackValues] = OptionalStackValues
    optional_stack_fields_type: type[OptionalStackFields] = OptionalStackFields
    control_values_type: type[ControlValues] = ControlValues
    control_fields_type: type[ControlFields] = ControlFields
    weight_values_type: type[WeightValues] = WeightValues
    weight_fields_type: type[WeightFields] = WeightFields
    bias_values_type: type[BiasValues] = BiasValues
    bias_fields_type: type[BiasFields] = BiasFields
    diagonal_values_type: type[DiagonalValues] = DiagonalValues
    diagonal_fields_type: type[DiagonalFields] = DiagonalFields
    mask_values_type: type[MaskValues] = MaskValues
    mask_fields_type: type[MaskFields] = MaskFields
    projection_values_type: type[ProjectionValues] = ProjectionValues
    projection_fields_type: type[ProjectionFields] = ProjectionFields
    generation_values_type: type[GenerationValues] = GenerationValues
    runtime_values_type: Callable[..., _Values]

    def __init__(self) -> None:
        self.dimension_fields = self._dimension_fields()
        self.main_stack_fields = self._main_stack_fields()
        self.submodule_stack_fields = self._submodule_stack_fields()
        self.adaptive_generator_stack_fields = self._adaptive_generator_stack_fields()
        self.residual_stack_fields = self.optional_stack_fields_type.for_stack(
            "residual_stack"
        )
        self.gate_stack_fields = self.optional_stack_fields_type.for_stack("gate_stack")
        self.halting_stack_fields = self.optional_stack_fields_type.for_stack(
            "halting_stack"
        )
        self.memory_stack_fields = self.optional_stack_fields_type.for_stack(
            "memory_stack"
        )
        self.recurrent_gate_stack_fields = self.optional_stack_fields_type.for_stack(
            "recurrent_gate_stack"
        )
        self.recurrent_halting_stack_fields = self.optional_stack_fields_type.for_stack(
            "recurrent_halting_stack"
        )
        self.weight_generator_stack_fields = self.optional_stack_fields_type.for_stack(
            "weight_generator_stack"
        )
        self.bias_generator_stack_fields = self.optional_stack_fields_type.for_stack(
            "bias_generator_stack"
        )
        self.diagonal_generator_stack_fields = (
            self.optional_stack_fields_type.for_stack("diagonal_generator_stack")
        )
        self.mask_generator_stack_fields = self.optional_stack_fields_type.for_stack(
            "mask_generator_stack"
        )
        self.control_fields = self._control_fields()
        self.weight_fields = self._weight_fields()
        self.bias_fields = self._bias_fields()
        self.diagonal_fields = self._diagonal_fields()
        self.mask_fields = self._mask_fields()
        self.input_projection_fields = self._input_projection_fields()
        self.output_projection_fields = self._output_projection_fields()
        reader = self.runtime_override_reader_type({})
        self._read_runtime_values(reader)
        self.fields = reader.fields

    def legacy_field_group(self, name: str, module: str) -> object:
        if name == "_FIELDS":
            return self.fields
        if name.startswith("_") and name.endswith("_FIELDS") and name.isupper():
            attribute = name[1:].lower()
            if attribute in vars(self):
                return getattr(self, attribute)
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    def _dimension_fields(self) -> RuntimeDimensionsFields:
        return self.runtime_dimensions_fields_type(
            batch_size=self._integer_field("batch_size", self.config.BATCH_SIZE),
            learning_rate=self._float_field("learning_rate", self.config.LEARNING_RATE),
            input_dim=self._integer_field("input_dim", self.config.INPUT_DIM),
            hidden_dim=self._integer_field("hidden_dim", self.config.HIDDEN_DIM),
            output_dim=self._integer_field("output_dim", self.config.OUTPUT_DIM),
        )

    def _main_stack_fields(self) -> StackFields:
        return self.stack_fields_type(
            hidden_dim=self._integer_field("hidden_dim", self.config.HIDDEN_DIM),
            num_layers=self._integer_field(
                "stack_num_layers", self.config.STACK_NUM_LAYERS
            ),
            last_layer_bias_option=self._enum_field(
                "stack_last_layer_bias_option",
                self.config.STACK_LAST_LAYER_BIAS_OPTION,
                LastLayerBiasOptions,
            ),
            apply_output_postprocessing_flag=self._boolean_field(
                "stack_apply_output_postprocessing_flag",
                self.config.STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            ),
            activation=self._enum_field(
                "stack_activation", self.config.STACK_ACTIVATION, ActivationOptions
            ),
            layer_norm_position=self._enum_field(
                "layer_norm_position",
                self.config.LAYER_NORM_POSITION,
                LayerNormPositionOptions,
            ),
            normalization=self._enum_field(
                "normalization",
                self.config.NORMALIZATION,
                NormalizationOptions,
            ),
            residual_connection_option=self._optional_implementation_field(
                "stack_residual_connection_option",
                self.config.STACK_RESIDUAL_CONNECTION_OPTION,
                ResidualConfig,
            ),
            residual_block_size=self._optional_integer_field(
                "stack_residual_block_size", self.config.STACK_RESIDUAL_BLOCK_SIZE
            ),
            residual_rms_norm_epsilon=self._optional_float_field(
                "stack_residual_rms_norm_epsilon",
                self.config.STACK_RESIDUAL_RMS_NORM_EPSILON,
            ),
            residual_model_flag=self._boolean_field(
                "stack_residual_model_flag", self.config.STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=self._float_field(
                "stack_dropout_probability", self.config.STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=self._boolean_field(
                "stack_bias_flag", self.config.STACK_BIAS_FLAG
            ),
        )

    def _submodule_stack_fields(self) -> StackFields:
        return self.stack_fields_type(
            hidden_dim=self._integer_field(
                "submodule_stack_hidden_dim", self.config.SUBMODULE_STACK_HIDDEN_DIM
            ),
            num_layers=self._integer_field(
                "submodule_stack_num_layers", self.config.SUBMODULE_STACK_NUM_LAYERS
            ),
            last_layer_bias_option=self._enum_field(
                "submodule_stack_last_layer_bias_option",
                self.config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
                LastLayerBiasOptions,
            ),
            apply_output_postprocessing_flag=self._boolean_field(
                "submodule_stack_apply_output_postprocessing_flag",
                self.config.SUBMODULE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            ),
            activation=self._enum_field(
                "submodule_stack_activation",
                self.config.SUBMODULE_STACK_ACTIVATION,
                ActivationOptions,
            ),
            layer_norm_position=self._enum_field(
                "submodule_stack_layer_norm_position",
                self.config.SUBMODULE_STACK_LAYER_NORM_POSITION,
                LayerNormPositionOptions,
            ),
            normalization=self._enum_field(
                "submodule_stack_normalization",
                self.config.SUBMODULE_STACK_NORMALIZATION,
                NormalizationOptions,
            ),
            residual_connection_option=self._optional_implementation_field(
                "submodule_stack_residual_connection_option",
                self.config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION,
                ResidualConfig,
            ),
            residual_block_size=self._optional_integer_field(
                "submodule_stack_residual_block_size",
                self.config.SUBMODULE_STACK_RESIDUAL_BLOCK_SIZE,
            ),
            residual_rms_norm_epsilon=self._optional_float_field(
                "submodule_stack_residual_rms_norm_epsilon",
                self.config.SUBMODULE_STACK_RESIDUAL_RMS_NORM_EPSILON,
            ),
            residual_model_flag=self._boolean_field(
                "submodule_stack_residual_model_flag",
                self.config.SUBMODULE_STACK_RESIDUAL_MODEL_FLAG,
            ),
            dropout_probability=self._float_field(
                "submodule_stack_dropout_probability",
                self.config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
            ),
            bias_flag=self._boolean_field(
                "submodule_stack_bias_flag", self.config.SUBMODULE_STACK_BIAS_FLAG
            ),
        )

    def _adaptive_generator_stack_fields(self) -> StackFields:
        return self.stack_fields_type(
            hidden_dim=self._integer_field(
                "adaptive_generator_stack_hidden_dim",
                self.config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
            ),
            num_layers=self._integer_field(
                "adaptive_generator_stack_num_layers",
                self.config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
            ),
            last_layer_bias_option=self._enum_field(
                "adaptive_generator_stack_last_layer_bias_option",
                self.config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
                LastLayerBiasOptions,
            ),
            apply_output_postprocessing_flag=self._boolean_field(
                "adaptive_generator_stack_apply_output_postprocessing_flag",
                self.config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            ),
            activation=self._enum_field(
                "adaptive_generator_stack_activation",
                self.config.ADAPTIVE_GENERATOR_STACK_ACTIVATION,
                ActivationOptions,
            ),
            layer_norm_position=self._enum_field(
                "adaptive_generator_stack_layer_norm_position",
                self.config.ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION,
                LayerNormPositionOptions,
            ),
            normalization=self._enum_field(
                "adaptive_generator_stack_normalization",
                self.config.ADAPTIVE_GENERATOR_STACK_NORMALIZATION,
                NormalizationOptions,
            ),
            residual_connection_option=self._optional_implementation_field(
                "adaptive_generator_stack_residual_connection_option",
                self.config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
                ResidualConfig,
            ),
            residual_block_size=self._optional_integer_field(
                "adaptive_generator_stack_residual_block_size",
                self.config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
            ),
            residual_rms_norm_epsilon=self._optional_float_field(
                "adaptive_generator_stack_residual_rms_norm_epsilon",
                self.config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
            ),
            residual_model_flag=self._boolean_field(
                "adaptive_generator_stack_residual_model_flag",
                self.config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            ),
            dropout_probability=self._float_field(
                "adaptive_generator_stack_dropout_probability",
                self.config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
            ),
            bias_flag=self._boolean_field(
                "adaptive_generator_stack_bias_flag",
                self.config.ADAPTIVE_GENERATOR_STACK_BIAS_FLAG,
            ),
        )

    def _control_fields(self) -> ControlFields:
        return self.control_fields_type(
            stack_gate_flag=self._boolean_field(
                "stack_gate_flag", self.config.STACK_GATE_FLAG
            ),
            gate_option=self._optional_enum_field(
                "gate_option", self.config.GATE_OPTION, LayerGateOptions
            ),
            gate_activation=self._optional_enum_field(
                "gate_activation", self.config.GATE_ACTIVATION, ActivationOptions
            ),
            shared_gate_config=self._optional_instance_field(
                "shared_gate_config", None, GateConfig
            ),
            stack_halting_flag=self._boolean_field(
                "stack_halting_flag", self.config.STACK_HALTING_FLAG
            ),
            halting_option=self._implementation_field(
                "halting_option", self.config.HALTING_OPTION, HaltingConfig
            ),
            halting_threshold=self._float_field(
                "halting_threshold", self.config.HALTING_THRESHOLD
            ),
            halting_dropout=self._float_field(
                "halting_dropout", self.config.HALTING_DROPOUT
            ),
            halting_hidden_state_mode=self._enum_field(
                "halting_hidden_state_mode",
                self.config.HALTING_HIDDEN_STATE_MODE,
                HaltingHiddenStateModeOptions,
            ),
            memory_flag=self._boolean_field("memory_flag", self.config.MEMORY_FLAG),
            memory_option=self._implementation_field(
                "memory_option", self.config.MEMORY_OPTION, DynamicMemoryConfig
            ),
            memory_position_option=self._enum_field(
                "memory_position_option",
                self.config.MEMORY_POSITION_OPTION,
                MemoryPositionOptions,
            ),
            memory_learning_rate=self._optional_float_field(
                "memory_test_time_training_learning_rate",
                self.config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
            ),
            memory_num_inner_steps=self._optional_integer_field(
                "memory_test_time_training_num_inner_steps",
                self.config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
            ),
            recurrent_flag=self._boolean_field(
                "recurrent_flag", self.config.RECURRENT_FLAG
            ),
            recurrent_max_steps=self._integer_field(
                "recurrent_max_steps", self.config.RECURRENT_MAX_STEPS
            ),
            recurrent_initial_iterations=self._integer_field(
                "recurrent_initial_iterations", self.config.RECURRENT_INITIAL_ITERATIONS
            ),
            recurrent_gradient_transition_count=self._optional_integer_field(
                "recurrent_gradient_transition_count",
                self.config.RECURRENT_GRADIENT_TRANSITION_COUNT,
            ),
            recurrent_no_gradient_transition_count=self._optional_integer_field(
                "recurrent_no_gradient_transition_count",
                self.config.RECURRENT_NO_GRADIENT_TRANSITION_COUNT,
            ),
            recurrent_iteration_increment=self._integer_field(
                "recurrent_iteration_increment",
                self.config.RECURRENT_ITERATION_INCREMENT,
            ),
            recurrent_forward_calls_before_iteration_increment=self._integer_field(
                "recurrent_forward_calls_before_iteration_increment",
                self.config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT,
            ),
            recurrent_smooth_iteration_growth_flag=self._boolean_field(
                "recurrent_smooth_iteration_growth_flag",
                self.config.RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG,
            ),
            recurrent_layer_norm_position=self._enum_field(
                "recurrent_layer_norm_position",
                self.config.RECURRENT_LAYER_NORM_POSITION,
                LayerNormPositionOptions,
            ),
            recurrent_normalization=self._enum_field(
                "recurrent_normalization",
                self.config.RECURRENT_NORMALIZATION,
                NormalizationOptions,
            ),
            recurrent_stack_gate_flag=self._boolean_field(
                "recurrent_stack_gate_flag", self.config.RECURRENT_STACK_GATE_FLAG
            ),
            recurrent_gate_option=self._optional_enum_field(
                "recurrent_gate_option",
                self.config.RECURRENT_GATE_OPTION,
                LayerGateOptions,
            ),
            recurrent_gate_activation=self._optional_enum_field(
                "recurrent_gate_activation",
                self.config.RECURRENT_GATE_ACTIVATION,
                ActivationOptions,
            ),
            recurrent_stack_halting_flag=self._boolean_field(
                "recurrent_stack_halting_flag", self.config.RECURRENT_STACK_HALTING_FLAG
            ),
            recurrent_halting_option=self._implementation_field(
                "recurrent_halting_option",
                self.config.RECURRENT_HALTING_OPTION,
                HaltingConfig,
            ),
            recurrent_halting_threshold=self._float_field(
                "recurrent_halting_threshold", self.config.RECURRENT_HALTING_THRESHOLD
            ),
            recurrent_halting_dropout=self._float_field(
                "recurrent_halting_dropout", self.config.RECURRENT_HALTING_DROPOUT
            ),
            recurrent_halting_hidden_state_mode=self._enum_field(
                "recurrent_halting_hidden_state_mode",
                self.config.RECURRENT_HALTING_HIDDEN_STATE_MODE,
                HaltingHiddenStateModeOptions,
            ),
        )

    def _weight_fields(self) -> WeightFields:
        return self.weight_fields_type(
            enabled=self._boolean_field(
                "weight_option_flag", self.config.WEIGHT_OPTION_FLAG
            ),
            option=self._optional_implementation_field(
                "weight_option", self.config.WEIGHT_OPTION, DynamicWeightConfig
            ),
            generator_depth=self._integer_field(
                "generator_depth", self.config.GENERATOR_DEPTH
            ),
            decay_schedule=self._enum_field(
                "weight_decay_schedule",
                self.config.WEIGHT_DECAY_SCHEDULE,
                WeightDecayScheduleOptions,
            ),
            decay_rate=self._float_field(
                "weight_decay_rate", self.config.WEIGHT_DECAY_RATE
            ),
            decay_warmup_batches=self._integer_field(
                "weight_decay_warmup_batches", self.config.WEIGHT_DECAY_WARMUP_BATCHES
            ),
            normalization_option=self._enum_field(
                "weight_normalization_option",
                self.config.WEIGHT_NORMALIZATION_OPTION,
                WeightNormalizationOptions,
            ),
            normalization_learnable_scale_flag=self._boolean_field(
                "weight_normalization_learnable_scale_flag",
                self.config.WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG,
            ),
            normalization_position_option=self._enum_field(
                "weight_normalization_position_option",
                self.config.WEIGHT_NORMALIZATION_POSITION_OPTION,
                WeightNormalizationPositionOptions,
            ),
            bank_expansion_factor=self._enum_field(
                "weight_bank_expansion_factor",
                self.config.WEIGHT_BANK_EXPANSION_FACTOR,
                BankExpansionFactorOptions,
            ),
        )

    def _bias_fields(self) -> BiasFields:
        return self.bias_fields_type(
            enabled=self._boolean_field(
                "bias_option_flag", self.config.BIAS_OPTION_FLAG
            ),
            option=self._optional_implementation_field(
                "bias_option", self.config.BIAS_OPTION, DynamicBiasConfig
            ),
            decay_schedule=self._enum_field(
                "bias_decay_schedule",
                self.config.BIAS_DECAY_SCHEDULE,
                WeightDecayScheduleOptions,
            ),
            decay_rate=self._float_field(
                "bias_decay_rate", self.config.BIAS_DECAY_RATE
            ),
            decay_warmup_batches=self._integer_field(
                "bias_decay_warmup_batches", self.config.BIAS_DECAY_WARMUP_BATCHES
            ),
            bank_expansion_factor=self._enum_field(
                "bias_bank_expansion_factor",
                self.config.BIAS_BANK_EXPANSION_FACTOR,
                BankExpansionFactorOptions,
            ),
        )

    def _diagonal_fields(self) -> DiagonalFields:
        return self.diagonal_fields_type(
            enabled=self._boolean_field(
                "diagonal_option_flag", self.config.DIAGONAL_OPTION_FLAG
            ),
            option=self._optional_implementation_field(
                "diagonal_option", self.config.DIAGONAL_OPTION, DynamicDiagonalConfig
            ),
        )

    def _mask_fields(self) -> MaskFields:
        return self.mask_fields_type(
            enabled=self._boolean_field(
                "mask_option_flag", self.config.MASK_OPTION_FLAG
            ),
            row_mask_option=self._optional_implementation_field(
                "row_mask_option", self.config.ROW_MASK_OPTION, AxisMaskConfig
            ),
            threshold=self._float_field("mask_threshold", self.config.MASK_THRESHOLD),
            floor=self._float_field("mask_floor", self.config.MASK_FLOOR),
            transition_width=self._float_field(
                "mask_transition_width", self.config.MASK_TRANSITION_WIDTH
            ),
            surrogate_scale=self._float_field(
                "mask_surrogate_scale", self.config.MASK_SURROGATE_SCALE
            ),
            dimension_option=self._enum_field(
                "mask_dimension_option",
                self.config.MASK_DIMENSION_OPTION,
                MaskDimensionOptions,
            ),
        )

    def _input_projection_fields(self) -> ProjectionFields:
        return self.projection_fields_type(
            weight_option=self._optional_implementation_field(
                "input_layer_weight_option",
                self.config.INPUT_LAYER_WEIGHT_OPTION,
                DynamicWeightConfig,
            ),
            generator_depth=self._integer_field(
                "input_layer_generator_depth", self.config.INPUT_LAYER_GENERATOR_DEPTH
            ),
            weight_decay_schedule=self._enum_field(
                "input_layer_weight_decay_schedule",
                self.config.INPUT_LAYER_WEIGHT_DECAY_SCHEDULE,
                WeightDecayScheduleOptions,
            ),
            weight_decay_rate=self._float_field(
                "input_layer_weight_decay_rate",
                self.config.INPUT_LAYER_WEIGHT_DECAY_RATE,
            ),
            weight_decay_warmup_batches=self._integer_field(
                "input_layer_weight_decay_warmup_batches",
                self.config.INPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES,
            ),
            weight_normalization_option=self._enum_field(
                "input_layer_weight_normalization_option",
                self.config.INPUT_LAYER_WEIGHT_NORMALIZATION_OPTION,
                WeightNormalizationOptions,
            ),
            weight_normalization_learnable_scale_flag=self._boolean_field(
                "input_layer_weight_normalization_learnable_scale_flag",
                self.config.INPUT_LAYER_WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG,
            ),
            weight_normalization_position_option=self._enum_field(
                "input_layer_weight_normalization_position_option",
                self.config.INPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION,
                WeightNormalizationPositionOptions,
            ),
            weight_bank_expansion_factor=self._enum_field(
                "input_layer_weight_bank_expansion_factor",
                self.config.INPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR,
                BankExpansionFactorOptions,
            ),
            bias_option=self._optional_implementation_field(
                "input_layer_bias_option",
                self.config.INPUT_LAYER_BIAS_OPTION,
                DynamicBiasConfig,
            ),
            bias_decay_schedule=self._enum_field(
                "input_layer_bias_decay_schedule",
                self.config.INPUT_LAYER_BIAS_DECAY_SCHEDULE,
                WeightDecayScheduleOptions,
            ),
            bias_decay_rate=self._float_field(
                "input_layer_bias_decay_rate", self.config.INPUT_LAYER_BIAS_DECAY_RATE
            ),
            bias_decay_warmup_batches=self._integer_field(
                "input_layer_bias_decay_warmup_batches",
                self.config.INPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES,
            ),
            bias_bank_expansion_factor=self._enum_field(
                "input_layer_bias_bank_expansion_factor",
                self.config.INPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR,
                BankExpansionFactorOptions,
            ),
            diagonal_option=self._optional_implementation_field(
                "input_layer_diagonal_option",
                self.config.INPUT_LAYER_DIAGONAL_OPTION,
                DynamicDiagonalConfig,
            ),
            row_mask_option=self._optional_implementation_field(
                "input_layer_row_mask_option",
                self.config.INPUT_LAYER_ROW_MASK_OPTION,
                AxisMaskConfig,
            ),
            mask_threshold=self._float_field(
                "input_layer_mask_threshold", self.config.INPUT_LAYER_MASK_THRESHOLD
            ),
            mask_floor=self._float_field(
                "input_layer_mask_floor", self.config.INPUT_LAYER_MASK_FLOOR
            ),
            mask_transition_width=self._float_field(
                "input_layer_mask_transition_width",
                self.config.INPUT_LAYER_MASK_TRANSITION_WIDTH,
            ),
            mask_surrogate_scale=self._float_field(
                "input_layer_mask_surrogate_scale",
                self.config.INPUT_LAYER_MASK_SURROGATE_SCALE,
            ),
            mask_dimension_option=self._enum_field(
                "input_layer_mask_dimension_option",
                self.config.INPUT_LAYER_MASK_DIMENSION_OPTION,
                MaskDimensionOptions,
            ),
        )

    def _output_projection_fields(self) -> ProjectionFields:
        return self.projection_fields_type(
            weight_option=self._optional_implementation_field(
                "output_layer_weight_option",
                self.config.OUTPUT_LAYER_WEIGHT_OPTION,
                DynamicWeightConfig,
            ),
            generator_depth=self._integer_field(
                "output_layer_generator_depth", self.config.OUTPUT_LAYER_GENERATOR_DEPTH
            ),
            weight_decay_schedule=self._enum_field(
                "output_layer_weight_decay_schedule",
                self.config.OUTPUT_LAYER_WEIGHT_DECAY_SCHEDULE,
                WeightDecayScheduleOptions,
            ),
            weight_decay_rate=self._float_field(
                "output_layer_weight_decay_rate",
                self.config.OUTPUT_LAYER_WEIGHT_DECAY_RATE,
            ),
            weight_decay_warmup_batches=self._integer_field(
                "output_layer_weight_decay_warmup_batches",
                self.config.OUTPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES,
            ),
            weight_normalization_option=self._enum_field(
                "output_layer_weight_normalization_option",
                self.config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_OPTION,
                WeightNormalizationOptions,
            ),
            weight_normalization_learnable_scale_flag=self._boolean_field(
                "output_layer_weight_normalization_learnable_scale_flag",
                self.config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_LEARNABLE_SCALE_FLAG,
            ),
            weight_normalization_position_option=self._enum_field(
                "output_layer_weight_normalization_position_option",
                self.config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION,
                WeightNormalizationPositionOptions,
            ),
            weight_bank_expansion_factor=self._enum_field(
                "output_layer_weight_bank_expansion_factor",
                self.config.OUTPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR,
                BankExpansionFactorOptions,
            ),
            bias_option=self._optional_implementation_field(
                "output_layer_bias_option",
                self.config.OUTPUT_LAYER_BIAS_OPTION,
                DynamicBiasConfig,
            ),
            bias_decay_schedule=self._enum_field(
                "output_layer_bias_decay_schedule",
                self.config.OUTPUT_LAYER_BIAS_DECAY_SCHEDULE,
                WeightDecayScheduleOptions,
            ),
            bias_decay_rate=self._float_field(
                "output_layer_bias_decay_rate", self.config.OUTPUT_LAYER_BIAS_DECAY_RATE
            ),
            bias_decay_warmup_batches=self._integer_field(
                "output_layer_bias_decay_warmup_batches",
                self.config.OUTPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES,
            ),
            bias_bank_expansion_factor=self._enum_field(
                "output_layer_bias_bank_expansion_factor",
                self.config.OUTPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR,
                BankExpansionFactorOptions,
            ),
            diagonal_option=self._optional_implementation_field(
                "output_layer_diagonal_option",
                self.config.OUTPUT_LAYER_DIAGONAL_OPTION,
                DynamicDiagonalConfig,
            ),
            row_mask_option=self._optional_implementation_field(
                "output_layer_row_mask_option",
                self.config.OUTPUT_LAYER_ROW_MASK_OPTION,
                AxisMaskConfig,
            ),
            mask_threshold=self._float_field(
                "output_layer_mask_threshold", self.config.OUTPUT_LAYER_MASK_THRESHOLD
            ),
            mask_floor=self._float_field(
                "output_layer_mask_floor", self.config.OUTPUT_LAYER_MASK_FLOOR
            ),
            mask_transition_width=self._float_field(
                "output_layer_mask_transition_width",
                self.config.OUTPUT_LAYER_MASK_TRANSITION_WIDTH,
            ),
            mask_surrogate_scale=self._float_field(
                "output_layer_mask_surrogate_scale",
                self.config.OUTPUT_LAYER_MASK_SURROGATE_SCALE,
            ),
            mask_dimension_option=self._enum_field(
                "output_layer_mask_dimension_option",
                self.config.OUTPUT_LAYER_MASK_DIMENSION_OPTION,
                MaskDimensionOptions,
            ),
        )

    def _read_dimensions(
        self, reader: RuntimeOverrideReader
    ) -> RuntimeDimensionsValues:
        fields = self.dimension_fields
        return self.runtime_dimensions_values_type(
            batch_size=reader.read(fields.batch_size),
            learning_rate=reader.read(fields.learning_rate),
            input_dim=reader.read(fields.input_dim),
            hidden_dim=reader.read(fields.hidden_dim),
            output_dim=reader.read(fields.output_dim),
        )

    def _read_stack(
        self, reader: RuntimeOverrideReader, fields: StackFields
    ) -> StackValues:
        return self.stack_values_type(
            hidden_dim=reader.read(fields.hidden_dim),
            num_layers=reader.read(fields.num_layers),
            last_layer_bias_option=reader.read(fields.last_layer_bias_option),
            apply_output_postprocessing_flag=reader.read(
                fields.apply_output_postprocessing_flag
            ),
            activation=reader.read(fields.activation),
            layer_norm_position=reader.read(fields.layer_norm_position),
            normalization=reader.read(fields.normalization),
            residual_connection_option=reader.read(fields.residual_connection_option),
            residual_block_size=reader.read(fields.residual_block_size),
            residual_rms_norm_epsilon=reader.read(fields.residual_rms_norm_epsilon),
            residual_model_flag=reader.read(fields.residual_model_flag),
            dropout_probability=reader.read(fields.dropout_probability),
            bias_flag=reader.read(fields.bias_flag),
        )

    def _read_optional_stack(
        self,
        reader: RuntimeOverrideReader,
        fields: OptionalStackFields,
    ) -> OptionalStackValues:
        return self.optional_stack_values_type(
            independent_flag=reader.read(fields.independent_flag),
            hidden_dim=reader.read(fields.hidden_dim),
            num_layers=reader.read(fields.num_layers),
            last_layer_bias_option=reader.read(fields.last_layer_bias_option),
            apply_output_postprocessing_flag=reader.read(
                fields.apply_output_postprocessing_flag
            ),
            activation=reader.read(fields.activation),
            layer_norm_position=reader.read(fields.layer_norm_position),
            normalization=reader.read(fields.normalization),
            residual_connection_option=reader.read(fields.residual_connection_option),
            residual_block_size=reader.read(fields.residual_block_size),
            residual_rms_norm_epsilon=reader.read(fields.residual_rms_norm_epsilon),
            residual_model_flag=reader.read(fields.residual_model_flag),
            dropout_probability=reader.read(fields.dropout_probability),
            bias_flag=reader.read(fields.bias_flag),
        )

    def _read_control(self, reader: RuntimeOverrideReader) -> ControlValues:
        fields = self.control_fields
        return self.control_values_type(
            stack_gate_flag=reader.read(fields.stack_gate_flag),
            gate_option=reader.read(fields.gate_option),
            gate_activation=reader.read(fields.gate_activation),
            shared_gate_config=reader.read(fields.shared_gate_config),
            stack_halting_flag=reader.read(fields.stack_halting_flag),
            halting_option=reader.read(fields.halting_option),
            halting_threshold=reader.read(fields.halting_threshold),
            halting_dropout=reader.read(fields.halting_dropout),
            halting_hidden_state_mode=reader.read(fields.halting_hidden_state_mode),
            memory_flag=reader.read(fields.memory_flag),
            memory_option=reader.read(fields.memory_option),
            memory_position_option=reader.read(fields.memory_position_option),
            memory_learning_rate=reader.read(fields.memory_learning_rate),
            memory_num_inner_steps=reader.read(fields.memory_num_inner_steps),
            recurrent_flag=reader.read(fields.recurrent_flag),
            recurrent_max_steps=reader.read(fields.recurrent_max_steps),
            recurrent_initial_iterations=reader.read(
                fields.recurrent_initial_iterations
            ),
            recurrent_gradient_transition_count=reader.read(
                fields.recurrent_gradient_transition_count
            ),
            recurrent_no_gradient_transition_count=reader.read(
                fields.recurrent_no_gradient_transition_count
            ),
            recurrent_iteration_increment=reader.read(
                fields.recurrent_iteration_increment
            ),
            recurrent_forward_calls_before_iteration_increment=reader.read(
                fields.recurrent_forward_calls_before_iteration_increment
            ),
            recurrent_smooth_iteration_growth_flag=reader.read(
                fields.recurrent_smooth_iteration_growth_flag
            ),
            recurrent_layer_norm_position=reader.read(
                fields.recurrent_layer_norm_position
            ),
            recurrent_normalization=reader.read(fields.recurrent_normalization),
            recurrent_stack_gate_flag=reader.read(fields.recurrent_stack_gate_flag),
            recurrent_gate_option=reader.read(fields.recurrent_gate_option),
            recurrent_gate_activation=reader.read(fields.recurrent_gate_activation),
            recurrent_stack_halting_flag=reader.read(
                fields.recurrent_stack_halting_flag
            ),
            recurrent_halting_option=reader.read(fields.recurrent_halting_option),
            recurrent_halting_threshold=reader.read(fields.recurrent_halting_threshold),
            recurrent_halting_dropout=reader.read(fields.recurrent_halting_dropout),
            recurrent_halting_hidden_state_mode=reader.read(
                fields.recurrent_halting_hidden_state_mode
            ),
        )

    def _read_weight(self, reader: RuntimeOverrideReader) -> WeightValues:
        fields = self.weight_fields
        return self.weight_values_type(
            enabled=reader.read(fields.enabled),
            option=reader.read(fields.option),
            generator_depth=reader.read(fields.generator_depth),
            decay_schedule=reader.read(fields.decay_schedule),
            decay_rate=reader.read(fields.decay_rate),
            decay_warmup_batches=reader.read(fields.decay_warmup_batches),
            normalization_option=reader.read(fields.normalization_option),
            normalization_learnable_scale_flag=reader.read(
                fields.normalization_learnable_scale_flag
            ),
            normalization_position_option=reader.read(
                fields.normalization_position_option
            ),
            bank_expansion_factor=reader.read(fields.bank_expansion_factor),
        )

    def _read_bias(self, reader: RuntimeOverrideReader) -> BiasValues:
        fields = self.bias_fields
        return self.bias_values_type(
            enabled=reader.read(fields.enabled),
            option=reader.read(fields.option),
            decay_schedule=reader.read(fields.decay_schedule),
            decay_rate=reader.read(fields.decay_rate),
            decay_warmup_batches=reader.read(fields.decay_warmup_batches),
            bank_expansion_factor=reader.read(fields.bank_expansion_factor),
        )

    def _read_diagonal(self, reader: RuntimeOverrideReader) -> DiagonalValues:
        fields = self.diagonal_fields
        return self.diagonal_values_type(
            enabled=reader.read(fields.enabled),
            option=reader.read(fields.option),
        )

    def _read_mask(self, reader: RuntimeOverrideReader) -> MaskValues:
        fields = self.mask_fields
        return self.mask_values_type(
            enabled=reader.read(fields.enabled),
            row_mask_option=reader.read(fields.row_mask_option),
            threshold=reader.read(fields.threshold),
            floor=reader.read(fields.floor),
            transition_width=reader.read(fields.transition_width),
            surrogate_scale=reader.read(fields.surrogate_scale),
            dimension_option=reader.read(fields.dimension_option),
        )

    def _read_projection(
        self,
        reader: RuntimeOverrideReader,
        fields: ProjectionFields,
    ) -> ProjectionValues:
        return self.projection_values_type(
            weight_option=reader.read(fields.weight_option),
            generator_depth=reader.read(fields.generator_depth),
            weight_decay_schedule=reader.read(fields.weight_decay_schedule),
            weight_decay_rate=reader.read(fields.weight_decay_rate),
            weight_decay_warmup_batches=reader.read(fields.weight_decay_warmup_batches),
            weight_normalization_option=reader.read(fields.weight_normalization_option),
            weight_normalization_learnable_scale_flag=reader.read(
                fields.weight_normalization_learnable_scale_flag
            ),
            weight_normalization_position_option=reader.read(
                fields.weight_normalization_position_option
            ),
            weight_bank_expansion_factor=reader.read(
                fields.weight_bank_expansion_factor
            ),
            bias_option=reader.read(fields.bias_option),
            bias_decay_schedule=reader.read(fields.bias_decay_schedule),
            bias_decay_rate=reader.read(fields.bias_decay_rate),
            bias_decay_warmup_batches=reader.read(fields.bias_decay_warmup_batches),
            bias_bank_expansion_factor=reader.read(fields.bias_bank_expansion_factor),
            diagonal_option=reader.read(fields.diagonal_option),
            row_mask_option=reader.read(fields.row_mask_option),
            mask_threshold=reader.read(fields.mask_threshold),
            mask_floor=reader.read(fields.mask_floor),
            mask_transition_width=reader.read(fields.mask_transition_width),
            mask_surrogate_scale=reader.read(fields.mask_surrogate_scale),
            mask_dimension_option=reader.read(fields.mask_dimension_option),
        )

    def _read_generation(
        self, reader: RuntimeOverrideReader, prefix: str = ""
    ) -> GenerationValues:
        return self.generation_values_type(
            weight_input_factor_source=reader.read(
                self._optional_enum_field(
                    f"{prefix}weight_input_factor_source",
                    getattr(self.config, f"{prefix.upper()}WEIGHT_INPUT_FACTOR_SOURCE"),
                    LowRankFactorSourceOptions,
                )
            ),
            weight_output_factor_source=reader.read(
                self._optional_enum_field(
                    f"{prefix}weight_output_factor_source",
                    getattr(
                        self.config, f"{prefix.upper()}WEIGHT_OUTPUT_FACTOR_SOURCE"
                    ),
                    LowRankFactorSourceOptions,
                )
            ),
            weight_mixture_num_experts=reader.read(
                self._optional_integer_field(
                    f"{prefix}weight_mixture_num_experts",
                    getattr(self.config, f"{prefix.upper()}WEIGHT_MIXTURE_NUM_EXPERTS"),
                )
            ),
            bias_mixture_num_experts=reader.read(
                self._optional_integer_field(
                    f"{prefix}bias_mixture_num_experts",
                    getattr(self.config, f"{prefix.upper()}BIAS_MIXTURE_NUM_EXPERTS"),
                )
            ),
            weight_mixture_top_k=reader.read(
                self._optional_integer_field(
                    f"{prefix}weight_mixture_top_k",
                    getattr(self.config, f"{prefix.upper()}WEIGHT_MIXTURE_TOP_K"),
                )
            ),
            bias_mixture_top_k=reader.read(
                self._optional_integer_field(
                    f"{prefix}bias_mixture_top_k",
                    getattr(self.config, f"{prefix.upper()}BIAS_MIXTURE_TOP_K"),
                )
            ),
            weight_mixture_normalize_probabilities_flag=reader.read(
                self._optional_boolean_field(
                    f"{prefix}weight_mixture_normalize_probabilities_flag",
                    getattr(
                        self.config,
                        f"{prefix.upper()}WEIGHT_MIXTURE_NORMALIZE_PROBABILITIES_FLAG",
                    ),
                )
            ),
            bias_mixture_normalize_probabilities_flag=reader.read(
                self._optional_boolean_field(
                    f"{prefix}bias_mixture_normalize_probabilities_flag",
                    getattr(
                        self.config,
                        f"{prefix.upper()}BIAS_MIXTURE_NORMALIZE_PROBABILITIES_FLAG",
                    ),
                )
            ),
            weight_input_factor_generator_stack=self._read_optional_stack(
                reader,
                self.optional_stack_fields_type.for_stack(
                    f"{prefix}weight_input_factor_generator_stack"
                ),
            ),
            weight_output_factor_generator_stack=self._read_optional_stack(
                reader,
                self.optional_stack_fields_type.for_stack(
                    f"{prefix}weight_output_factor_generator_stack"
                ),
            ),
            weight_coefficient_generator_stack=self._read_optional_stack(
                reader,
                self.optional_stack_fields_type.for_stack(
                    f"{prefix}weight_coefficient_generator_stack"
                ),
            ),
            weight_mixture_router_generator_stack=self._read_optional_stack(
                reader,
                self.optional_stack_fields_type.for_stack(
                    f"{prefix}weight_mixture_router_generator_stack"
                ),
            ),
            bias_mixture_router_generator_stack=self._read_optional_stack(
                reader,
                self.optional_stack_fields_type.for_stack(
                    f"{prefix}bias_mixture_router_generator_stack"
                ),
            ),
        )

    def _read_grouping(self, reader: RuntimeOverrideReader, prefix: str = ""):
        values: dict[str, object] = {}
        key = prefix + "grouping_scope"
        values["grouping_scope"] = reader.read(
            self._optional_enum_field(
                key,
                getattr(self.config, key.upper()),
                AdaptiveParameterGroupingScopeOptions,
            )
        )
        key = prefix + "group_count"
        values["group_count"] = reader.read(
            self._optional_integer_field(key, getattr(self.config, key.upper()))
        )
        key = prefix + "chunk_size"
        values["chunk_size"] = reader.read(
            self._optional_integer_field(key, getattr(self.config, key.upper()))
        )
        key = prefix + "grouping_sequence_length"
        values["grouping_sequence_length"] = reader.read(
            self._optional_integer_field(key, getattr(self.config, key.upper()))
        )
        key = prefix + "grouping_input_order"
        values["grouping_input_order"] = reader.read(
            self._optional_enum_field(
                key,
                getattr(self.config, key.upper()),
                AdaptiveParameterInputOrderOptions,
            )
        )
        key = prefix + "grouping_method"
        values["grouping_method"] = reader.read(
            self._optional_implementation_field(
                key, getattr(self.config, key.upper()), GroupingConfig
            )
        )
        key = prefix + "grouping_summary_normalization"
        values["grouping_summary_normalization"] = reader.read(
            self._optional_enum_field(
                key,
                getattr(self.config, key.upper()),
                SummaryNormalizationOptions,
            )
        )
        key = prefix + "grouping_model_config"
        values["grouping_model_config"] = reader.read(
            self._optional_instance_field(
                key, getattr(self.config, key.upper()), ConfigBase
            )
        )
        key = prefix + "grouping_model_option"
        values["grouping_model_option"] = reader.read(
            self._optional_enum_field(
                key,
                getattr(self.config, key.upper()),
                _learned_grouping.GroupingModelOptions,
            )
        )
        key = prefix + "grouping_model_bias_flag"
        values["grouping_model_bias_flag"] = reader.read(
            self._optional_boolean_field(key, getattr(self.config, key.upper()))
        )
        key = prefix + "grouping_attention_hidden_dim"
        values["grouping_attention_hidden_dim"] = reader.read(
            self._optional_integer_field(key, getattr(self.config, key.upper()))
        )
        key = prefix + "grouping_rms_norm_epsilon"
        values["grouping_rms_norm_epsilon"] = reader.read(
            self._optional_float_field(key, getattr(self.config, key.upper()))
        )
        return _learned_grouping.grouping_from_fields(values, prefix=prefix or "main")

    def _read_runtime_values(self, reader: RuntimeOverrideReader) -> _Values:
        return self.runtime_values_type(
            residual_coefficient_activation=reader.read(
                self._enum_field(
                    "residual_coefficient_activation",
                    self.config.RESIDUAL_COEFFICIENT_ACTIVATION,
                    ActivationOptions,
                )
            ),
            grouping=self._read_grouping(reader),
            input_grouping=self._read_grouping(reader, "input_layer_"),
            output_grouping=self._read_grouping(reader, "output_layer_"),
            generation=self._read_generation(reader),
            input_generation=self._read_generation(reader, "input_layer_"),
            output_generation=self._read_generation(reader, "output_layer_"),
            dimensions=self._read_dimensions(reader),
            main_stack=self._read_stack(reader, self.main_stack_fields),
            submodule_stack=self._read_stack(reader, self.submodule_stack_fields),
            adaptive_generator_stack=self._read_stack(
                reader, self.adaptive_generator_stack_fields
            ),
            residual_stack=self._read_optional_stack(
                reader, self.residual_stack_fields
            ),
            gate_stack=self._read_optional_stack(reader, self.gate_stack_fields),
            halting_stack=self._read_optional_stack(reader, self.halting_stack_fields),
            memory_stack=self._read_optional_stack(reader, self.memory_stack_fields),
            recurrent_gate_stack=self._read_optional_stack(
                reader, self.recurrent_gate_stack_fields
            ),
            recurrent_halting_stack=self._read_optional_stack(
                reader, self.recurrent_halting_stack_fields
            ),
            weight_generator_stack=self._read_optional_stack(
                reader, self.weight_generator_stack_fields
            ),
            bias_generator_stack=self._read_optional_stack(
                reader, self.bias_generator_stack_fields
            ),
            diagonal_generator_stack=self._read_optional_stack(
                reader, self.diagonal_generator_stack_fields
            ),
            mask_generator_stack=self._read_optional_stack(
                reader, self.mask_generator_stack_fields
            ),
            control=self._read_control(reader),
            weight=self._read_weight(reader),
            bias=self._read_bias(reader),
            diagonal=self._read_diagonal(reader),
            mask=self._read_mask(reader),
            input_projection=self._read_projection(
                reader, self.input_projection_fields
            ),
            output_projection=self._read_projection(
                reader, self.output_projection_fields
            ),
        )

    def _normalize_overrides(
        self, overrides: Mapping[str, object]
    ) -> dict[str, object]:
        normalized: dict[str, object] = {}
        for supplied_key, value in cast(Mapping[object, object], overrides).items():
            if not isinstance(supplied_key, str):
                raise TypeError(
                    f"{self.package}: runtime override keys must be str, got "
                    f"{type(supplied_key).__name__}"
                )
            runtime_field = self.fields.get(supplied_key)
            if runtime_field is None:
                accepted = ", ".join(sorted(self.fields))
                raise ValueError(
                    f"{self.package}: unknown runtime key {supplied_key!r}; "
                    f"accepted keys: {accepted}"
                )
            if not runtime_field.accepts(value):
                raise_type_error(
                    self.package, supplied_key, value, runtime_field.expected
                )
            normalized[supplied_key] = value
        return normalized

    def runtime_default_values(
        self,
        overrides: Mapping[str, object] | None,
    ) -> _Values:
        normalized = self._normalize_overrides(overrides or {})
        return self._read_runtime_values(self.runtime_override_reader_type(normalized))
