from typing import Generic, TypeVar

from model_runtime.packages.settings_validation import BuildSettingsValidation
from models._adaptive_runtime_defaults import AdaptiveRuntimeDefaults
from models._adaptive_runtime_fields import (
    OptionalStackFields,
    OptionalStackValues,
    RuntimeDefaultValues,
    StackFields,
    StackValues,
)

_Values = TypeVar("_Values", bound=RuntimeDefaultValues)


class AdaptiveRuntimeValidation(Generic[_Values]):
    def __init__(
        self,
        schema: AdaptiveRuntimeDefaults[_Values],
        validation: BuildSettingsValidation,
    ) -> None:
        self.schema = schema
        self.validation = validation

    def _validate_required_positive(self, values: RuntimeDefaultValues) -> None:
        self.validation.dimensions(values.dimensions)
        for field, value in (
            (self.schema.main_stack_fields.num_layers, values.main_stack.num_layers),
            (
                self.schema.submodule_stack_fields.hidden_dim,
                values.submodule_stack.hidden_dim,
            ),
            (
                self.schema.submodule_stack_fields.num_layers,
                values.submodule_stack.num_layers,
            ),
            (
                self.schema.adaptive_generator_stack_fields.hidden_dim,
                values.adaptive_generator_stack.hidden_dim,
            ),
            (
                self.schema.adaptive_generator_stack_fields.num_layers,
                values.adaptive_generator_stack.num_layers,
            ),
            (
                self.schema.control_fields.recurrent_max_steps,
                values.control.recurrent_max_steps,
            ),
        ):
            self.validation.positive(field.key, value)

    def _validate_stack_probability(
        self, fields: StackFields, values: StackValues
    ) -> None:
        self.validation.probability(
            fields.dropout_probability.key, values.dropout_probability
        )

    def _validate_optional_stack_pattern(
        self,
        fields: OptionalStackFields,
        values: OptionalStackValues,
    ) -> None:
        if values.hidden_dim is not None:
            self.validation.positive(fields.hidden_dim.key, values.hidden_dim)
        if values.num_layers is not None:
            self.validation.positive(fields.num_layers.key, values.num_layers)
        if values.dropout_probability is not None:
            self.validation.probability(
                fields.dropout_probability.key, values.dropout_probability
            )

    def _validate_stack_patterns(self, values: RuntimeDefaultValues) -> None:
        for fields, stack_values in (
            (self.schema.main_stack_fields, values.main_stack),
            (self.schema.submodule_stack_fields, values.submodule_stack),
            (
                self.schema.adaptive_generator_stack_fields,
                values.adaptive_generator_stack,
            ),
        ):
            self._validate_stack_probability(fields, stack_values)
        for fields, stack_values in (
            (self.schema.residual_stack_fields, values.residual_stack),
            (self.schema.gate_stack_fields, values.gate_stack),
            (self.schema.halting_stack_fields, values.halting_stack),
            (self.schema.memory_stack_fields, values.memory_stack),
            (self.schema.recurrent_gate_stack_fields, values.recurrent_gate_stack),
            (
                self.schema.recurrent_halting_stack_fields,
                values.recurrent_halting_stack,
            ),
            (self.schema.weight_generator_stack_fields, values.weight_generator_stack),
            (self.schema.bias_generator_stack_fields, values.bias_generator_stack),
            (
                self.schema.diagonal_generator_stack_fields,
                values.diagonal_generator_stack,
            ),
            (self.schema.mask_generator_stack_fields, values.mask_generator_stack),
        ):
            self._validate_optional_stack_pattern(fields, stack_values)

    def _validate_probabilities(self, values: RuntimeDefaultValues) -> None:
        control = values.control
        mask = values.mask
        input_projection = values.input_projection
        output_projection = values.output_projection
        self.validation.dropout(
            self.schema.control_fields.halting_dropout.key, control.halting_dropout
        )
        self.validation.halting_threshold(
            self.schema.control_fields.halting_threshold.key, control.halting_threshold
        )
        self.validation.dropout(
            self.schema.control_fields.recurrent_halting_dropout.key,
            control.recurrent_halting_dropout,
        )
        self.validation.halting_threshold(
            self.schema.control_fields.recurrent_halting_threshold.key,
            control.recurrent_halting_threshold,
        )
        for field, value in (
            (self.schema.mask_fields.threshold, mask.threshold),
            (self.schema.mask_fields.floor, mask.floor),
            (
                self.schema.input_projection_fields.mask_threshold,
                input_projection.mask_threshold,
            ),
            (
                self.schema.input_projection_fields.mask_floor,
                input_projection.mask_floor,
            ),
            (
                self.schema.output_projection_fields.mask_threshold,
                output_projection.mask_threshold,
            ),
            (
                self.schema.output_projection_fields.mask_floor,
                output_projection.mask_floor,
            ),
        ):
            self.validation.probability(field.key, value)

    def _validate_positive_controls(self, values: RuntimeDefaultValues) -> None:
        mask = values.mask
        input_projection = values.input_projection
        output_projection = values.output_projection
        for field, value in (
            (self.schema.mask_fields.surrogate_scale, mask.surrogate_scale),
            (self.schema.mask_fields.transition_width, mask.transition_width),
            (
                self.schema.input_projection_fields.mask_surrogate_scale,
                input_projection.mask_surrogate_scale,
            ),
            (
                self.schema.input_projection_fields.mask_transition_width,
                input_projection.mask_transition_width,
            ),
            (
                self.schema.output_projection_fields.mask_surrogate_scale,
                output_projection.mask_surrogate_scale,
            ),
            (
                self.schema.output_projection_fields.mask_transition_width,
                output_projection.mask_transition_width,
            ),
        ):
            self.validation.positive(field.key, value)

    def _validate_decay_values(self, values: RuntimeDefaultValues) -> None:
        input_projection = values.input_projection
        output_projection = values.output_projection
        for field, value in (
            (self.schema.weight_fields.decay_rate, values.weight.decay_rate),
            (self.schema.bias_fields.decay_rate, values.bias.decay_rate),
            (
                self.schema.input_projection_fields.weight_decay_rate,
                input_projection.weight_decay_rate,
            ),
            (
                self.schema.input_projection_fields.bias_decay_rate,
                input_projection.bias_decay_rate,
            ),
            (
                self.schema.output_projection_fields.weight_decay_rate,
                output_projection.weight_decay_rate,
            ),
            (
                self.schema.output_projection_fields.bias_decay_rate,
                output_projection.bias_decay_rate,
            ),
        ):
            self.validation.nonnegative(field.key, value)
        for field, value in (
            (
                self.schema.weight_fields.decay_warmup_batches,
                values.weight.decay_warmup_batches,
            ),
            (
                self.schema.bias_fields.decay_warmup_batches,
                values.bias.decay_warmup_batches,
            ),
            (
                self.schema.input_projection_fields.weight_decay_warmup_batches,
                input_projection.weight_decay_warmup_batches,
            ),
            (
                self.schema.input_projection_fields.bias_decay_warmup_batches,
                input_projection.bias_decay_warmup_batches,
            ),
            (
                self.schema.output_projection_fields.weight_decay_warmup_batches,
                output_projection.weight_decay_warmup_batches,
            ),
            (
                self.schema.output_projection_fields.bias_decay_warmup_batches,
                output_projection.bias_decay_warmup_batches,
            ),
        ):
            self.validation.nonnegative(field.key, value)

    def _validate_control_invariants(self, values: RuntimeDefaultValues) -> None:
        control = values.control
        if control.memory_learning_rate is not None:
            self.validation.positive(
                self.schema.control_fields.memory_learning_rate.key,
                control.memory_learning_rate,
            )
        if control.memory_num_inner_steps is not None:
            self.validation.positive(
                self.schema.control_fields.memory_num_inner_steps.key,
                control.memory_num_inner_steps,
            )
        for enabled, option, flag_field, option_field in (
            (
                values.weight.enabled,
                values.weight.option,
                self.schema.weight_fields.enabled,
                self.schema.weight_fields.option,
            ),
            (
                values.bias.enabled,
                values.bias.option,
                self.schema.bias_fields.enabled,
                self.schema.bias_fields.option,
            ),
            (
                values.diagonal.enabled,
                values.diagonal.option,
                self.schema.diagonal_fields.enabled,
                self.schema.diagonal_fields.option,
            ),
            (
                values.mask.enabled,
                values.mask.row_mask_option,
                self.schema.mask_fields.enabled,
                self.schema.mask_fields.row_mask_option,
            ),
        ):
            if enabled and option is None:
                raise ValueError(
                    f"{self.schema.package}: runtime key {option_field.key!r} must be set when "
                    f"{flag_field.key!r} is True"
                )
        self.validation.gate(control.stack_gate_flag, control.gate_option)
        self.validation.gate(
            control.recurrent_stack_gate_flag,
            control.recurrent_gate_option,
            prefix="recurrent_",
        )
        self.validation.shared_gate(control.stack_gate_flag, control.shared_gate_config)

    def validate(self, values: RuntimeDefaultValues) -> None:
        self._validate_required_positive(values)
        self._validate_stack_patterns(values)
        self._validate_probabilities(values)
        self._validate_positive_controls(values)
        self._validate_decay_values(values)
        self._validate_control_invariants(values)
