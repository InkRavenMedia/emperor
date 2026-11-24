from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final

from emperor.layers import LastLayerBiasOptions
from models._recurrent_variants import variant_updates
from models._runtime_binding import bind_runtime_options
from models.linears.linear_adaptive import SETTINGS_VALIDATION
from models.linears.linear_adaptive._residual import (
    ResidualStackOptions,
    ResidualStackSource,
    inherit_if_unset,
    resolve_residual_stack_options,
)
from models.linears.linear_adaptive._runtime_default_values import (
    BiasValues,
    DiagonalValues,
    GenerationValues,
    MaskValues,
    OptionalStackValues,
    ProjectionValues,
    RuntimeDefaults,
    RuntimeDefaultValues,
    StackValues,
    WeightValues,
    runtime_default_values,
)
from models.linears.linear_adaptive.runtime_options import (
    AdaptiveBiasOptions,
    AdaptiveDiagonalOptions,
    AdaptiveGenerationOptions,
    AdaptiveMaskOptions,
    AdaptiveProjectionOptions,
    AdaptiveWeightOptions,
    GateOptions,
    GeneratorStackOptions,
    HaltingOptions,
    MemoryOptions,
    RecurrenceOptions,
    RuntimeOptions,
    StackOptions,
)

from . import config


def _stack(values: StackValues) -> StackOptions:
    return StackOptions(
        hidden_dim=values.hidden_dim,
        num_layers=values.num_layers,
        last_layer_bias_option=values.last_layer_bias_option,
        apply_output_postprocessing_flag=values.apply_output_postprocessing_flag,
        activation=values.activation,
        layer_norm_position=values.layer_norm_position,
        normalization=values.normalization,
        residual_connection_option=values.residual_connection_option,
        residual_block_size=values.residual_block_size,
        residual_rms_norm_epsilon=values.residual_rms_norm_epsilon,
        residual_model_flag=values.residual_model_flag,
        dropout_probability=values.dropout_probability,
        bias_flag=values.bias_flag,
    )


def _resolved_stack(
    values: OptionalStackValues,
    defaults: StackOptions,
) -> StackOptions:
    if not values.independent_flag:
        return defaults
    return replace(
        defaults,
        hidden_dim=inherit_if_unset(values.hidden_dim, defaults.hidden_dim),
        num_layers=inherit_if_unset(values.num_layers, defaults.num_layers),
        last_layer_bias_option=inherit_if_unset(
            values.last_layer_bias_option, defaults.last_layer_bias_option
        ),
        apply_output_postprocessing_flag=inherit_if_unset(
            values.apply_output_postprocessing_flag,
            defaults.apply_output_postprocessing_flag,
        ),
        activation=inherit_if_unset(values.activation, defaults.activation),
        layer_norm_position=inherit_if_unset(
            values.layer_norm_position, defaults.layer_norm_position
        ),
        normalization=inherit_if_unset(values.normalization, defaults.normalization),
        residual_connection_option=inherit_if_unset(
            values.residual_connection_option, defaults.residual_connection_option
        ),
        residual_block_size=values.residual_block_size,
        residual_rms_norm_epsilon=values.residual_rms_norm_epsilon,
        residual_model_flag=values.residual_model_flag,
        dropout_probability=inherit_if_unset(
            values.dropout_probability, defaults.dropout_probability
        ),
        bias_flag=inherit_if_unset(values.bias_flag, defaults.bias_flag),
    )


def _generator_stack(
    values: OptionalStackValues,
    defaults: StackOptions,
) -> GeneratorStackOptions:
    return GeneratorStackOptions(
        independent=values.independent_flag,
        stack=_resolved_stack(values, defaults),
    )


@dataclass(frozen=True, slots=True)
class _ResolvedStacks:
    main: StackOptions
    submodule: StackOptions
    residual: ResidualStackOptions
    gate: StackOptions
    halting: StackOptions
    memory: StackOptions
    recurrent_gate: StackOptions
    recurrent_halting: StackOptions
    adaptive_generator: StackOptions


def _resolve_stacks(values: RuntimeDefaultValues) -> _ResolvedStacks:
    main = _stack(values.main_stack)
    submodule = _stack(values.submodule_stack)
    residual_values = values.residual_stack
    residual = resolve_residual_stack_options(
        ResidualStackSource(
            coefficient_activation=values.residual_coefficient_activation,
            independent_flag=residual_values.independent_flag,
            hidden_dim=residual_values.hidden_dim,
            num_layers=residual_values.num_layers,
            activation=residual_values.activation,
            layer_norm_position=residual_values.layer_norm_position,
            normalization=residual_values.normalization,
            residual_connection_option=residual_values.residual_connection_option,
            residual_block_size=residual_values.residual_block_size,
            residual_rms_norm_epsilon=residual_values.residual_rms_norm_epsilon,
            residual_model_flag=residual_values.residual_model_flag,
            dropout_probability=residual_values.dropout_probability,
            last_layer_bias_option=residual_values.last_layer_bias_option,
            apply_output_postprocessing_flag=(
                residual_values.apply_output_postprocessing_flag
            ),
            bias_flag=residual_values.bias_flag,
        ),
        submodule,
    )
    gate = _resolved_stack(values.gate_stack, submodule)
    halting = _resolved_stack(
        values.halting_stack,
        replace(
            submodule,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
        ),
    )
    memory = _resolved_stack(values.memory_stack, submodule)
    return _ResolvedStacks(
        main=main,
        submodule=submodule,
        residual=residual,
        gate=gate,
        halting=halting,
        memory=memory,
        recurrent_gate=_resolved_stack(values.recurrent_gate_stack, gate),
        recurrent_halting=_resolved_stack(values.recurrent_halting_stack, halting),
        adaptive_generator=_stack(values.adaptive_generator_stack),
    )


@dataclass(frozen=True, slots=True)
class _ControlDefaults:
    gate: GateOptions
    halting: HaltingOptions
    memory: MemoryOptions
    recurrence: RecurrenceOptions


def _resolve_control_defaults(
    values: RuntimeDefaultValues,
    stacks: _ResolvedStacks,
) -> _ControlDefaults:
    control = values.control
    return _ControlDefaults(
        gate=GateOptions(
            enabled=control.stack_gate_flag,
            option=control.gate_option,
            activation=control.gate_activation,
            stack=stacks.gate,
            shared_config=control.shared_gate_config,
        ),
        halting=HaltingOptions(
            enabled=control.stack_halting_flag,
            threshold=control.halting_threshold,
            dropout_probability=control.halting_dropout,
            hidden_state_mode=control.halting_hidden_state_mode,
            stack=stacks.halting,
        ),
        memory=MemoryOptions(
            enabled=control.memory_flag,
            option=control.memory_option,
            position=control.memory_position_option,
            test_time_training_learning_rate=control.memory_learning_rate,
            test_time_training_num_inner_steps=control.memory_num_inner_steps,
            stack=stacks.memory,
        ),
        recurrence=RecurrenceOptions(
            enabled=control.recurrent_flag,
            max_steps=control.recurrent_max_steps,
            initial_iterations=control.recurrent_initial_iterations,
            gradient_transition_count=control.recurrent_gradient_transition_count,
            no_gradient_transition_count=control.recurrent_no_gradient_transition_count,
            iteration_increment=control.recurrent_iteration_increment,
            forward_calls_before_iteration_increment=(
                control.recurrent_forward_calls_before_iteration_increment
            ),
            smooth_iteration_growth_flag=(
                control.recurrent_smooth_iteration_growth_flag
            ),
            layer_norm_position=control.recurrent_layer_norm_position,
            normalization=control.recurrent_normalization,
            gate=GateOptions(
                enabled=control.recurrent_stack_gate_flag,
                option=control.recurrent_gate_option,
                activation=control.recurrent_gate_activation,
                stack=stacks.recurrent_gate,
            ),
            halting=HaltingOptions(
                enabled=control.recurrent_stack_halting_flag,
                threshold=control.recurrent_halting_threshold,
                dropout_probability=control.recurrent_halting_dropout,
                hidden_state_mode=control.recurrent_halting_hidden_state_mode,
                stack=stacks.recurrent_halting,
            ),
        ),
    )


def _projection(values: ProjectionValues) -> AdaptiveProjectionOptions:
    return AdaptiveProjectionOptions(
        weight_option=values.weight_option,
        generator_depth=values.generator_depth,
        weight_decay_schedule=values.weight_decay_schedule,
        weight_decay_rate=values.weight_decay_rate,
        weight_decay_warmup_batches=values.weight_decay_warmup_batches,
        weight_normalization_option=values.weight_normalization_option,
        weight_normalization_learnable_scale_flag=values.weight_normalization_learnable_scale_flag,
        weight_normalization_position_option=(
            values.weight_normalization_position_option
        ),
        weight_bank_expansion_factor=values.weight_bank_expansion_factor,
        bias_option=values.bias_option,
        bias_decay_schedule=values.bias_decay_schedule,
        bias_decay_rate=values.bias_decay_rate,
        bias_decay_warmup_batches=values.bias_decay_warmup_batches,
        bias_bank_expansion_factor=values.bias_bank_expansion_factor,
        diagonal_option=values.diagonal_option,
        row_mask_option=values.row_mask_option,
        mask_dimension_option=values.mask_dimension_option,
        mask_threshold=values.mask_threshold,
        mask_surrogate_scale=values.mask_surrogate_scale,
        mask_floor=values.mask_floor,
        mask_transition_width=values.mask_transition_width,
    )


@dataclass(frozen=True, slots=True)
class _AdaptiveDefaults:
    weight: AdaptiveWeightOptions
    bias: AdaptiveBiasOptions
    diagonal: AdaptiveDiagonalOptions
    mask: AdaptiveMaskOptions
    input_projection: AdaptiveProjectionOptions
    output_projection: AdaptiveProjectionOptions


def _resolve_adaptive_defaults(
    values: RuntimeDefaultValues,
    adaptive_generator_stack: StackOptions,
) -> _AdaptiveDefaults:
    weight: WeightValues = values.weight
    bias: BiasValues = values.bias
    diagonal: DiagonalValues = values.diagonal
    mask: MaskValues = values.mask
    return _AdaptiveDefaults(
        weight=AdaptiveWeightOptions(
            enabled=weight.enabled,
            option=weight.option,
            generator_depth=weight.generator_depth,
            normalization_option=weight.normalization_option,
            normalization_learnable_scale_flag=weight.normalization_learnable_scale_flag,
            normalization_position_option=weight.normalization_position_option,
            decay_schedule=weight.decay_schedule,
            decay_rate=weight.decay_rate,
            decay_warmup_batches=weight.decay_warmup_batches,
            bank_expansion_factor=weight.bank_expansion_factor,
            generator_stack=_generator_stack(
                values.weight_generator_stack,
                adaptive_generator_stack,
            ),
        ),
        bias=AdaptiveBiasOptions(
            enabled=bias.enabled,
            option=bias.option,
            decay_schedule=bias.decay_schedule,
            decay_rate=bias.decay_rate,
            decay_warmup_batches=bias.decay_warmup_batches,
            bank_expansion_factor=bias.bank_expansion_factor,
            generator_stack=_generator_stack(
                values.bias_generator_stack,
                adaptive_generator_stack,
            ),
        ),
        diagonal=AdaptiveDiagonalOptions(
            enabled=diagonal.enabled,
            option=diagonal.option,
            generator_stack=_generator_stack(
                values.diagonal_generator_stack,
                adaptive_generator_stack,
            ),
        ),
        mask=AdaptiveMaskOptions(
            enabled=mask.enabled,
            row_mask_option=mask.row_mask_option,
            dimension_option=mask.dimension_option,
            threshold=mask.threshold,
            surrogate_scale=mask.surrogate_scale,
            floor=mask.floor,
            transition_width=mask.transition_width,
            generator_stack=_generator_stack(
                values.mask_generator_stack,
                adaptive_generator_stack,
            ),
        ),
        input_projection=_projection(values.input_projection),
        output_projection=_projection(values.output_projection),
    )


def _resolve_generation(
    values: GenerationValues, defaults: StackOptions, factor_defaults: StackOptions
) -> AdaptiveGenerationOptions:
    for name in (
        "weight_input_factor",
        "weight_output_factor",
        "weight_coefficient",
        "weight_mixture_router",
        "bias_mixture_router",
    ):
        stack = getattr(values, f"{name}_generator_stack")
        if stack.hidden_dim is not None:
            SETTINGS_VALIDATION.positive(
                f"{name}_generator_stack_hidden_dim", stack.hidden_dim
            )
        if stack.num_layers is not None:
            SETTINGS_VALIDATION.positive(
                f"{name}_generator_stack_num_layers", stack.num_layers
            )
        if stack.dropout_probability is not None:
            SETTINGS_VALIDATION.probability(
                f"{name}_generator_stack_dropout_probability", stack.dropout_probability
            )
    return AdaptiveGenerationOptions(
        weight_input_factor_source=values.weight_input_factor_source,
        weight_output_factor_source=values.weight_output_factor_source,
        weight_mixture_num_experts=values.weight_mixture_num_experts,
        bias_mixture_num_experts=values.bias_mixture_num_experts,
        weight_mixture_top_k=values.weight_mixture_top_k,
        bias_mixture_top_k=values.bias_mixture_top_k,
        weight_mixture_normalize_probabilities_flag=values.weight_mixture_normalize_probabilities_flag,
        bias_mixture_normalize_probabilities_flag=values.bias_mixture_normalize_probabilities_flag,
        weight_input_factor_generator_stack=_generator_stack(
            values.weight_input_factor_generator_stack, factor_defaults
        ),
        weight_output_factor_generator_stack=_generator_stack(
            values.weight_output_factor_generator_stack, factor_defaults
        ),
        weight_coefficient_generator_stack=_generator_stack(
            values.weight_coefficient_generator_stack, factor_defaults
        ),
        weight_mixture_router_generator_stack=_generator_stack(
            values.weight_mixture_router_generator_stack, defaults
        ),
        bias_mixture_router_generator_stack=_generator_stack(
            values.bias_mixture_router_generator_stack, defaults
        ),
    )


def _runtime(values: RuntimeDefaultValues) -> RuntimeOptions:
    stacks = _resolve_stacks(values)
    control = _resolve_control_defaults(values, stacks)
    adaptive = _resolve_adaptive_defaults(values, stacks.adaptive_generator)
    dimensions = values.dimensions
    return RuntimeOptions(
        grouping_config=values.grouping,
        generation=_resolve_generation(
            values.generation,
            stacks.adaptive_generator,
            _resolved_stack(values.weight_generator_stack, stacks.adaptive_generator),
        ),
        batch_size=dimensions.batch_size,
        learning_rate=dimensions.learning_rate,
        input_dim=dimensions.input_dim,
        hidden_dim=dimensions.hidden_dim,
        output_dim=dimensions.output_dim,
        stack=stacks.main,
        submodule_stack=stacks.submodule,
        residual_stack=stacks.residual,
        gate=control.gate,
        halting=control.halting,
        memory=control.memory,
        recurrence=control.recurrence,
        adaptive_generator_stack=stacks.adaptive_generator,
        weight=adaptive.weight,
        bias=adaptive.bias,
        diagonal=adaptive.diagonal,
        mask=adaptive.mask,
        input_projection=replace(
            adaptive.input_projection,
            grouping_config=values.input_grouping,
            generation=_resolve_generation(
                values.input_generation,
                stacks.adaptive_generator,
                stacks.adaptive_generator,
            ),
        ),
        output_projection=replace(
            adaptive.output_projection,
            grouping_config=values.output_grouping,
            generation=_resolve_generation(
                values.output_generation,
                stacks.adaptive_generator,
                stacks.adaptive_generator,
            ),
        ),
        halting_option=values.control.halting_option,
        recurrent_halting_option=values.control.recurrent_halting_option,
    )


@bind_runtime_options(config, variant_updates)
def runtime_from_flat(
    overrides: Mapping[str, object] | None = None,
) -> RuntimeOptions:
    values = runtime_default_values(overrides)
    RuntimeDefaults.validation.validate(values)
    return _runtime(values)


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_flat()
