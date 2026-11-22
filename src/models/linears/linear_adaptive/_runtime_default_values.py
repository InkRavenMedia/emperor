from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from models import _adaptive_runtime_fields as fields
from models._adaptive_runtime_defaults import AdaptiveRuntimeDefaults
from models._adaptive_runtime_validation import AdaptiveRuntimeValidation
from models.linears.linear_adaptive import SETTINGS_VALIDATION, config

_Value = TypeVar("_Value")


class RuntimeField(fields.RuntimeField[_Value]):
    __slots__ = ()
    package = __name__.rpartition(".")[0]


class RuntimeOverrideReader(fields.RuntimeOverrideReader):
    __slots__ = ()


class RuntimeDimensionsValues(fields.RuntimeDimensionsValues):
    __slots__ = ()


class RuntimeDimensionsFields(fields.RuntimeDimensionsFields):
    __slots__ = ()


class StackValues(fields.StackValues):
    __slots__ = ()


class StackFields(fields.StackFields):
    __slots__ = ()


class OptionalStackValues(fields.OptionalStackValues):
    __slots__ = ()


class OptionalStackFields(fields.OptionalStackFields):
    __slots__ = ()
    config = config
    field_type = RuntimeField


class ControlValues(fields.ControlValues):
    __slots__ = ()


class ControlFields(fields.ControlFields):
    __slots__ = ()


class WeightValues(fields.WeightValues):
    __slots__ = ()


class WeightFields(fields.WeightFields):
    __slots__ = ()


class BiasValues(fields.BiasValues):
    __slots__ = ()


class BiasFields(fields.BiasFields):
    __slots__ = ()


class DiagonalValues(fields.DiagonalValues):
    __slots__ = ()


class DiagonalFields(fields.DiagonalFields):
    __slots__ = ()


class MaskValues(fields.MaskValues):
    __slots__ = ()


class MaskFields(fields.MaskFields):
    __slots__ = ()


class ProjectionValues(fields.ProjectionValues):
    __slots__ = ()


class ProjectionFields(fields.ProjectionFields):
    __slots__ = ()


class GenerationValues(fields.GenerationValues):
    __slots__ = ()
    weight_input_factor_generator_stack: OptionalStackValues
    weight_output_factor_generator_stack: OptionalStackValues
    weight_coefficient_generator_stack: OptionalStackValues
    weight_mixture_router_generator_stack: OptionalStackValues
    bias_mixture_router_generator_stack: OptionalStackValues


class RuntimeDefaultValues(fields.RuntimeDefaultValues):
    __slots__ = ()
    generation: GenerationValues
    input_generation: GenerationValues
    output_generation: GenerationValues
    dimensions: RuntimeDimensionsValues
    main_stack: StackValues
    submodule_stack: StackValues
    adaptive_generator_stack: StackValues
    residual_stack: OptionalStackValues
    gate_stack: OptionalStackValues
    halting_stack: OptionalStackValues
    memory_stack: OptionalStackValues
    recurrent_gate_stack: OptionalStackValues
    recurrent_halting_stack: OptionalStackValues
    weight_generator_stack: OptionalStackValues
    bias_generator_stack: OptionalStackValues
    diagonal_generator_stack: OptionalStackValues
    mask_generator_stack: OptionalStackValues
    control: ControlValues
    weight: WeightValues
    bias: BiasValues
    diagonal: DiagonalValues
    mask: MaskValues
    input_projection: ProjectionValues
    output_projection: ProjectionValues


class _PackageRuntimeDefaults(AdaptiveRuntimeDefaults[RuntimeDefaultValues]):
    config = config
    package = __name__.rpartition(".")[0]
    field_type = RuntimeField
    runtime_values_type = RuntimeDefaultValues
    runtime_override_reader_type = RuntimeOverrideReader
    runtime_dimensions_values_type = RuntimeDimensionsValues
    runtime_dimensions_fields_type = RuntimeDimensionsFields
    stack_values_type = StackValues
    stack_fields_type = StackFields
    optional_stack_values_type = OptionalStackValues
    optional_stack_fields_type = OptionalStackFields
    control_values_type = ControlValues
    control_fields_type = ControlFields
    weight_values_type = WeightValues
    weight_fields_type = WeightFields
    bias_values_type = BiasValues
    bias_fields_type = BiasFields
    diagonal_values_type = DiagonalValues
    diagonal_fields_type = DiagonalFields
    mask_values_type = MaskValues
    mask_fields_type = MaskFields
    projection_values_type = ProjectionValues
    projection_fields_type = ProjectionFields
    generation_values_type = GenerationValues


class RuntimeDefaults:
    schema = _PackageRuntimeDefaults()
    validation = AdaptiveRuntimeValidation(schema, SETTINGS_VALIDATION)


def runtime_default_values(
    overrides: Mapping[str, object] | None,
) -> RuntimeDefaultValues:
    return RuntimeDefaults.schema.runtime_default_values(overrides)


def __getattr__(name: str) -> object:
    return RuntimeDefaults.schema.legacy_field_group(name, __name__)
