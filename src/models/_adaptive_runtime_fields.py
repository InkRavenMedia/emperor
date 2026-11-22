from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any, ClassVar, Generic, Never, Protocol, TypeGuard, TypeVar, cast

from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GroupingConfig,
    LowRankFactorSourceOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
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
from model_runtime.packages.settings_validation import BuildSettingsValidation

_ValueT = TypeVar("_ValueT")
_ValueT_co = TypeVar("_ValueT_co", covariant=True)
_EnumT = TypeVar("_EnumT", bound=Enum)
_ImplementationT = TypeVar("_ImplementationT")


def _is_float(value: object) -> TypeGuard[int | float]:
    return type(value) in (int, float)


def _is_implementation(
    value: object,
    base_type: type[_ImplementationT],
) -> TypeGuard[type[_ImplementationT]]:
    if base_type is ResidualConfig:
        return BuildSettingsValidation.is_selectable_residual(value)
    return isinstance(value, type) and issubclass(value, base_type)


def _type_name(expected: object) -> str:
    return str(expected).replace("<class '", "").replace("'>", "")


def _implementation_type_name(base_type: type[object]) -> str:
    return f"type[{base_type.__module__}.{base_type.__qualname__}]"


def _instance_type_name(instance_type: type[object]) -> str:
    return f"{instance_type.__module__}.{instance_type.__qualname__}"


def raise_type_error(package: str, key: str, value: object, expected: str) -> Never:
    raise TypeError(
        f"{package}: runtime key {key!r} has type {type(value).__name__}; "
        f"expected {expected}"
    )


class _ValueRule(Protocol[_ValueT_co]):
    __slots__ = ()

    @property
    def expected(self) -> str: ...

    def accepts(self, value: object) -> bool: ...

    def read(self, key: str, value: object, package: str) -> _ValueT_co: ...


class _CheckedRule(Generic[_ValueT]):
    __slots__ = ()

    def read(
        self: _ValueRule[object], key: str, value: object, package: str
    ) -> _ValueT:
        if self.accepts(value):
            return cast(_ValueT, value)
        raise_type_error(package, key, value, self.expected)


@dataclass(frozen=True, slots=True)
class _BooleanRule(_CheckedRule[bool]):
    expected: str = "bool"

    def accepts(self, value: object) -> bool:
        return type(value) is bool


@dataclass(frozen=True, slots=True)
class _IntegerRule(_CheckedRule[int]):
    expected: str = "int"

    def accepts(self, value: object) -> bool:
        return type(value) is int


@dataclass(frozen=True, slots=True)
class _FloatRule(_CheckedRule[float]):
    expected: str = "float"

    def accepts(self, value: object) -> bool:
        return _is_float(value)


@dataclass(frozen=True, slots=True)
class _EnumRule(_CheckedRule[_EnumT]):
    enum_type: type[_EnumT]

    @property
    def expected(self) -> str:
        return _type_name(self.enum_type)

    def accepts(self, value: object) -> bool:
        return isinstance(value, self.enum_type)


@dataclass(frozen=True, slots=True)
class _ImplementationRule(_CheckedRule[type[_ImplementationT]]):
    base_type: type[_ImplementationT]

    @property
    def expected(self) -> str:
        return _implementation_type_name(self.base_type)

    def accepts(self, value: object) -> bool:
        return _is_implementation(value, self.base_type)


@dataclass(frozen=True, slots=True)
class _InstanceRule(_CheckedRule[_ValueT]):
    instance_type: type[_ValueT]

    @property
    def expected(self) -> str:
        return _instance_type_name(self.instance_type)

    def accepts(self, value: object) -> bool:
        return isinstance(value, self.instance_type)


@dataclass(frozen=True, slots=True)
class _OptionalRule(Generic[_ValueT]):
    inner: _ValueRule[_ValueT]

    @property
    def expected(self) -> str:
        return f"{self.inner.expected} | None"

    def accepts(self, value: object) -> bool:
        return value is None or self.inner.accepts(value)

    def read(self, key: str, value: object, package: str) -> _ValueT | None:
        if value is None:
            return None
        return self.inner.read(key, value, package)


@dataclass(frozen=True, slots=True)
class _OptionalEnumRule(_CheckedRule[_EnumT | None]):
    enum_type: type[_EnumT]

    @property
    def expected(self) -> str:
        return f"{self.enum_type.__module__}.{self.enum_type.__qualname__} | None"

    def accepts(self, value: object) -> bool:
        return value is None or isinstance(value, self.enum_type)


@dataclass(frozen=True, slots=True)
class RuntimeField(Generic[_ValueT]):
    package: ClassVar[str]
    key: str
    default: _ValueT
    rule: _ValueRule[_ValueT]

    @property
    def expected(self) -> str:
        return self.rule.expected

    def accepts(self, value: object) -> bool:
        return self.rule.accepts(value)

    def read(self, value: object) -> _ValueT:
        return self.rule.read(self.key, value, self.package)


class _BoundaryField(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def expected(self) -> str: ...

    def accepts(self, value: object) -> bool: ...


class RuntimeFieldFactory:
    field_type: type[RuntimeField[Any]]
    boolean_rule = _BooleanRule()
    integer_rule = _IntegerRule()
    float_rule = _FloatRule()

    def _boolean_field(self, key: str, default: bool) -> RuntimeField[bool]:
        return self.field_type(key, default, self.boolean_rule)

    def _optional_boolean_field(
        self,
        key: str,
        default: bool | None,
    ) -> RuntimeField[bool | None]:
        return self.field_type(key, default, _OptionalRule(self.boolean_rule))

    def _integer_field(self, key: str, default: int) -> RuntimeField[int]:
        return self.field_type(key, default, self.integer_rule)

    def _optional_integer_field(
        self,
        key: str,
        default: int | None,
    ) -> RuntimeField[int | None]:
        return self.field_type(key, default, _OptionalRule(self.integer_rule))

    def _float_field(self, key: str, default: float) -> RuntimeField[float]:
        return self.field_type(key, default, self.float_rule)

    def _optional_float_field(
        self,
        key: str,
        default: float | None,
    ) -> RuntimeField[float | None]:
        return self.field_type(key, default, _OptionalRule(self.float_rule))

    def _enum_field(
        self,
        key: str,
        default: _EnumT,
        enum_type: type[_EnumT],
    ) -> RuntimeField[_EnumT]:
        return self.field_type(key, default, _EnumRule(enum_type))

    def _optional_enum_field(
        self,
        key: str,
        default: _EnumT | None,
        enum_type: type[_EnumT],
    ) -> RuntimeField[_EnumT | None]:
        return self.field_type(key, default, _OptionalEnumRule(enum_type))

    def _implementation_field(
        self,
        key: str,
        default: type[_ImplementationT],
        base_type: type[_ImplementationT],
    ) -> RuntimeField[type[_ImplementationT]]:
        return self.field_type(key, default, _ImplementationRule(base_type))

    def _optional_implementation_field(
        self,
        key: str,
        default: type[_ImplementationT] | None,
        base_type: type[_ImplementationT],
    ) -> RuntimeField[type[_ImplementationT] | None]:
        return self.field_type(
            key, default, _OptionalRule(_ImplementationRule(base_type))
        )

    def _optional_instance_field(
        self,
        key: str,
        default: _ValueT | None,
        instance_type: type[_ValueT],
    ) -> RuntimeField[_ValueT | None]:
        return self.field_type(
            key, default, _OptionalRule(_InstanceRule(instance_type))
        )


@dataclass(slots=True)
class RuntimeOverrideReader:
    values: Mapping[str, object]
    fields: dict[str, _BoundaryField] = field(
        default_factory=dict[str, _BoundaryField], init=False
    )

    def read(self, runtime_field: RuntimeField[_ValueT]) -> _ValueT:
        self.fields[runtime_field.key] = runtime_field
        value = self.values.get(runtime_field.key, runtime_field.default)
        return runtime_field.read(value)


@dataclass(frozen=True, slots=True)
class RuntimeDimensionsValues:
    batch_size: int
    learning_rate: float
    input_dim: int
    hidden_dim: int
    output_dim: int


@dataclass(frozen=True, slots=True)
class RuntimeDimensionsFields:
    batch_size: RuntimeField[int]
    learning_rate: RuntimeField[float]
    input_dim: RuntimeField[int]
    hidden_dim: RuntimeField[int]
    output_dim: RuntimeField[int]


@dataclass(frozen=True, slots=True)
class StackValues:
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
    residual_model_flag: bool
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float
    bias_flag: bool


@dataclass(frozen=True, slots=True)
class StackFields:
    hidden_dim: RuntimeField[int]
    num_layers: RuntimeField[int]
    last_layer_bias_option: RuntimeField[LastLayerBiasOptions]
    apply_output_postprocessing_flag: RuntimeField[bool]
    activation: RuntimeField[ActivationOptions]
    layer_norm_position: RuntimeField[LayerNormPositionOptions]
    normalization: RuntimeField[NormalizationOptions] = field(
        default=cast(RuntimeField[NormalizationOptions], NormalizationOptions.RMS_NORM),
        kw_only=True,
    )
    residual_connection_option: RuntimeField[type[ResidualConfig] | None]
    residual_model_flag: RuntimeField[bool]
    residual_block_size: RuntimeField[int | None]
    residual_rms_norm_epsilon: RuntimeField[float | None]
    dropout_probability: RuntimeField[float]
    bias_flag: RuntimeField[bool]


@dataclass(frozen=True, slots=True)
class OptionalStackValues:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    normalization: NormalizationOptions | None = field(default=None, kw_only=True)
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float | None
    bias_flag: bool | None


@dataclass(frozen=True, slots=True)
class OptionalStackFields:
    config: ClassVar[ModuleType]
    field_type: ClassVar[type[RuntimeField[Any]]]
    independent_flag: RuntimeField[bool]
    hidden_dim: RuntimeField[int | None]
    num_layers: RuntimeField[int | None]
    last_layer_bias_option: RuntimeField[LastLayerBiasOptions | None]
    apply_output_postprocessing_flag: RuntimeField[bool | None]
    activation: RuntimeField[ActivationOptions | None]
    layer_norm_position: RuntimeField[LayerNormPositionOptions | None]
    normalization: RuntimeField[NormalizationOptions | None] = field(
        default=cast(RuntimeField[NormalizationOptions | None], None), kw_only=True
    )
    residual_connection_option: RuntimeField[type[ResidualConfig] | None]
    residual_model_flag: RuntimeField[bool]
    residual_block_size: RuntimeField[int | None]
    residual_rms_norm_epsilon: RuntimeField[float | None]
    dropout_probability: RuntimeField[float | None]
    bias_flag: RuntimeField[bool | None]

    @classmethod
    def for_stack(cls, prefix: str) -> OptionalStackFields:
        def setting(suffix: str, rule: _ValueRule[_ValueT]) -> RuntimeField[_ValueT]:
            key = f"{prefix}_{suffix}"
            return cls.field_type(key, getattr(cls.config, key.upper()), rule)

        return cls(
            independent_flag=setting(
                "independent_flag", RuntimeFieldFactory.boolean_rule
            ),
            hidden_dim=setting(
                "hidden_dim", _OptionalRule(RuntimeFieldFactory.integer_rule)
            ),
            layer_norm_position=setting(
                "layer_norm_position", _OptionalEnumRule(LayerNormPositionOptions)
            ),
            normalization=setting(
                "normalization", _OptionalEnumRule(NormalizationOptions)
            ),
            num_layers=setting(
                "num_layers", _OptionalRule(RuntimeFieldFactory.integer_rule)
            ),
            activation=setting("activation", _OptionalEnumRule(ActivationOptions)),
            residual_connection_option=setting(
                "residual_connection_option",
                _OptionalRule(_ImplementationRule(ResidualConfig)),
            ),
            residual_block_size=setting(
                "residual_block_size", _OptionalRule(RuntimeFieldFactory.integer_rule)
            ),
            residual_rms_norm_epsilon=setting(
                "residual_rms_norm_epsilon",
                _OptionalRule(RuntimeFieldFactory.float_rule),
            ),
            residual_model_flag=setting(
                "residual_model_flag", RuntimeFieldFactory.boolean_rule
            ),
            dropout_probability=setting(
                "dropout_probability", _OptionalRule(RuntimeFieldFactory.float_rule)
            ),
            last_layer_bias_option=setting(
                "last_layer_bias_option", _OptionalEnumRule(LastLayerBiasOptions)
            ),
            apply_output_postprocessing_flag=setting(
                "apply_output_postprocessing_flag",
                _OptionalRule(RuntimeFieldFactory.boolean_rule),
            ),
            bias_flag=setting(
                "bias_flag", _OptionalRule(RuntimeFieldFactory.boolean_rule)
            ),
        )


@dataclass(frozen=True, slots=True)
class ControlValues:
    stack_gate_flag: bool
    gate_option: LayerGateOptions | None
    gate_activation: ActivationOptions | None
    shared_gate_config: GateConfig | None
    stack_halting_flag: bool
    halting_option: type[HaltingConfig]
    halting_threshold: float
    halting_dropout: float
    halting_hidden_state_mode: HaltingHiddenStateModeOptions
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    memory_learning_rate: float | None
    memory_num_inner_steps: int | None
    recurrent_flag: bool
    recurrent_max_steps: int
    recurrent_initial_iterations: int
    recurrent_gradient_transition_count: int | None
    recurrent_no_gradient_transition_count: int | None
    recurrent_iteration_increment: int
    recurrent_forward_calls_before_iteration_increment: int
    recurrent_smooth_iteration_growth_flag: bool
    recurrent_layer_norm_position: LayerNormPositionOptions
    recurrent_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    recurrent_stack_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_stack_halting_flag: bool
    recurrent_halting_option: type[HaltingConfig]
    recurrent_halting_threshold: float
    recurrent_halting_dropout: float
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions


@dataclass(frozen=True, slots=True)
class ControlFields:
    stack_gate_flag: RuntimeField[bool]
    gate_option: RuntimeField[LayerGateOptions | None]
    gate_activation: RuntimeField[ActivationOptions | None]
    shared_gate_config: RuntimeField[GateConfig | None]
    stack_halting_flag: RuntimeField[bool]
    halting_option: RuntimeField[type[HaltingConfig]]
    halting_threshold: RuntimeField[float]
    halting_dropout: RuntimeField[float]
    halting_hidden_state_mode: RuntimeField[HaltingHiddenStateModeOptions]
    memory_flag: RuntimeField[bool]
    memory_option: RuntimeField[type[DynamicMemoryConfig]]
    memory_position_option: RuntimeField[MemoryPositionOptions]
    memory_learning_rate: RuntimeField[float | None]
    memory_num_inner_steps: RuntimeField[int | None]
    recurrent_flag: RuntimeField[bool]
    recurrent_max_steps: RuntimeField[int]
    recurrent_initial_iterations: RuntimeField[int]
    recurrent_gradient_transition_count: RuntimeField[int | None]
    recurrent_no_gradient_transition_count: RuntimeField[int | None]
    recurrent_iteration_increment: RuntimeField[int]
    recurrent_forward_calls_before_iteration_increment: RuntimeField[int]
    recurrent_smooth_iteration_growth_flag: RuntimeField[bool]
    recurrent_layer_norm_position: RuntimeField[LayerNormPositionOptions]
    recurrent_normalization: RuntimeField[NormalizationOptions] = field(
        default=cast(
            RuntimeField[NormalizationOptions], NormalizationOptions.LAYER_NORM
        ),
        kw_only=True,
    )
    recurrent_stack_gate_flag: RuntimeField[bool]
    recurrent_gate_option: RuntimeField[LayerGateOptions | None]
    recurrent_gate_activation: RuntimeField[ActivationOptions | None]
    recurrent_stack_halting_flag: RuntimeField[bool]
    recurrent_halting_option: RuntimeField[type[HaltingConfig]]
    recurrent_halting_threshold: RuntimeField[float]
    recurrent_halting_dropout: RuntimeField[float]
    recurrent_halting_hidden_state_mode: RuntimeField[HaltingHiddenStateModeOptions]


@dataclass(frozen=True, slots=True)
class WeightValues:
    enabled: bool
    option: type[DynamicWeightConfig] | None
    generator_depth: int
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    normalization_option: WeightNormalizationOptions
    normalization_learnable_scale_flag: bool
    normalization_position_option: WeightNormalizationPositionOptions
    bank_expansion_factor: BankExpansionFactorOptions


@dataclass(frozen=True, slots=True)
class WeightFields:
    enabled: RuntimeField[bool]
    option: RuntimeField[type[DynamicWeightConfig] | None]
    generator_depth: RuntimeField[int]
    decay_schedule: RuntimeField[WeightDecayScheduleOptions]
    decay_rate: RuntimeField[float]
    decay_warmup_batches: RuntimeField[int]
    normalization_option: RuntimeField[WeightNormalizationOptions]
    normalization_learnable_scale_flag: RuntimeField[bool]
    normalization_position_option: RuntimeField[WeightNormalizationPositionOptions]
    bank_expansion_factor: RuntimeField[BankExpansionFactorOptions]


@dataclass(frozen=True, slots=True)
class BiasValues:
    enabled: bool
    option: type[DynamicBiasConfig] | None
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions


@dataclass(frozen=True, slots=True)
class BiasFields:
    enabled: RuntimeField[bool]
    option: RuntimeField[type[DynamicBiasConfig] | None]
    decay_schedule: RuntimeField[WeightDecayScheduleOptions]
    decay_rate: RuntimeField[float]
    decay_warmup_batches: RuntimeField[int]
    bank_expansion_factor: RuntimeField[BankExpansionFactorOptions]


@dataclass(frozen=True, slots=True)
class DiagonalValues:
    enabled: bool
    option: type[DynamicDiagonalConfig] | None


@dataclass(frozen=True, slots=True)
class DiagonalFields:
    enabled: RuntimeField[bool]
    option: RuntimeField[type[DynamicDiagonalConfig] | None]


@dataclass(frozen=True, slots=True)
class MaskValues:
    enabled: bool
    row_mask_option: type[AxisMaskConfig] | None
    threshold: float
    floor: float
    transition_width: float
    surrogate_scale: float
    dimension_option: MaskDimensionOptions


@dataclass(frozen=True, slots=True)
class MaskFields:
    enabled: RuntimeField[bool]
    row_mask_option: RuntimeField[type[AxisMaskConfig] | None]
    threshold: RuntimeField[float]
    floor: RuntimeField[float]
    transition_width: RuntimeField[float]
    surrogate_scale: RuntimeField[float]
    dimension_option: RuntimeField[MaskDimensionOptions]


@dataclass(frozen=True, slots=True)
class ProjectionValues:
    weight_option: type[DynamicWeightConfig] | None
    generator_depth: int
    weight_decay_schedule: WeightDecayScheduleOptions
    weight_decay_rate: float
    weight_decay_warmup_batches: int
    weight_normalization_option: WeightNormalizationOptions
    weight_normalization_learnable_scale_flag: bool
    weight_normalization_position_option: WeightNormalizationPositionOptions
    weight_bank_expansion_factor: BankExpansionFactorOptions
    bias_option: type[DynamicBiasConfig] | None
    bias_decay_schedule: WeightDecayScheduleOptions
    bias_decay_rate: float
    bias_decay_warmup_batches: int
    bias_bank_expansion_factor: BankExpansionFactorOptions
    diagonal_option: type[DynamicDiagonalConfig] | None
    row_mask_option: type[AxisMaskConfig] | None
    mask_threshold: float
    mask_floor: float
    mask_transition_width: float
    mask_surrogate_scale: float
    mask_dimension_option: MaskDimensionOptions


@dataclass(frozen=True, slots=True)
class ProjectionFields:
    weight_option: RuntimeField[type[DynamicWeightConfig] | None]
    generator_depth: RuntimeField[int]
    weight_decay_schedule: RuntimeField[WeightDecayScheduleOptions]
    weight_decay_rate: RuntimeField[float]
    weight_decay_warmup_batches: RuntimeField[int]
    weight_normalization_option: RuntimeField[WeightNormalizationOptions]
    weight_normalization_learnable_scale_flag: RuntimeField[bool]
    weight_normalization_position_option: RuntimeField[
        WeightNormalizationPositionOptions
    ]
    weight_bank_expansion_factor: RuntimeField[BankExpansionFactorOptions]
    bias_option: RuntimeField[type[DynamicBiasConfig] | None]
    bias_decay_schedule: RuntimeField[WeightDecayScheduleOptions]
    bias_decay_rate: RuntimeField[float]
    bias_decay_warmup_batches: RuntimeField[int]
    bias_bank_expansion_factor: RuntimeField[BankExpansionFactorOptions]
    diagonal_option: RuntimeField[type[DynamicDiagonalConfig] | None]
    row_mask_option: RuntimeField[type[AxisMaskConfig] | None]
    mask_threshold: RuntimeField[float]
    mask_floor: RuntimeField[float]
    mask_transition_width: RuntimeField[float]
    mask_surrogate_scale: RuntimeField[float]
    mask_dimension_option: RuntimeField[MaskDimensionOptions]


@dataclass(frozen=True, slots=True)
class GenerationValues:
    weight_input_factor_source: LowRankFactorSourceOptions | None
    weight_output_factor_source: LowRankFactorSourceOptions | None
    weight_mixture_num_experts: int | None
    bias_mixture_num_experts: int | None
    weight_mixture_top_k: int | None
    bias_mixture_top_k: int | None
    weight_mixture_normalize_probabilities_flag: bool | None
    bias_mixture_normalize_probabilities_flag: bool | None
    weight_input_factor_generator_stack: OptionalStackValues
    weight_output_factor_generator_stack: OptionalStackValues
    weight_coefficient_generator_stack: OptionalStackValues
    weight_mixture_router_generator_stack: OptionalStackValues
    bias_mixture_router_generator_stack: OptionalStackValues


@dataclass(frozen=True, slots=True)
class RuntimeDefaultValues:
    residual_coefficient_activation: ActivationOptions
    grouping: GroupingConfig | None
    input_grouping: GroupingConfig | None
    output_grouping: GroupingConfig | None
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
