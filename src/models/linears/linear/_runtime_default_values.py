from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeGuard, TypeVar, cast

import models.linears.linear.config as config
from emperor.halting import HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from models.linears.linear import SETTINGS_VALIDATION
from models.linears.linear._residual import (
    ResidualStackOptions,
    ResidualStackSource,
    inherit_if_unset,
    resolve_residual_stack_options,
)
from models.linears.linear.runtime_options import (
    ControllerStackOptions,
    GateOptions,
    HaltingOptions,
    MainStackOptions,
    MemoryOptions,
    RecurrenceOptions,
)

_PACKAGE_NAME = "models.linears.linear"
_T = TypeVar("_T")
_EnumT = TypeVar("_EnumT", bound=Enum)
_CONFIG_DEFAULTS: dict[str, object] = {
    name.lower(): value for name, value in vars(config).items() if name.isupper()
}
_USE_CONFIG_DEFAULT = object()


@dataclass(frozen=True, slots=True)
class _ValueType(Generic[_T]):
    name: str
    accepts: Callable[[object], bool]

    def validate(self, key: str, value: object) -> _T:
        if self.accepts(value):
            return cast(_T, value)
        raise TypeError(
            f"{_PACKAGE_NAME}: {key!r} has type {type(value).__name__}; "
            f"expected {self.name}"
        )


def _is_implementation(value: object, base_type: type[_T]) -> TypeGuard[type[_T]]:
    return isinstance(value, type) and issubclass(value, base_type)


def _enum_type(enum_type: type[_EnumT]) -> _ValueType[_EnumT]:
    return _ValueType(enum_type.__name__, lambda value: isinstance(value, enum_type))


def _exact_type(expected: type[_T]) -> _ValueType[_T]:
    return _ValueType(expected.__name__, lambda value: type(value) is expected)


_BOOLEAN = _exact_type(bool)
_INTEGER = _exact_type(int)
_FLOAT = _exact_type(float)
_RESIDUAL: _ValueType[type[ResidualConfig]] = _ValueType(
    "concrete type[ResidualConfig]",
    ResidualConfig.is_concrete_implementation,
)


@dataclass(slots=True)
class RuntimeOverrideReader:
    supplied: Mapping[str, object]
    remaining: dict[str, object] = field(init=False)
    accepted: set[str] = field(default_factory=set[str], init=False)

    def __post_init__(self) -> None:
        self.remaining = dict(self.supplied)
        _validate_override_keys(self.remaining)

    def reject_unknown(self, accepted: frozenset[str]) -> None:
        for key in self.remaining:
            if key not in accepted:
                accepted_text = ", ".join(sorted(accepted))
                raise ValueError(
                    f"{_PACKAGE_NAME}: unknown runtime override {key!r}; "
                    f"accepted keys: {accepted_text}"
                )

    def boolean(self, key: str) -> bool:
        return self._required(key, _BOOLEAN)

    def optional_boolean(self, key: str) -> bool | None:
        return self._optional(key, _BOOLEAN)

    def integer(self, key: str) -> int:
        return self._required(key, _INTEGER)

    def optional_integer(self, key: str) -> int | None:
        return self._optional(key, _INTEGER)

    def floating(self, key: str) -> float:
        return self._required(key, _FLOAT)

    def optional_floating(self, key: str) -> float | None:
        return self._optional(key, _FLOAT)

    def enum(
        self,
        key: str,
        enum_type: type[_EnumT],
        *,
        default: object = _USE_CONFIG_DEFAULT,
    ) -> _EnumT:
        return self._required(key, _enum_type(enum_type), default=default)

    def optional_enum(self, key: str, enum_type: type[_EnumT]) -> _EnumT | None:
        return self._optional(key, _enum_type(enum_type))

    def implementation(self, key: str, base_type: type[_T]) -> type[_T]:
        implementation_type: _ValueType[type[_T]] = _ValueType(
            f"type[{base_type.__name__}]",
            lambda value: _is_implementation(value, base_type),
        )
        return self._required(key, implementation_type)

    def optional_residual(self, key: str) -> type[ResidualConfig] | None:
        return self._optional(key, _RESIDUAL)

    def _required(
        self,
        key: str,
        value_type: _ValueType[_T],
        *,
        default: object = _USE_CONFIG_DEFAULT,
    ) -> _T:
        return value_type.validate(key, self._take(key, default))

    def _optional(self, key: str, value_type: _ValueType[_T]) -> _T | None:
        value = self._take(key)
        return None if value is None else value_type.validate(key, value)

    def _take(self, key: str, default: object = _USE_CONFIG_DEFAULT) -> object:
        self.accepted.add(key)
        if default is _USE_CONFIG_DEFAULT:
            default = _CONFIG_DEFAULTS[key]
        return self.remaining.pop(key, default)


def _validate_override_keys(keys: Iterable[object]) -> None:
    for key in keys:
        if not isinstance(key, str):
            raise TypeError(
                f"{_PACKAGE_NAME}: runtime override keys must be str; got "
                f"{type(key).__name__}"
            )


@dataclass(frozen=True, slots=True)
class BuildDimensions:
    batch_size: int
    learning_rate: float
    input_dim: int
    hidden_dim: int
    output_dim: int


@dataclass(frozen=True, slots=True)
class StackFields:
    independent_flag: str
    hidden_dim: str
    num_layers: str
    last_layer_bias_option: str
    apply_output_postprocessing_flag: str
    activation: str
    layer_norm_position: str
    normalization: str = field(kw_only=True)
    residual_connection_option: str
    residual_model_flag: str
    residual_block_size: str
    residual_rms_norm_epsilon: str
    dropout_probability: str
    bias_flag: str

    @classmethod
    def for_stack(cls, prefix: str) -> StackFields:
        return cls(
            independent_flag=f"{prefix}_independent_flag",
            hidden_dim=f"{prefix}_hidden_dim",
            num_layers=f"{prefix}_num_layers",
            last_layer_bias_option=f"{prefix}_last_layer_bias_option",
            apply_output_postprocessing_flag=f"{prefix}_apply_output_postprocessing_flag",
            activation=f"{prefix}_activation",
            layer_norm_position=f"{prefix}_layer_norm_position",
            normalization=f"{prefix}_normalization",
            residual_connection_option=f"{prefix}_residual_connection_option",
            residual_block_size=f"{prefix}_residual_block_size",
            residual_rms_norm_epsilon=f"{prefix}_residual_rms_norm_epsilon",
            residual_model_flag=f"{prefix}_residual_model_flag",
            dropout_probability=f"{prefix}_dropout_probability",
            bias_flag=f"{prefix}_bias_flag",
        )


_RESIDUAL_STACK_FIELDS = StackFields.for_stack("residual_stack")
GATE_STACK_FIELDS = StackFields.for_stack("gate_stack")
HALTING_STACK_FIELDS = StackFields.for_stack("halting_stack")
MEMORY_STACK_FIELDS = StackFields.for_stack("memory_stack")
RECURRENT_GATE_STACK_FIELDS = StackFields.for_stack("recurrent_gate_stack")
RECURRENT_HALTING_STACK_FIELDS = StackFields.for_stack("recurrent_halting_stack")


@dataclass(frozen=True, slots=True)
class _ControllerStackOverrides:
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

    def resolve(self, defaults: ControllerStackOptions) -> ControllerStackOptions:
        if not self.independent_flag:
            return defaults
        return ControllerStackOptions(
            hidden_dim=inherit_if_unset(self.hidden_dim, defaults.hidden_dim),
            num_layers=inherit_if_unset(self.num_layers, defaults.num_layers),
            activation=inherit_if_unset(self.activation, defaults.activation),
            layer_norm_position=inherit_if_unset(
                self.layer_norm_position, defaults.layer_norm_position
            ),
            normalization=inherit_if_unset(self.normalization, defaults.normalization),
            residual_connection_option=inherit_if_unset(
                self.residual_connection_option, defaults.residual_connection_option
            ),
            residual_model_flag=self.residual_model_flag,
            residual_block_size=self.residual_block_size,
            residual_rms_norm_epsilon=self.residual_rms_norm_epsilon,
            dropout_probability=inherit_if_unset(
                self.dropout_probability, defaults.dropout_probability
            ),
            last_layer_bias_option=inherit_if_unset(
                self.last_layer_bias_option, defaults.last_layer_bias_option
            ),
            apply_output_postprocessing_flag=inherit_if_unset(
                self.apply_output_postprocessing_flag,
                defaults.apply_output_postprocessing_flag,
            ),
            bias_flag=inherit_if_unset(self.bias_flag, defaults.bias_flag),
        )


def read_dimensions(reader: RuntimeOverrideReader) -> BuildDimensions:
    dimensions = BuildDimensions(
        batch_size=reader.integer("batch_size"),
        learning_rate=reader.floating("learning_rate"),
        input_dim=reader.integer("input_dim"),
        hidden_dim=reader.integer("hidden_dim"),
        output_dim=reader.integer("output_dim"),
    )
    SETTINGS_VALIDATION.dimensions(dimensions)
    return dimensions


def read_main_stack(reader: RuntimeOverrideReader) -> MainStackOptions:
    num_layers = reader.integer("stack_num_layers")
    dropout = reader.floating("stack_dropout_probability")
    _require_positive("stack_num_layers", num_layers)
    _require_probability("stack_dropout_probability", dropout)
    return MainStackOptions(
        bias_flag=reader.boolean("stack_bias_flag"),
        layer_norm_position=reader.enum(
            "layer_norm_position", LayerNormPositionOptions
        ),
        normalization=reader.enum("normalization", NormalizationOptions),
        num_layers=num_layers,
        activation=reader.enum("stack_activation", ActivationOptions),
        residual_connection_option=reader.optional_residual(
            "stack_residual_connection_option"
        ),
        residual_block_size=reader.optional_integer("stack_residual_block_size"),
        residual_rms_norm_epsilon=reader.optional_floating(
            "stack_residual_rms_norm_epsilon"
        ),
        residual_model_flag=reader.boolean("stack_residual_model_flag"),
        dropout_probability=dropout,
        last_layer_bias_option=reader.enum(
            "stack_last_layer_bias_option", LastLayerBiasOptions
        ),
        apply_output_postprocessing_flag=reader.boolean(
            "stack_apply_output_postprocessing_flag"
        ),
    )


def read_submodule_stack(reader: RuntimeOverrideReader) -> ControllerStackOptions:
    hidden_dim = reader.integer("submodule_stack_hidden_dim")
    num_layers = reader.integer("submodule_stack_num_layers")
    dropout = reader.floating("submodule_stack_dropout_probability")
    _require_positive("submodule_stack_hidden_dim", hidden_dim)
    _require_positive("submodule_stack_num_layers", num_layers)
    _require_probability("submodule_stack_dropout_probability", dropout)
    return ControllerStackOptions(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=reader.enum(
            "submodule_stack_last_layer_bias_option", LastLayerBiasOptions
        ),
        apply_output_postprocessing_flag=reader.boolean(
            "submodule_stack_apply_output_postprocessing_flag"
        ),
        activation=reader.enum("submodule_stack_activation", ActivationOptions),
        layer_norm_position=reader.enum(
            "submodule_stack_layer_norm_position", LayerNormPositionOptions
        ),
        normalization=reader.enum(
            "submodule_stack_normalization", NormalizationOptions
        ),
        residual_connection_option=reader.optional_residual(
            "submodule_stack_residual_connection_option"
        ),
        residual_block_size=reader.optional_integer(
            "submodule_stack_residual_block_size"
        ),
        residual_rms_norm_epsilon=reader.optional_floating(
            "submodule_stack_residual_rms_norm_epsilon"
        ),
        residual_model_flag=reader.boolean("submodule_stack_residual_model_flag"),
        dropout_probability=dropout,
        bias_flag=reader.boolean("submodule_stack_bias_flag"),
    )


def read_residual_stack(
    reader: RuntimeOverrideReader, defaults: ControllerStackOptions
) -> ResidualStackOptions:
    fields = _RESIDUAL_STACK_FIELDS
    hidden_dim = reader.optional_integer(fields.hidden_dim)
    num_layers = reader.optional_integer(fields.num_layers)
    dropout = reader.optional_floating(fields.dropout_probability)
    _validate_stack_measures(fields, hidden_dim, num_layers, dropout)
    source = ResidualStackSource(
        independent_flag=reader.boolean(fields.independent_flag),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=reader.optional_enum(fields.activation, ActivationOptions),
        layer_norm_position=reader.optional_enum(
            fields.layer_norm_position, LayerNormPositionOptions
        ),
        normalization=reader.optional_enum(fields.normalization, NormalizationOptions),
        residual_connection_option=reader.optional_residual(
            fields.residual_connection_option
        ),
        residual_block_size=reader.optional_integer(fields.residual_block_size),
        residual_rms_norm_epsilon=reader.optional_floating(
            fields.residual_rms_norm_epsilon
        ),
        residual_model_flag=reader.boolean(fields.residual_model_flag),
        dropout_probability=dropout,
        last_layer_bias_option=reader.optional_enum(
            fields.last_layer_bias_option, LastLayerBiasOptions
        ),
        apply_output_postprocessing_flag=reader.optional_boolean(
            fields.apply_output_postprocessing_flag
        ),
        bias_flag=reader.optional_boolean(fields.bias_flag),
        coefficient_activation=reader.enum(
            "residual_coefficient_activation",
            ActivationOptions,
            default=config.RESIDUAL_COEFFICIENT_ACTIVATION,
        ),
    )
    return resolve_residual_stack_options(source, defaults)


def read_controller_stack(
    reader: RuntimeOverrideReader, fields: StackFields, defaults: ControllerStackOptions
) -> ControllerStackOptions:
    overrides = _ControllerStackOverrides(
        independent_flag=reader.boolean(fields.independent_flag),
        hidden_dim=reader.optional_integer(fields.hidden_dim),
        num_layers=reader.optional_integer(fields.num_layers),
        last_layer_bias_option=reader.optional_enum(
            fields.last_layer_bias_option, LastLayerBiasOptions
        ),
        apply_output_postprocessing_flag=reader.optional_boolean(
            fields.apply_output_postprocessing_flag
        ),
        activation=reader.optional_enum(fields.activation, ActivationOptions),
        layer_norm_position=reader.optional_enum(
            fields.layer_norm_position, LayerNormPositionOptions
        ),
        normalization=reader.optional_enum(fields.normalization, NormalizationOptions),
        residual_connection_option=reader.optional_residual(
            fields.residual_connection_option
        ),
        residual_block_size=reader.optional_integer(fields.residual_block_size),
        residual_rms_norm_epsilon=reader.optional_floating(
            fields.residual_rms_norm_epsilon
        ),
        residual_model_flag=reader.boolean(fields.residual_model_flag),
        dropout_probability=reader.optional_floating(fields.dropout_probability),
        bias_flag=reader.optional_boolean(fields.bias_flag),
    )
    _validate_stack_measures(
        fields,
        overrides.hidden_dim,
        overrides.num_layers,
        overrides.dropout_probability,
    )
    return overrides.resolve(defaults)


def read_gate_options(
    reader: RuntimeOverrideReader, stack: ControllerStackOptions
) -> GateOptions:
    enabled, option = _read_gate_selection(reader)
    return GateOptions(
        enabled=enabled,
        option=option,
        activation=reader.optional_enum("gate_activation", ActivationOptions),
        stack=stack,
    )


def read_halting_options(
    reader: RuntimeOverrideReader, stack: ControllerStackOptions
) -> HaltingOptions:
    threshold, dropout = _read_halting_measures(reader)
    return _read_halting_controls(reader, stack, threshold=threshold, dropout=dropout)


def read_memory_options(
    reader: RuntimeOverrideReader, stack: ControllerStackOptions
) -> MemoryOptions:
    learning_rate = reader.optional_floating("memory_test_time_training_learning_rate")
    inner_steps = reader.optional_integer("memory_test_time_training_num_inner_steps")
    _require_positive("memory_test_time_training_learning_rate", learning_rate)
    _require_positive("memory_test_time_training_num_inner_steps", inner_steps)
    return MemoryOptions(
        enabled=reader.boolean("memory_flag"),
        implementation=reader.implementation("memory_option", DynamicMemoryConfig),
        position=reader.enum("memory_position_option", MemoryPositionOptions),
        test_time_training_learning_rate=learning_rate,
        test_time_training_num_inner_steps=inner_steps,
        stack=stack,
    )


def read_recurrence_options(
    reader: RuntimeOverrideReader,
    gate_stack: ControllerStackOptions,
    halting_stack: ControllerStackOptions,
) -> RecurrenceOptions:
    max_steps = reader.integer("recurrent_max_steps")
    _require_positive("recurrent_max_steps", max_steps)
    gate_enabled, gate_option = _read_gate_selection(reader, prefix="recurrent_")
    threshold, dropout = _read_halting_measures(reader, prefix="recurrent_")
    return RecurrenceOptions(
        enabled=reader.boolean("recurrent_flag"),
        max_steps=max_steps,
        initial_iterations=reader.optional_integer("recurrent_initial_iterations"),
        gradient_transition_count=reader.optional_integer(
            "recurrent_gradient_transition_count"
        ),
        no_gradient_transition_count=reader.optional_integer(
            "recurrent_no_gradient_transition_count"
        ),
        iteration_increment=reader.integer("recurrent_iteration_increment"),
        forward_calls_before_iteration_increment=reader.integer(
            "recurrent_forward_calls_before_iteration_increment"
        ),
        smooth_iteration_growth_flag=reader.boolean(
            "recurrent_smooth_iteration_growth_flag"
        ),
        layer_norm_position=reader.enum(
            "recurrent_layer_norm_position", LayerNormPositionOptions
        ),
        normalization=reader.enum("recurrent_normalization", NormalizationOptions),
        gate=GateOptions(
            enabled=gate_enabled,
            option=gate_option,
            activation=reader.optional_enum(
                "recurrent_gate_activation", ActivationOptions
            ),
            stack=gate_stack,
        ),
        halting=_read_halting_controls(
            reader,
            halting_stack,
            prefix="recurrent_",
            threshold=threshold,
            dropout=dropout,
        ),
    )


def _read_gate_selection(
    reader: RuntimeOverrideReader, *, prefix: str = ""
) -> tuple[bool, LayerGateOptions | None]:
    enabled_key = f"{prefix}stack_gate_flag"
    option_key = f"{prefix}gate_option"
    enabled = reader.boolean(enabled_key)
    option = reader.optional_enum(option_key, LayerGateOptions)
    SETTINGS_VALIDATION.gate(enabled, option, prefix=prefix)
    return enabled, option


def _read_halting_measures(
    reader: RuntimeOverrideReader, *, prefix: str = ""
) -> tuple[float, float]:
    threshold_key = f"{prefix}halting_threshold"
    dropout_key = f"{prefix}halting_dropout"
    threshold = reader.floating(threshold_key)
    dropout = reader.floating(dropout_key)
    SETTINGS_VALIDATION.halting_threshold(threshold_key, threshold)
    _require_probability(dropout_key, dropout)
    return threshold, dropout


def _read_halting_controls(
    reader: RuntimeOverrideReader,
    stack: ControllerStackOptions,
    *,
    threshold: float,
    dropout: float,
    prefix: str = "",
) -> HaltingOptions:
    return HaltingOptions(
        enabled=reader.boolean(f"{prefix}stack_halting_flag"),
        threshold=threshold,
        dropout_probability=dropout,
        hidden_state_mode=reader.enum(
            f"{prefix}halting_hidden_state_mode", HaltingHiddenStateModeOptions
        ),
        stack=stack,
    )


def _validate_stack_measures(
    fields: StackFields,
    hidden_dim: int | None,
    num_layers: int | None,
    dropout: float | None,
) -> None:
    _require_positive(fields.hidden_dim, hidden_dim)
    _require_positive(fields.num_layers, num_layers)
    _require_probability(fields.dropout_probability, dropout)


def _require_positive(key: str, value: int | float | None) -> None:
    if value is not None:
        SETTINGS_VALIDATION.positive(key, value)


def _require_probability(key: str, value: float | None) -> None:
    if value is not None:
        SETTINGS_VALIDATION.dropout(key, value)
