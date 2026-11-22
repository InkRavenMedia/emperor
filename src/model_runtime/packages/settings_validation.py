from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from inspect import isabstract
from typing import TYPE_CHECKING, Protocol, TypeGuard, get_args, get_type_hints

if TYPE_CHECKING:
    from emperor.layers import ResidualConfig


class _BuildDimensions(Protocol):
    @property
    def batch_size(self) -> int: ...

    @property
    def learning_rate(self) -> float: ...

    @property
    def input_dim(self) -> int: ...

    @property
    def hidden_dim(self) -> int: ...

    @property
    def output_dim(self) -> int: ...


class _LayerStack(Protocol):
    @property
    def num_layers(self) -> int: ...

    @property
    def dropout_probability(self) -> float: ...

    @property
    def residual_connection_option(self) -> "type[ResidualConfig] | None": ...


@dataclass(frozen=True, slots=True)
class BuildSettingsValidation:
    package: str
    allow_integer_floats: bool
    allow_full_dropout: bool
    invalid_value_format: str = "{key!r} {requirement}; got {value!r}"

    def dimensions(self, dimensions: _BuildDimensions) -> None:
        from model_runtime.packages.configuration import require_finite_learning_rate

        require_finite_learning_rate("LEARNING_RATE", dimensions.learning_rate)
        self.positive_integer("batch_size", dimensions.batch_size)
        self.positive_float("learning_rate", dimensions.learning_rate)
        self.positive_integer("input_dim", dimensions.input_dim)
        self.positive_integer("hidden_dim", dimensions.hidden_dim)
        self.positive_integer("output_dim", dimensions.output_dim)

    def stack(self, stack: _LayerStack, prefix: str = "stack") -> None:
        self.positive_integer(f"{prefix}_num_layers", stack.num_layers)
        self.dropout(f"{prefix}_dropout_probability", stack.dropout_probability)
        self.residual(
            f"{prefix}_residual_connection_option", stack.residual_connection_option
        )

    def optional_stack(
        self,
        *,
        hidden_dim: int | None,
        num_layers: int | None,
        dropout_probability: float | None,
        residual_connection_option: object,
        prefix: str = "stack",
    ) -> None:
        if hidden_dim is not None:
            self.positive_integer(f"{prefix}_hidden_dim", hidden_dim)
        if num_layers is not None:
            self.positive_integer(f"{prefix}_num_layers", num_layers)
        if dropout_probability is not None:
            self.dropout(f"{prefix}_dropout_probability", dropout_probability)
        self.residual(
            f"{prefix}_residual_connection_option", residual_connection_option
        )

    def residual(self, key: str, selected: object) -> None:
        if selected is not None and not self.is_selectable_residual(selected):
            raise TypeError(f"{key} must be a concrete ResidualConfig type or None.")

    @staticmethod
    def is_selectable_residual(value: object) -> TypeGuard[type["ResidualConfig"]]:
        from emperor.config import ConfigBase
        from emperor.layers import ResidualConfig

        if not isinstance(value, type) or not issubclass(value, ResidualConfig):
            return False
        if isabstract(value):
            return False
        owner_hook = "_registry_owner"
        return (
            getattr(value, owner_hook) is not getattr(ResidualConfig, owner_hook)
            or value.registry_owner is not ConfigBase.registry_owner
        )

    def flat_stack_controls(self, settings: object) -> None:
        """Check stack and controller fields in a package's flat settings dataclass."""
        for key, validator, optional in _stack_control_fields(type(settings)):
            value = getattr(settings, key)
            if value is None and optional:
                continue
            if validator is None:
                prefix = key.removesuffix("stack_gate_flag")
                self.gate(
                    value, getattr(settings, f"{prefix}gate_option"), prefix=prefix
                )
            else:
                validator(self, key, value)

    def positive_integer(self, key: str, value: object) -> None:
        value = self.__integer_value(key, value)
        self.positive(key, value)

    def positive_float(self, key: str, value: object) -> None:
        value = self.__float_value(key, value)
        self.positive(key, value)

    def positive(self, key: str, value: int | float) -> None:
        if not value > 0:
            raise self.__value_error(key, "must be positive", value)

    def nonnegative_integer(self, key: str, value: object) -> None:
        value = self.__integer_value(key, value)
        self.nonnegative(key, value)

    def nonnegative_float(self, key: str, value: object) -> None:
        value = self.__float_value(key, value)
        self.nonnegative(key, value)

    def nonnegative(self, key: str, value: int | float) -> None:
        if not value >= 0:
            raise self.__value_error(key, "must be non-negative", value)

    def probability(self, key: str, value: object) -> None:
        value = self.__float_value(key, value)
        if not 0.0 <= value <= 1.0:
            raise self.__value_error(key, "must be between 0 and 1", value)

    def dropout(self, key: str, value: object) -> None:
        if self.allow_full_dropout:
            self.probability(key, value)
            return
        value = self.__float_value(key, value)
        if not 0.0 <= value < 1.0:
            raise self.__value_error(key, "must be in [0.0, 1.0)", value)

    def halting_threshold(self, key: str, value: object) -> None:
        value = self.__float_value(key, value)
        if not 0.0 < value <= 1.0:
            raise self.__value_error(key, "must be in (0.0, 1.0]", value)

    def gate(self, enabled: object, option: object, *, prefix: str = "") -> None:
        from emperor.layers import LayerGateOptions

        flag_key = f"{prefix}stack_gate_flag"
        option_key = f"{prefix}gate_option"
        if type(enabled) is not bool:
            raise TypeError(f"{self.package}: {flag_key!r} must be bool")
        if option is not None and not isinstance(option, LayerGateOptions):
            raise TypeError(
                f"{self.package}: {option_key!r} must be a LayerGateOptions member"
            )
        if enabled and option is None:
            raise ValueError(
                f"{self.package}: {option_key!r} must be set when {flag_key!r} is True"
            )

    def shared_gate(self, layer_gate_enabled: bool, shared_config: object) -> None:
        if layer_gate_enabled and shared_config is not None:
            raise ValueError(
                f"{self.package}: 'stack_gate_flag' and 'shared_gate_config' are mutually exclusive"
            )

    def __value_error(
        self, key: str, requirement: str, value: int | float
    ) -> ValueError:
        message = self.invalid_value_format.format(
            key=key, requirement=requirement, value=value
        )
        return ValueError(f"{self.package}: {message}")

    def __integer_value(self, key: str, value: object) -> int:
        if type(value) is not int:
            raise TypeError(
                f"{self.package}: {key!r} must be int; got {type(value).__name__}"
            )
        return value

    def __float_value(self, key: str, value: object) -> float:
        if type(value) is float or (self.allow_integer_floats and type(value) is int):
            return value
        raise TypeError(
            f"{self.package}: {key!r} must be float; got {type(value).__name__}"
        )


_FieldValidator = Callable[[BuildSettingsValidation, str, object], None]


# Flat package settings expose the same stack and controller roles at named scopes.
# Precompute the relevant fields once per settings type.
@cache
def _stack_control_fields(
    settings_type: type[object],
) -> tuple[tuple[str, _FieldValidator | None, bool], ...]:
    validators: dict[str, _FieldValidator | None] = {
        "num_layers": BuildSettingsValidation.positive_integer,
        "stack_hidden_dim": BuildSettingsValidation.positive_integer,
        "recurrent_max_steps": BuildSettingsValidation.positive_integer,
        "dropout_probability": BuildSettingsValidation.dropout,
        "halting_dropout": BuildSettingsValidation.dropout,
        "halting_threshold": BuildSettingsValidation.halting_threshold,
        "residual_connection_option": BuildSettingsValidation.residual,
        "stack_gate_flag": None,
        "decay_warmup_batches": BuildSettingsValidation.nonnegative_integer,
        "decay_rate": BuildSettingsValidation.nonnegative_float,
        "mask_threshold": BuildSettingsValidation.probability,
        "mask_floor": BuildSettingsValidation.probability,
        "mask_surrogate_scale": BuildSettingsValidation.positive_float,
        "mask_transition_width": BuildSettingsValidation.positive_float,
    }
    fields: list[tuple[str, _FieldValidator | None, bool]] = []
    for key, hint in get_type_hints(settings_type).items():
        for suffix, validator in validators.items():
            if key == suffix or key.endswith("_" + suffix):
                fields.append((key, validator, type(None) in get_args(hint)))
                break
    return tuple(fields)
