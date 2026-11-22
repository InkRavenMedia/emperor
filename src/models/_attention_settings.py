"""Validation shared by the attention model packages' typed construction options."""

from collections.abc import Mapping
from copy import copy
from dataclasses import Field, replace
from functools import cache
from typing import Any, ClassVar, Generic, Protocol, TypeVar

from emperor.layers import LayerGateOptions, ResidualConfig
from model_runtime.packages.configuration import require_finite_learning_rate
from model_runtime.packages.settings_validation import BuildSettingsValidation

_Settings = TypeVar("_Settings")


@cache
def strict_float_validation(package: str) -> BuildSettingsValidation:
    return BuildSettingsValidation(
        package, allow_integer_floats=False, allow_full_dropout=True
    )


@cache
def integer_compatible_validation(package: str) -> BuildSettingsValidation:
    return BuildSettingsValidation(
        package, allow_integer_floats=True, allow_full_dropout=True
    )


class _StackSource(Protocol):
    @property
    def hidden_dim(self) -> int | None: ...

    @property
    def num_layers(self) -> int | None: ...

    @property
    def dropout_probability(self) -> float | None: ...

    @property
    def residual_connection_option(self) -> type[ResidualConfig] | None: ...


class _LayerController(Protocol):
    @property
    def stack_gate_flag(self) -> bool: ...

    @property
    def gate_option(self) -> LayerGateOptions | None: ...

    @property
    def halting_threshold(self) -> float | None: ...

    @property
    def halting_dropout(self) -> float: ...


class _RecurrentController(Protocol):
    @property
    def recurrent_stack_gate_flag(self) -> bool: ...

    @property
    def recurrent_gate_option(self) -> LayerGateOptions | None: ...

    @property
    def recurrent_halting_threshold(self) -> float | None: ...

    @property
    def recurrent_halting_dropout(self) -> float: ...


class _Memory(Protocol):
    @property
    def memory_num_memory_slots(self) -> int: ...

    @property
    def memory_test_time_training_learning_rate(self) -> float | None: ...

    @property
    def memory_test_time_training_num_inner_steps(self) -> int | None: ...


class _Mask(Protocol):
    @property
    def mask_threshold(self) -> float: ...

    @property
    def mask_floor(self) -> float: ...

    @property
    def mask_surrogate_scale(self) -> float: ...

    @property
    def mask_transition_width(self) -> float: ...


class _OptionsRecord(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


_Record = TypeVar("_Record", bound=_OptionsRecord)


class RuntimeBuilder(Generic[_Settings]):
    """Refresh the builder's derived fields together after validating replacement settings."""

    _runtime: _Settings

    @property
    def runtime(self) -> _Settings:
        return self._runtime

    @runtime.setter
    def runtime(self, value: _Settings) -> None:
        replacement = copy(self)
        replacement._configure_runtime(value)
        self.__dict__.update(replacement.__dict__)

    def _configure_runtime(self, runtime: _Settings) -> None:
        raise NotImplementedError


def validate_dimensions(
    policy: BuildSettingsValidation, dimensions: Mapping[str, object]
) -> None:
    if "learning_rate" in dimensions:
        require_finite_learning_rate("LEARNING_RATE", dimensions["learning_rate"])
    for field in (
        "batch_size",
        "learning_rate",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "sequence_length",
    ):
        if field in dimensions:
            if field == "learning_rate":
                policy.positive_float(field, dimensions[field])
            else:
                policy.positive_integer(field, dimensions[field])


def validate_residual(field: str, option: object) -> None:
    if option is not None and not ResidualConfig.is_concrete_implementation(option):
        raise TypeError(f"{field} must be a concrete ResidualConfig type or None.")


def validate_stack_source(
    policy: BuildSettingsValidation, source: _StackSource
) -> None:
    if source.hidden_dim is not None:
        policy.positive_integer("hidden_dim", source.hidden_dim)
    if source.num_layers is not None:
        policy.positive_integer("num_layers", source.num_layers)
    if source.dropout_probability is not None:
        policy.dropout("dropout_probability", source.dropout_probability)
    validate_residual("residual_connection_option", source.residual_connection_option)


def validate_controller(
    policy: BuildSettingsValidation,
    controller: _LayerController,
    *,
    allow_unset_threshold: bool = False,
) -> None:
    policy.gate(controller.stack_gate_flag, controller.gate_option)
    policy.shared_gate(
        controller.stack_gate_flag, getattr(controller, "shared_gate_config", None)
    )
    if controller.halting_threshold is not None or not allow_unset_threshold:
        policy.halting_threshold("halting_threshold", controller.halting_threshold)
    policy.probability("halting_dropout", controller.halting_dropout)


def validate_recurrent_controller(
    policy: BuildSettingsValidation,
    controller: _RecurrentController,
    *,
    allow_unset_threshold: bool = False,
) -> None:
    policy.gate(
        controller.recurrent_stack_gate_flag,
        controller.recurrent_gate_option,
        prefix="recurrent_",
    )
    if controller.recurrent_halting_threshold is not None or not allow_unset_threshold:
        policy.halting_threshold(
            "recurrent_halting_threshold",
            controller.recurrent_halting_threshold,
        )
    policy.probability(
        "recurrent_halting_dropout", controller.recurrent_halting_dropout
    )


def validate_memory(policy: BuildSettingsValidation, memory: _Memory) -> None:
    policy.positive_integer("memory_num_memory_slots", memory.memory_num_memory_slots)
    if memory.memory_test_time_training_learning_rate is not None:
        policy.positive_float(
            "memory_test_time_training_learning_rate",
            memory.memory_test_time_training_learning_rate,
        )
    if memory.memory_test_time_training_num_inner_steps is not None:
        policy.positive_integer(
            "memory_test_time_training_num_inner_steps",
            memory.memory_test_time_training_num_inner_steps,
        )


def validate_mask(policy: BuildSettingsValidation, options: _Mask) -> None:
    policy.probability("mask_threshold", options.mask_threshold)
    policy.probability("mask_floor", options.mask_floor)
    policy.positive_float("mask_surrogate_scale", options.mask_surrogate_scale)
    policy.positive_float("mask_transition_width", options.mask_transition_width)


def require_stages(experiment: object, *fields: str) -> None:
    if experiment is None:
        raise ValueError("experiment_config is required to build the model")
    for name in fields:
        if getattr(experiment, name, None) is None:
            raise ValueError(f"experiment_config.{name} is required")


def replace_option_paths(
    options: _Record, updates: Mapping[tuple[str, ...], Any]
) -> _Record:
    """Apply sibling updates together so coupled gate options validate as one choice."""
    direct: dict[str, Any] = {}
    nested: dict[str, dict[tuple[str, ...], Any]] = {}
    for (field, *tail), value in updates.items():
        if tail:
            nested.setdefault(field, {})[tuple(tail)] = value
        else:
            direct[field] = value
    for field, children in nested.items():
        direct[field] = replace_option_paths(
            direct.get(field, getattr(options, field)), children
        )
    return replace(options, **direct)
