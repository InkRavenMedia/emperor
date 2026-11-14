from __future__ import annotations

import torch
from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.layers._composition.recurrent.config import (
    HierarchicalReasoningModelRecurrentConfig,
    RecurrentCompositionConfig,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
)

_GRADIENT_WINDOW_FIELDS = {
    "no_gradient_transition_count",
    "gradient_transition_count",
}


class RecurrentIterationScheduleValidator(ValidatorBase):
    @staticmethod
    def validate_config(config: object) -> None:
        concrete_config_types = (
            RecurrentLayerConfig,
            TinyRecursiveModelRecurrentConfig,
            HierarchicalReasoningModelRecurrentConfig,
        )
        if not isinstance(config, concrete_config_types):
            raise TypeError(
                "config must be a concrete RecurrentCompositionConfig, got "
                f"{type(config).__name__}."
            )
        if isinstance(config, RecurrentLayerConfig):
            transitions_per_iteration = 1
        elif isinstance(config, TinyRecursiveModelRecurrentConfig):
            transitions_per_iteration = config.latent_updates_per_answer_update + 1
        else:
            transitions_per_iteration = config.low_cycles + 1
        _validate_smooth_iteration_growth_controls(
            config,
            transitions_per_iteration=transitions_per_iteration,
        )

    @staticmethod
    def validate_checkpoint_progress(
        value: object,
        checkpoint_key: str,
    ) -> int:
        if not isinstance(value, Tensor):
            raise TypeError(
                f"{checkpoint_key} must be a Tensor in a recurrent checkpoint."
            )
        if value.shape != torch.Size([]):
            raise ValueError(f"{checkpoint_key} must be a scalar Tensor.")
        if value.dtype != torch.long:
            raise TypeError(f"{checkpoint_key} must use torch.long dtype.")
        progress = int(value.item())
        if progress < 0:
            raise ValueError(f"{checkpoint_key} must be non-negative.")
        return progress


def _validate_recurrent_iteration_controls(
    config: object,
    maximum_iterations: int,
    transitions_per_iteration: int,
) -> None:
    initial_iterations = config.initial_iterations
    _validate_positive_integer("initial_iterations", initial_iterations)
    if initial_iterations > maximum_iterations:
        raise ValueError(
            "initial_iterations must be less than or equal to the variant's "
            f"maximum of {maximum_iterations}."
        )

    for field_name in (
        "iteration_increment",
        "forward_calls_before_iteration_increment",
    ):
        _validate_positive_integer(field_name, getattr(config, field_name))

    minimum_transition_count = initial_iterations * transitions_per_iteration
    no_gradient_count = config.no_gradient_transition_count
    gradient_count = config.gradient_transition_count
    _validate_gradient_window_counts(
        no_gradient_count, gradient_count, minimum_transition_count
    )


def _validate_gradient_window_counts(
    no_gradient_count: object,
    gradient_count: object,
    minimum_transition_count: int,
) -> None:
    if no_gradient_count is not None:
        if isinstance(no_gradient_count, bool) or not isinstance(
            no_gradient_count, int
        ):
            raise TypeError(
                "no_gradient_transition_count must be int, "
                f"got {type(no_gradient_count).__name__}."
            )
        if no_gradient_count < 0:
            raise ValueError(
                "no_gradient_transition_count must be greater than or equal to 0."
            )
        if no_gradient_count >= minimum_transition_count:
            raise ValueError(
                "no_gradient_transition_count must be less than the minimum active "
                f"{minimum_transition_count} transitions so at least one transition "
                "uses gradients."
            )
    if gradient_count is not None:
        _validate_positive_integer("gradient_transition_count", gradient_count)
        if gradient_count > minimum_transition_count:
            raise ValueError(
                "gradient_transition_count must be less than or equal to the "
                f"minimum active transition count of {minimum_transition_count}."
            )
    if gradient_count is not None and no_gradient_count is not None:
        raise ValueError(
            "gradient_transition_count and no_gradient_transition_count are "
            "mutually exclusive."
        )


def _validate_smooth_iteration_growth_controls(
    config: RecurrentCompositionConfig,
    transitions_per_iteration: int,
) -> None:
    value = config.smooth_iteration_growth_flag
    if value is not None and not isinstance(value, bool):
        raise TypeError(
            "smooth_iteration_growth_flag must be bool or None for "
            f"{type(config).__name__}, "
            f"got {type(value).__name__}."
        )
    if value is not True:
        return

    _validate_smooth_gradient_window(config, transitions_per_iteration)
    _validate_smooth_growth_cadence(config)


def _validate_smooth_gradient_window(
    config: RecurrentCompositionConfig, transitions_per_iteration: int
) -> None:
    gradient_count = config.gradient_transition_count
    no_gradient_count = config.no_gradient_transition_count
    if gradient_count is None and no_gradient_count is None:
        raise ValueError(
            "smooth iteration growth requires either gradient_transition_count or "
            "explicit no_gradient_transition_count=0."
        )
    if gradient_count is not None:
        _validate_positive_integer("gradient_transition_count", gradient_count)
        initial_iterations = config.initial_iterations
        _validate_positive_integer("initial_iterations", initial_iterations)
        minimum_transition_count = initial_iterations * transitions_per_iteration
        if gradient_count > minimum_transition_count:
            raise ValueError(
                "gradient_transition_count must be less than or equal to the "
                f"minimum active transition count of {minimum_transition_count}."
            )
    elif no_gradient_count != 0:
        raise ValueError(
            "smooth iteration growth only supports "
            "no_gradient_transition_count equal to 0."
        )
    if gradient_count is not None and no_gradient_count is not None:
        raise ValueError(
            "gradient_transition_count and no_gradient_transition_count are "
            "mutually exclusive."
        )


def _validate_smooth_growth_cadence(config: RecurrentCompositionConfig) -> None:
    iteration_increment = config.iteration_increment
    if (
        isinstance(iteration_increment, bool)
        or not isinstance(iteration_increment, int)
        or iteration_increment != 1
    ):
        raise ValueError(
            "smooth iteration growth requires iteration_increment to equal 1."
        )

    cadence = config.forward_calls_before_iteration_increment
    if (
        isinstance(cadence, bool)
        or not isinstance(cadence, int)
        or cadence < 2
        or cadence % 2 != 0
    ):
        raise ValueError(
            "smooth iteration growth requires "
            "forward_calls_before_iteration_increment to be an even integer "
            "greater than or equal to 2."
        )
    if config.memory_config is not None:
        raise ValueError("smooth iteration growth does not support memory_config.")


def _validate_positive_integer(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than or equal to 1.")
