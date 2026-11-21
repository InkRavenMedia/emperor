from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor._validation import (
    ValidatorBase,
    _config_validator,
    _configuration_visits,
    _validate_grouped_row_preservation,
)
from emperor.layers._composition.gate.validation import LayerGateValidator
from emperor.layers._composition.residual.validation import (
    ResidualConnectionValidator,
)
from emperor.layers._options import LayerNormPositionOptions, NormalizationOptions
from emperor.layers._validation.common import (
    _HALTING_CONFIG_FIELDS,
    _MEMORY_CONFIG_FIELDS,
    _matches_config_contract,
    _validate_halting_lifecycle_owner,
    _validate_no_grouping_with_context_controllers,
)

_RECURRENT_SHARED_OPTIONAL_FIELDS = {
    "smooth_iteration_growth_flag",
    "gate_config",
    "residual_config",
    "halting_config",
    "memory_config",
}

if TYPE_CHECKING:
    from emperor.layers._composition.recurrent.config import (
        RecurrentCompositionConfig,
    )


def _validate_initialization_standard_deviation(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(
            "initialization_standard_deviation must be a finite non-negative number."
        )


def _validate_variant_hidden(
    hidden: object,
    expected_feature_dim: int,
    field_name: str,
    owner_name: str,
) -> None:
    if not isinstance(hidden, Tensor) or not torch.is_floating_point(hidden):
        raise TypeError(f"{field_name} must be a floating-point Tensor.")
    if hidden.ndim < 2:
        raise ValueError(f"{field_name} must have rank >= 2 with feature-last layout.")
    if any(dimension == 0 for dimension in hidden.shape[:-1]):
        raise ValueError(f"{field_name} must have non-empty leading dimensions.")
    if hidden.shape[-1] != expected_feature_dim:
        raise ValueError(
            f"{field_name} last dimension must be {expected_feature_dim} for "
            f"{owner_name}, got {hidden.shape[-1]}."
        )


def _validate_variant_state(
    state: object,
    expected_feature_dim: int,
    owner_name: str,
) -> None:
    from emperor.layers._state import LayerState

    if not isinstance(state, LayerState):
        raise TypeError(
            f"state must be an instance of LayerState for {owner_name}, "
            f"got {type(state).__name__}"
        )
    _validate_variant_hidden(
        state.hidden,
        expected_feature_dim,
        field_name="state.hidden",
        owner_name=owner_name,
    )


def _validate_variant_transition_output(
    output_state: object,
    transition_input: Tensor,
    expected_feature_dim: int,
    owner_name: str,
    transition_name: str,
) -> None:
    from emperor.layers._state import LayerState

    if not isinstance(output_state, LayerState):
        raise TypeError(
            f"{transition_name} transition block must return LayerState, got "
            f"{type(output_state).__name__}."
        )
    _validate_variant_hidden(
        output_state.hidden,
        expected_feature_dim,
        field_name="transition output hidden",
        owner_name=owner_name,
    )
    try:
        output_state.validate_recurrent_shape(transition_input)
    except ValueError as error:
        raise ValueError(
            f"{transition_name} transition block must preserve hidden shape."
        ) from error
    if output_state.hidden.dtype != transition_input.dtype:
        raise ValueError(
            f"{transition_name} transition block must preserve hidden dtype."
        )
    if output_state.hidden.device != transition_input.device:
        raise ValueError(
            f"{transition_name} transition block must preserve hidden device."
        )


def _validate_recurrent_normalization_config(config: object, owner_name: str) -> None:
    for field_name, option_type in (
        ("recurrent_layer_norm_position", LayerNormPositionOptions),
        ("recurrent_normalization", NormalizationOptions),
    ):
        value = getattr(config, field_name)
        if not isinstance(value, option_type):
            raise TypeError(
                f"{field_name} must be a {option_type.__name__} value for "
                f"{owner_name}, got {type(value).__name__}."
            )


def _validate_recurrent_halting_config(halting_config: object, owner_name: str) -> None:
    if halting_config is not None:
        if not _matches_config_contract(halting_config, _HALTING_CONFIG_FIELDS):
            raise TypeError(
                "halting_config must be an instance of HaltingConfig for "
                f"{owner_name}, got {type(halting_config).__name__}."
            )
        _validate_halting_lifecycle_owner(
            halting_config,
            field_name="halting_config",
            owner_name=owner_name,
        )


def _validate_recurrent_memory_config(memory_config: object, owner_name: str) -> None:
    if memory_config is not None and not _matches_config_contract(
        memory_config,
        _MEMORY_CONFIG_FIELDS,
    ):
        raise TypeError(
            "memory_config must be an instance of DynamicMemoryConfig for "
            f"{owner_name}, got {type(memory_config).__name__}"
        )


class RecurrentCompositionValidator(ValidatorBase):
    GATE_VALIDATOR = LayerGateValidator
    RESIDUAL_VALIDATOR = ResidualConnectionValidator

    @classmethod
    def _validate_recurrent_controller_config(
        cls, config: RecurrentCompositionConfig
    ) -> None:
        owner_name = type(config).__name__
        for visit in _configuration_visits(config, owner_name):
            validate_placement = getattr(
                _config_validator(visit.config), "validate_recurrent_placement", None
            )
            if callable(validate_placement):
                validate_placement(
                    visit.config, path=visit.path, recurrent_config=config
                )
        _validate_grouped_row_preservation(config, root=owner_name)
        _validate_recurrent_normalization_config(config, owner_name)
        cls.GATE_VALIDATOR.validate_recurrent_gate_config(
            config.gate_config,
            owner_name=f"{owner_name}.gate_config",
        )
        cls.RESIDUAL_VALIDATOR.validate_residual_config(
            config.residual_config,
            owner_name=owner_name,
        )
        cls._validate_residual_execution(config)
        halting_config = config.halting_config
        _validate_recurrent_halting_config(halting_config, owner_name)
        memory_config = config.memory_config
        _validate_recurrent_memory_config(memory_config, owner_name)

        _validate_no_grouping_with_context_controllers(
            config,
            owner_name=owner_name,
            controllers=(("memory_config", memory_config),),
        )

    @staticmethod
    def _validate_residual_execution(config: RecurrentCompositionConfig) -> None:
        if config.residual_config is None:
            return
        residual_owner = config.residual_config.registry_owner()
        residual_owner.VALIDATOR.validate_stateless_execution(
            config.residual_config,
            owner_name=type(config).__name__,
        )

    @staticmethod
    def validate_halting_output(
        output_hidden: Tensor,
        candidate_hidden: Tensor,
    ) -> None:
        if output_hidden.shape != candidate_hidden.shape:
            raise ValueError(
                "recurrent halting must preserve the candidate hidden shape."
            )
        if output_hidden.dtype != candidate_hidden.dtype:
            raise ValueError(
                "recurrent halting must preserve the candidate hidden dtype."
            )
        if output_hidden.device != candidate_hidden.device:
            raise ValueError(
                "recurrent halting must preserve the candidate hidden device."
            )

    @staticmethod
    def validate_initial_buffer(
        buffer: Tensor,
        hidden: Tensor,
        name: str,
        expected_feature_dim: int,
    ) -> None:
        if tuple(buffer.shape) != (expected_feature_dim,):
            raise ValueError(
                f"{name} must have shape ({expected_feature_dim},), got "
                f"{tuple(buffer.shape)}."
            )
        if buffer.dtype != hidden.dtype or buffer.device != hidden.device:
            raise ValueError(
                f"{name} dtype/device must match state.hidden; move the recurrent "
                "composition to the input dtype and device before forward."
            )
