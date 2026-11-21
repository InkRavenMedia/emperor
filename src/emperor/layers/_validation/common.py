from __future__ import annotations

from numbers import Integral

from emperor._validation import _adaptive_grouping_paths
from emperor.config import ConfigBase

_HALTING_CONFIG_FIELDS = (
    "input_dim",
    "threshold",
    "dropout_probability",
    "hidden_state_mode",
    "halting_gate_config",
)
_MEMORY_CONFIG_FIELDS = (
    "input_dim",
    "output_dim",
    "memory_position_option",
    "test_time_training_learning_rate",
    "test_time_training_num_inner_steps",
    "model_config",
)


def _validate_halting_lifecycle_owner(
    halting_config,
    *,
    field_name: str,
    owner_name: str,
) -> None:
    try:
        owner = halting_config._registry_owner()
    except NotImplementedError as exc:
        raise ValueError(
            f"{field_name} must be a concrete halting config for {owner_name}"
        ) from exc

    supports_interface = getattr(owner, "implements_halting_interface", None)
    if (
        isinstance(owner, type)
        and callable(supports_interface)
        and supports_interface()
    ):
        _minimum_halting_steps(halting_config, owner_name)
        return
    built_owner_name = getattr(owner, "__name__", type(owner).__name__)
    raise ValueError(
        f"{field_name} {type(halting_config).__name__} builds "
        f"{built_owner_name}, which does not implement the HaltingInterface "
        f"required by {owner_name}"
    )


def _positive_step_count(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{field_name} must be an integer or None, received {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(
            f"{field_name} must be greater than or equal to 1, received {value}"
        )
    return int(value)


def _minimum_halting_steps(halting_config, owner_name: str) -> int:
    configured = getattr(halting_config, "min_steps", None)
    if configured is None:
        return 1
    minimum = _positive_step_count(configured, "min_steps")
    if minimum > 1:
        strategy = halting_config._registry_owner()
        if getattr(strategy, "supports_minimum_step_delay", False) is not True:
            raise ValueError(
                f"halting_config.min_steps={minimum} requires minimum-step delay "
                f"support for {owner_name}. Strategy {strategy.__name__} can halt "
                "on its first step."
            )
    return minimum


def _validate_halting_owner_step_contract(
    halting_config,
    *,
    owner_step_limit: int | None,
    owner_name: str,
) -> None:
    minimum = _minimum_halting_steps(halting_config, owner_name)
    if minimum == 1:
        return
    if owner_step_limit is None:
        raise ValueError(
            f"halting_config.min_steps={minimum} is not supported for {owner_name}. "
            "This owner has no defined owner step limit."
        )
    if minimum > _positive_step_count(owner_step_limit, "owner_step_limit"):
        raise ValueError(
            "min_steps must be less than or equal to the owner step limit, "
            f"received min_steps={minimum} and owner_step_limit={owner_step_limit}."
        )


def _validate_halting_required_update_count(
    halting_config,
    *,
    required_update_count: int,
    owner_name: str,
) -> None:
    minimum = _minimum_halting_steps(halting_config, owner_name)
    if minimum < required_update_count:
        raise ValueError(
            "halting_config.min_steps must be greater than or equal to the required "
            f"update count for {owner_name}, received min_steps={minimum} and "
            f"required_update_count={required_update_count}."
        )


def _matches_config_contract(config: object, field_names: tuple[str, ...]) -> bool:
    return isinstance(config, ConfigBase) and all(
        hasattr(config, field_name) for field_name in field_names
    )


def _validate_no_grouping_with_context_controllers(
    config: ConfigBase,
    *,
    owner_name: str,
    controllers: tuple[tuple[str, object | None], ...],
) -> None:
    active_controller_names = tuple(
        name for name, controller in controllers if controller is not None
    )
    if not active_controller_names:
        return

    grouping_paths = _adaptive_grouping_paths(
        config,
        root=owner_name,
    )
    if not grouping_paths:
        return
    controller_list = ", ".join(active_controller_names)
    raise ValueError(
        f"{owner_name} cannot combine enabled adaptive parameter grouping with "
        f"{controller_list}: context sharing is restricted inside "
        f"memory owners. Found grouping at {grouping_paths[0]}."
    )
