"""Trainer and runtime setup records and Lightning option translation."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from model_runtime.packages import RuntimeDefaultsSpec


class TrainerSettings(TypedDict):
    trainer_args: dict[str, Any]
    callbacks: list[Callback]
    terminal_checkpoint_callback: NotRequired[ModelCheckpoint | None]


class RuntimeSettings(TypedDict):
    data_num_workers: int | None
    run_test_after_fit: bool
    seed: int | None


def trainer_arguments(
    runtime_defaults: RuntimeDefaultsSpec, config_overrides: dict[str, Any]
) -> dict[str, Any]:
    trainer_args: dict[str, Any] = {}
    for key, value in runtime_defaults.items_with_prefix("TRAINER_"):
        if value is None:
            continue
        clean_key = key[len("TRAINER_") :].lower()
        trainer_args[clean_key] = config_overrides.get(key.lower(), value)

    for key, value in config_overrides.items():
        if not key.startswith("trainer_"):
            continue
        clean_key = key[len("trainer_") :]
        if value is not None:
            trainer_args[clean_key] = value
    return trainer_args
