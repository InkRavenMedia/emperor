from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from emperor.config import ConfigBase, ModelConfig
    from models._recurrent_compression import CompressionOptions


class _SequenceSettings(Protocol):
    runtime: Final[CompressionOptions]
    learning_rate: Final[float | None]
    batch_size: Final[int | None]
    input_dim: Final[int | None]
    hidden_dim: Final[int | None]
    output_dim: Final[int | None]
    sequence_length: Final[int | None]


def sequence_model_config(
    settings: _SequenceSettings, experiment_config: ConfigBase
) -> ModelConfig:
    from emperor.config import ModelConfig
    from models._recurrent_compression import configure_sequence_stacks

    compression = settings.runtime.recurrent_compression_config
    experiment_config = configure_sequence_stacks(experiment_config, compression)
    return ModelConfig(
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
        input_dim=settings.input_dim,
        hidden_dim=settings.hidden_dim,
        output_dim=settings.output_dim,
        sequence_length=settings.sequence_length,
        experiment_config=experiment_config,
    )
