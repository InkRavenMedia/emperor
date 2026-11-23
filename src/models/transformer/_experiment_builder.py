from collections.abc import Callable
from typing import TYPE_CHECKING, Final, Protocol, TypeVar, cast

from emperor.config import ConfigBase
from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.layers import (
    LayerStackConfig,
    NormalizationOptions,
    RecurrentCompositionConfig,
    RecurrentCompressionConfig,
)
from models import _sequence_expansion
from models._positional_embedding import supported_position_fields
from models._recurrent_compression import configure_sequence_stacks

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class _EncoderOptions(Protocol):
    sequence_expansion_flag: Final[bool]


class _ExperimentSettings(Protocol):
    positional_embedding_option: Final[type[AbsolutePositionalEmbeddingConfig]]
    model_dim: Final[int]
    vocab_size: Final[int]
    source_sequence_length: Final[int]
    target_sequence_length: Final[int]
    dropout_probability: Final[float]
    encoder_options: Final[_EncoderOptions]
    encoder_output_normalization: Final[NormalizationOptions]
    decoder_output_normalization: Final[NormalizationOptions]


_S = TypeVar("_S", bound=_ExperimentSettings)
_Stack = TypeVar("_Stack", bound=ConfigBase)
_RecurrentIn = TypeVar(
    "_RecurrentIn", bound=RecurrentCompositionConfig, contravariant=True
)
_Recurrent = TypeVar("_Recurrent", bound=RecurrentCompositionConfig)
_Experiment = TypeVar("_Experiment", covariant=True)


class _ExperimentFactory(Protocol[_RecurrentIn, _Experiment]):
    def __call__(
        self,
        *,
        source_positional_embedding_config: AbsolutePositionalEmbeddingConfig,
        target_positional_embedding_config: AbsolutePositionalEmbeddingConfig,
        encoder_config: LayerStackConfig | _RecurrentIn,
        decoder_config: LayerStackConfig | _RecurrentIn,
        vocab_size: int,
        model_dim: int,
        source_sequence_length: int,
        target_sequence_length: int,
        dropout_probability: float,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        label_smoothing: float,
        warmup_steps: int,
        generation_metrics_flag: bool,
        encoder_output_normalization: NormalizationOptions,
        decoder_output_normalization: NormalizationOptions,
    ) -> _Experiment: ...


def position_builder(
    runtime: _ExperimentSettings,
) -> Callable[[int], AbsolutePositionalEmbeddingConfig]:
    positional = runtime.positional_embedding_option
    position_kwargs = supported_position_fields(
        positional,
        embedding_dim=runtime.model_dim,
        padding_idx=0,
        auto_expand_flag=False,
    )

    def build_position(length: int) -> AbsolutePositionalEmbeddingConfig:
        return positional(num_embeddings=length, **position_kwargs)

    return build_position


def expanded_stacks(
    runtime: _S,
    build_encoder: Callable[[_S], _Stack],
    build_decoder: Callable[[_S], _Stack],
) -> tuple[_Stack, _Stack]:
    encoder_config = build_encoder(runtime)
    source_capacity = None
    if runtime.encoder_options.sequence_expansion_flag:
        source_capacity = _sequence_expansion.expanded_sequence_capacity(
            encoder_config, runtime.source_sequence_length
        )
    encoder_config = _sequence_expansion.with_expansion_attention_capacity(
        encoder_config, runtime.source_sequence_length
    )
    resolved_decoder_config = build_decoder(runtime)
    decoder_config = _sequence_expansion.with_expansion_attention_capacity(
        resolved_decoder_config, runtime.target_sequence_length, source_capacity
    )
    return cast(_Stack, encoder_config), cast(_Stack, decoder_config)


def assemble_experiment(
    runtime: _ExperimentSettings,
    create_experiment: _ExperimentFactory[_Recurrent, _Experiment],
    encoder_config: LayerStackConfig | _Recurrent,
    decoder_config: LayerStackConfig | _Recurrent,
    source_positional_embedding_config: AbsolutePositionalEmbeddingConfig,
    target_positional_embedding_config: AbsolutePositionalEmbeddingConfig,
) -> _Experiment:
    return create_experiment(
        source_positional_embedding_config=source_positional_embedding_config,
        target_positional_embedding_config=target_positional_embedding_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        vocab_size=runtime.vocab_size,
        model_dim=runtime.model_dim,
        source_sequence_length=runtime.source_sequence_length,
        target_sequence_length=runtime.target_sequence_length,
        dropout_probability=runtime.dropout_probability,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        label_smoothing=0.1,
        warmup_steps=4_000,
        generation_metrics_flag=True,
        encoder_output_normalization=runtime.encoder_output_normalization,
        decoder_output_normalization=runtime.decoder_output_normalization,
    )


class _TranslationSettings(Protocol):
    recurrent_compression_config: Final[RecurrentCompressionConfig | None]
    batch_size: Final[int]
    learning_rate: Final[float]
    vocab_size: Final[int]
    model_dim: Final[int]
    source_sequence_length: Final[int]
    target_sequence_length: Final[int]


_Translation = TypeVar("_Translation", bound=_TranslationSettings)


def build_translation_model_config(
    runtime: _Translation, build_experiment: Callable[[_Translation], ConfigBase]
) -> "ModelConfig":
    from emperor.config import ModelConfig

    sequence_length = max(
        runtime.source_sequence_length, runtime.target_sequence_length
    )
    experiment_config = configure_sequence_stacks(
        build_experiment(runtime), runtime.recurrent_compression_config
    )
    return ModelConfig(
        batch_size=runtime.batch_size,
        learning_rate=runtime.learning_rate,
        sequence_length=sequence_length,
        input_dim=runtime.vocab_size,
        hidden_dim=runtime.model_dim,
        output_dim=runtime.vocab_size,
        experiment_config=experiment_config,
    )
