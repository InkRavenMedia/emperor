from collections.abc import Callable
from typing import Final, Generic, Protocol, TypeVar


def _validate_dimensions(dimensions: tuple[tuple[str, int], ...]) -> None:
    for name, value in dimensions:
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0, received {value}.")


def _validate_embedding_dropout(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "embedding dropout_probability must be in [0.0, 1.0], "
            f"received {probability}."
        )


class _BertEmbedding(Protocol):
    dropout_probability: Final[float]
    token_type_vocab_size: Final[int]


class _MaskedWordHead(Protocol):
    decoder_weight_tying_flag: Final[bool]


class _NextSentenceHead(Protocol):
    output_dim: Final[int]


class _GptEmbedding(Protocol):
    hierarchical_language_model_flag: Final[bool]
    dropout_probability: Final[float]
    byte_embedding_dim: Final[int]
    byte_num_heads: Final[int]


class _LanguageHead(Protocol):
    weight_tying_flag: Final[bool]


_Bert = TypeVar("_Bert", bound=_BertEmbedding)
_Masked = TypeVar("_Masked", bound=_MaskedWordHead)
_Sentence = TypeVar("_Sentence", bound=_NextSentenceHead)
_Gpt = TypeVar("_Gpt", bound=_GptEmbedding)
_Lm = TypeVar("_Lm", bound=_LanguageHead)
_Result = TypeVar("_Result")


class BertBoundaryBuilder(Generic[_Bert, _Masked, _Sentence, _Result]):
    _boundary: Callable[[_Bert, _Masked, _Sentence], _Result]
    input_dim: int
    hidden_dim: int
    output_dim: int
    sequence_length: int
    embedding_options: _Bert
    mlm_head_options: _Masked
    nsp_head_options: _Sentence

    def build_boundary_config(self) -> _Result:
        _validate_dimensions(self._dimensions())
        if self.nsp_head_options.output_dim != 2:
            raise ValueError(
                "BERT next-sentence prediction requires exactly 2 output "
                f"classes, received {self.nsp_head_options.output_dim}."
            )
        _validate_embedding_dropout(self.embedding_options.dropout_probability)
        if (
            self.mlm_head_options.decoder_weight_tying_flag
            and self.input_dim != self.output_dim
        ):
            raise ValueError(
                "BERT MLM decoder weight tying requires input_dim to equal "
                f"output_dim, received input_dim={self.input_dim} and "
                f"output_dim={self.output_dim}."
            )
        return self._boundary(
            self.embedding_options, self.mlm_head_options, self.nsp_head_options
        )

    def _dimensions(self) -> tuple[tuple[str, int], ...]:
        return (
            ("input_dim", self.input_dim),
            ("hidden_dim", self.hidden_dim),
            ("output_dim", self.output_dim),
            ("sequence_length", self.sequence_length),
            ("token_type_vocab_size", self.embedding_options.token_type_vocab_size),
            ("nsp_output_dim", self.nsp_head_options.output_dim),
        )


class GptBoundaryBuilder(Generic[_Gpt, _Lm, _Result]):
    _boundary: Callable[[_Gpt, _Lm], _Result]
    _exclusive_hierarchical_lm = False
    input_dim: int
    hidden_dim: int
    output_dim: int
    sequence_length: int
    embedding_options: _Gpt
    lm_head_options: _Lm

    def build_boundary_config(self) -> _Result:
        self._validate()
        return self._boundary(self.embedding_options, self.lm_head_options)

    def _validate_embedding_modes(self) -> None:
        pass

    def _validate(self) -> None:
        options = self.embedding_options
        if type(options.hierarchical_language_model_flag) is not bool:
            raise TypeError("hierarchical_language_model_flag must be bool")
        if options.hierarchical_language_model_flag:
            if self.lm_head_options.weight_tying_flag:
                raise ValueError(
                    "Hierarchical language modeling requires lm_head_weight_tying_flag=False"
                )
            if self._exclusive_hierarchical_lm and (
                getattr(options, "contextual_flag", False)
                or getattr(options, "hierarchical_flag", False)
            ):
                raise ValueError(
                    "Hierarchical language modeling conflicts with contextual and encoder-only embedding modes"
                )
            for name in (
                "byte_embedding_dim",
                "byte_encoder_num_layers",
                "byte_decoder_num_layers",
                "byte_num_heads",
                "byte_feed_forward_dim",
                "byte_limit",
            ):
                if (
                    type(getattr(options, name)) is not int
                    or getattr(options, name) <= 0
                ):
                    raise ValueError(f"hierarchical {name} must be a positive integer")
            if options.byte_embedding_dim % options.byte_num_heads:
                raise ValueError(
                    "hierarchical byte width must be divisible by byte heads"
                )
        self._validate_embedding_modes()
        _validate_dimensions(
            (
                ("input_dim", self.input_dim),
                ("hidden_dim", self.hidden_dim),
                ("output_dim", self.output_dim),
                ("sequence_length", self.sequence_length),
            )
        )
        _validate_embedding_dropout(self.embedding_options.dropout_probability)
        if self.lm_head_options.weight_tying_flag and self.input_dim != self.output_dim:
            raise ValueError(
                "GPT LM head weight tying requires input_dim to equal output_dim, "
                f"received input_dim={self.input_dim} and output_dim={self.output_dim}."
            )
