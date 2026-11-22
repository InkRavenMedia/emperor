from dataclasses import dataclass

from models._language_boundary import BertBoundaryBuilder
from models.bert.expert_linear_adaptive.runtime_options import (
    BertEmbeddingOptions,
    BertMlmHeadOptions,
    BertNspHeadOptions,
)


@dataclass(frozen=True)
class BertBoundaryConfig:
    embedding_options: BertEmbeddingOptions
    mlm_head_options: BertMlmHeadOptions
    nsp_head_options: BertNspHeadOptions


@dataclass(frozen=True)
class BoundaryConfigDependencies:
    input_dim: int
    output_dim: int
    sequence_length: int
    embedding_options: BertEmbeddingOptions
    mlm_head_options: BertMlmHeadOptions
    nsp_head_options: BertNspHeadOptions


class BoundaryConfigFactory(
    BertBoundaryBuilder[
        BertEmbeddingOptions, BertMlmHeadOptions, BertNspHeadOptions, BertBoundaryConfig
    ]
):
    _boundary = staticmethod(BertBoundaryConfig)

    def __init__(self, dependencies: BoundaryConfigDependencies) -> None:
        self.input_dim = dependencies.input_dim
        self.output_dim = dependencies.output_dim
        self.sequence_length = dependencies.sequence_length
        self.embedding_options = dependencies.embedding_options
        self.mlm_head_options = dependencies.mlm_head_options
        self.nsp_head_options = dependencies.nsp_head_options

    def _dimensions(self) -> tuple[tuple[str, int], ...]:
        return (
            ("input_dim", self.input_dim),
            ("output_dim", self.output_dim),
            ("sequence_length", self.sequence_length),
            ("token_type_vocab_size", self.embedding_options.token_type_vocab_size),
        )
