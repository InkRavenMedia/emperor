from dataclasses import dataclass

import models.bert.linear_adaptive.config as config
from models._language_boundary import BertBoundaryBuilder
from models.bert.linear_adaptive import _config_defaults as config_defaults
from models.bert.linear_adaptive.runtime_options import (
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
    hidden_dim: int
    output_dim: int
    sequence_length: int
    embedding_options: BertEmbeddingOptions | None
    mlm_head_options: BertMlmHeadOptions | None
    nsp_head_options: BertNspHeadOptions | None


class BoundaryConfigFactory(
    BertBoundaryBuilder[
        BertEmbeddingOptions, BertMlmHeadOptions, BertNspHeadOptions, BertBoundaryConfig
    ]
):
    _boundary = staticmethod(BertBoundaryConfig)

    def __init__(self, dependencies: BoundaryConfigDependencies) -> None:
        self.input_dim = dependencies.input_dim
        self.hidden_dim = dependencies.hidden_dim
        self.output_dim = dependencies.output_dim
        self.sequence_length = dependencies.sequence_length
        embedding_options = dependencies.embedding_options
        if embedding_options is None:
            embedding_options = config_defaults.bert_embedding_options(config)
        self.embedding_options = embedding_options
        mlm_head_options = dependencies.mlm_head_options
        if mlm_head_options is None:
            mlm_head_options = config_defaults.bert_mlm_head_options(config)
        self.mlm_head_options = mlm_head_options
        nsp_head_options = dependencies.nsp_head_options
        if nsp_head_options is None:
            nsp_head_options = config_defaults.bert_nsp_head_options(config)
        self.nsp_head_options = nsp_head_options
