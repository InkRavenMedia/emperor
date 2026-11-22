from dataclasses import dataclass

import models.gpt.expert_linear.config as config
from models._language_boundary import GptBoundaryBuilder
from models.gpt.expert_linear._config_defaults import (
    gpt_embedding_options,
    gpt_lm_head_options,
)
from models.gpt.expert_linear.runtime_options import (
    GptEmbeddingOptions,
    GptLmHeadOptions,
)


@dataclass(frozen=True)
class GptBoundaryConfig:
    embedding_options: GptEmbeddingOptions
    lm_head_options: GptLmHeadOptions


@dataclass(frozen=True)
class BoundaryConfigDependencies:
    input_dim: int
    hidden_dim: int
    output_dim: int
    sequence_length: int
    embedding_options: GptEmbeddingOptions | None
    lm_head_options: GptLmHeadOptions | None


class BoundaryConfigFactory(
    GptBoundaryBuilder[GptEmbeddingOptions, GptLmHeadOptions, GptBoundaryConfig]
):
    _boundary = staticmethod(GptBoundaryConfig)
    _exclusive_hierarchical_lm = True

    def __init__(self, dependencies: BoundaryConfigDependencies) -> None:
        self.input_dim = dependencies.input_dim
        self.hidden_dim = dependencies.hidden_dim
        self.output_dim = dependencies.output_dim
        self.sequence_length = dependencies.sequence_length
        embedding_options = dependencies.embedding_options
        if embedding_options is None:
            embedding_options = gpt_embedding_options(config)
        self.embedding_options = embedding_options
        lm_head_options = dependencies.lm_head_options
        if lm_head_options is None:
            lm_head_options = gpt_lm_head_options(config)
        self.lm_head_options = lm_head_options
