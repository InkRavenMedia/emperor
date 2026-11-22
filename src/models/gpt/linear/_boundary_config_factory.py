from dataclasses import dataclass

import models.gpt.linear.config as config
from models._language_boundary import GptBoundaryBuilder
from models.gpt.linear import _config_defaults as config_defaults
from models.gpt.linear.runtime_options import GptEmbeddingOptions, GptLmHeadOptions


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

    def __init__(self, dependencies: BoundaryConfigDependencies) -> None:
        self.input_dim = dependencies.input_dim
        self.hidden_dim = dependencies.hidden_dim
        self.output_dim = dependencies.output_dim
        self.sequence_length = dependencies.sequence_length
        self.embedding_options = (
            dependencies.embedding_options
            or config_defaults.gpt_embedding_options(config)
        )
        self.lm_head_options = (
            dependencies.lm_head_options or config_defaults.gpt_lm_head_options(config)
        )
