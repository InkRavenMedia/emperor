from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.embedding.absolute import (
        TextLearnedPositionalEmbeddingConfig,
        TextSinusoidalPositionalEmbeddingConfig,
    )


@dataclass
class SequenceExpansionConfig(ConfigBase):
    embedding_dim: int | None = optional_field(
        "Hidden feature width; inferred by the transformer block."
    )
    max_new_tokens: int | None = optional_field(
        "Maximum positions added per call; select the most probable count in 1..max_new_tokens for each sample."
    )
    model_config: ConfigBase | None = optional_field(
        "Required count predictor config. Receives input_dim=embedding_dim "
        "and output_dim=max_new_tokens."
    )
    positional_embedding_config: "TextLearnedPositionalEmbeddingConfig | TextSinusoidalPositionalEmbeddingConfig | None" = optional_field(  # noqa: E501
        "Optional text positional embedding for generated positions; None disables it. "
        "Its embedding_dim is inferred; capacity and other options remain caller-configured."
    )

    def _registry_owner(self) -> type:
        from emperor.transformer.sequence_expansion._model import SequenceExpansion

        return SequenceExpansion
