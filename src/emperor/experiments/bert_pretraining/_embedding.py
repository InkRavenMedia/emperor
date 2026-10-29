"""Bind BERT's masked token IDs to native hierarchical byte embeddings."""

from torch import Tensor, nn

from emperor.embedding._token_text import TokenTextAdapter
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
from emperor.layers import LayerState


class BertTextTokenEmbedding(nn.Module):
    def __init__(
        self,
        config: HierarchicalByteEmbeddingConfig,
        vocabulary_size: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if not isinstance(config, HierarchicalByteEmbeddingConfig):
            raise TypeError(
                "BERT text embeddings require HierarchicalByteEmbeddingConfig."
            )
        if config.output_dim != hidden_dim:
            raise ValueError(
                "BERT text embedding output dimension must equal hidden_dim."
            )
        maximum_token_bytes = config.max_token_bytes
        self.token_vocabulary = TokenTextAdapter(vocabulary_size, maximum_token_bytes)
        self.embedding = config.build()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> LayerState:
        token_texts = self.token_vocabulary(input_ids)
        active_tokens = attention_mask != 0
        return self.embedding(token_texts, attention_mask=active_tokens)
