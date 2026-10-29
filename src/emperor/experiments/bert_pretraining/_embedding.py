"""Keep BERT's masked token IDs bound to exact text-based input spellings."""

from torch import Tensor, nn

from emperor.embedding._token_text import TokenTextAdapter
from emperor.embedding.contextual import ByteContextualEmbeddingConfig
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
from emperor.layers import LayerState


class BertTextTokenEmbedding(nn.Module):
    def __init__(
        self,
        config: HierarchicalByteEmbeddingConfig | ByteContextualEmbeddingConfig,
        vocabulary_size: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if not isinstance(
            config, (HierarchicalByteEmbeddingConfig, ByteContextualEmbeddingConfig)
        ):
            raise TypeError(
                "BERT text embeddings require HierarchicalByteEmbeddingConfig or ByteContextualEmbeddingConfig."
            )
        self.uses_contextual_embedding = isinstance(
            config, ByteContextualEmbeddingConfig
        )
        output_dimension = self.__output_dimension(config)
        if output_dimension != hidden_dim:
            raise ValueError(
                "BERT text embedding output dimension must equal hidden_dim."
            )
        maximum_token_bytes = self.__maximum_token_bytes(config)
        self.token_vocabulary = TokenTextAdapter(vocabulary_size, maximum_token_bytes)
        self.embedding = config.build()

    @staticmethod
    def __output_dimension(
        config: HierarchicalByteEmbeddingConfig | ByteContextualEmbeddingConfig,
    ) -> int | None:
        if isinstance(config, ByteContextualEmbeddingConfig):
            return config.hidden_dim
        return config.output_dim

    @staticmethod
    def __maximum_token_bytes(
        config: HierarchicalByteEmbeddingConfig | ByteContextualEmbeddingConfig,
    ) -> int | None:
        if isinstance(config, HierarchicalByteEmbeddingConfig):
            return config.max_token_bytes
        return None

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> LayerState:
        token_texts = self.token_vocabulary(input_ids)
        active_tokens = attention_mask != 0
        embedding_state = self.embedding(token_texts, attention_mask=active_tokens)
        return LayerState(hidden=embedding_state.hidden, loss=embedding_state.loss)
