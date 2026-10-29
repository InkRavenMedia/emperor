from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from emperor.embedding.hierarchical import BYTE_VALUES, END_OF_TOKEN
from emperor.embedding.hierarchical.encoder._config import (
    HierarchicalByteEmbeddingConfig,
)
from emperor.embedding.hierarchical.encoder._validation import (
    HierarchicalByteEmbeddingValidator,
)
from emperor.layers import LayerState
from emperor.nn import Module, grouped_forward_calls


class HierarchicalByteEmbedding(Module):
    VALIDATOR = HierarchicalByteEmbeddingValidator
    POOLING_SYMBOL_ID = END_OF_TOKEN

    def __init__(
        self,
        cfg: HierarchicalByteEmbeddingConfig,
        overrides: HierarchicalByteEmbeddingConfig | None = None,
    ) -> None:
        self.VALIDATOR.validate_config_types(cfg, overrides)
        super().__init__()
        overridden_config = self._override_config(cfg, overrides)
        self.cfg = copy.deepcopy(overridden_config)
        self._max_group_tokens = self.VALIDATOR.resolve_config(self.cfg)
        self.byte_embedding = nn.Embedding(BYTE_VALUES, self.cfg.byte_embedding_dim)
        self.byte_position = (
            self.cfg.byte_position_config.build()
            if self.cfg.byte_position_config is not None
            else None
        )
        self.encoder = self.cfg.encoder_config.build()
        self.projection = self._build_from_config(
            self.cfg.projection_config,
            input_dim=self.cfg.byte_embedding_dim,
            output_dim=self.cfg.output_dim,
        )

    def forward(
        self,
        token_texts: Sequence[Sequence[str | bytes]],
        attention_mask: Tensor | None = None,
    ) -> LayerState:
        batch_size, token_count = self.VALIDATOR.validate_forward_inputs(
            token_texts, attention_mask
        )
        encoded_tokens = self.__encode_token_rows(token_texts, token_count)
        flat_token_count = len(encoded_tokens)
        valid_token_indices = self.__select_valid_token_indices(
            flat_token_count, attention_mask
        )
        token_groups = self.__group_tokens_by_byte_length(
            encoded_tokens, valid_token_indices
        )
        self.VALIDATOR.validate_token_group_sizes(token_groups, self._max_group_tokens)
        auxiliary_loss = self.byte_embedding.weight.new_zeros(())
        if not valid_token_indices:
            return self.__empty_token_embeddings(
                batch_size, token_count, auxiliary_loss
            )

        valid_token_count = len(valid_token_indices)
        pooled_state = self.__encode_token_groups(
            token_groups, valid_token_count, auxiliary_loss
        )
        projected_state = self.__project_token_embeddings(pooled_state)
        return self.__restore_token_positions(
            projected_state, valid_token_indices, batch_size, token_count
        )

    def __encode_token_rows(self, token_texts, token_count) -> list[bytes]:
        flat_tokens = [token for token_row in token_texts for token in token_row]
        return [
            self.VALIDATOR.validate_and_encode_token(
                token, token_index, token_count, self.cfg.max_token_bytes
            )
            for token_index, token in enumerate(flat_tokens)
        ]

    def __select_valid_token_indices(
        self, flat_token_count, attention_mask
    ) -> list[int]:
        valid_tokens = (
            [True] * flat_token_count
            if attention_mask is None
            else attention_mask.reshape(-1).tolist()
        )
        return [
            token_index for token_index, is_valid in enumerate(valid_tokens) if is_valid
        ]

    def __group_tokens_by_byte_length(self, encoded_tokens, valid_token_indices):
        token_groups = defaultdict(list)
        for compact_index, original_index in enumerate(valid_token_indices):
            token_bytes = encoded_tokens[original_index]
            token_byte_count = len(token_bytes)
            token_groups[token_byte_count].append((compact_index, token_bytes))
        return token_groups

    def __empty_token_embeddings(self, batch_size, token_count, auxiliary_loss):
        output_shape = (batch_size, token_count, self.cfg.output_dim)
        empty_embeddings = self.byte_embedding.weight.new_zeros(output_shape)
        return LayerState(hidden=empty_embeddings, loss=auxiliary_loss)

    def __encode_token_groups(self, token_groups, valid_token_count, auxiliary_loss):
        pooled_embeddings = None
        pooled_shape = (valid_token_count, self.cfg.byte_embedding_dim)
        with grouped_forward_calls() as group_schedule:
            for token_group in token_groups.values():
                with group_schedule():
                    compact_indices = [index for index, _ in token_group]
                    encoded_features, group_loss = self.__encode_token_group(
                        token_group
                    )
                    if pooled_embeddings is None:
                        pooled_embeddings = encoded_features.new_zeros(pooled_shape)
                    pooling_symbol_features = encoded_features[:, 0]
                    pooled_embeddings[compact_indices] = pooling_symbol_features
                    group_token_count = len(token_group)
                    group_weight = group_token_count / valid_token_count
                    weighted_group_loss = group_loss * group_weight
                    auxiliary_loss = auxiliary_loss + weighted_group_loss
        return LayerState(hidden=pooled_embeddings, loss=auxiliary_loss)

    def __encode_token_group(self, token_group):
        pooling_prefixed_bytes = [
            [self.POOLING_SYMBOL_ID, *token_bytes] for _, token_bytes in token_group
        ]
        byte_ids = torch.tensor(
            pooling_prefixed_bytes,
            dtype=torch.long,
            device=self.byte_embedding.weight.device,
        )
        byte_embeddings = self.byte_embedding(byte_ids)
        positioned_byte_embeddings = byte_embeddings
        if self.byte_position is not None:
            positioned_byte_embeddings = byte_embeddings + self.byte_position(byte_ids)
        return self.encoder(source_token_embeddings=positioned_byte_embeddings)

    def __project_token_embeddings(self, pooled_state):
        projection_input = LayerState(hidden=pooled_state.hidden)
        projected_state = self.projection(projection_input)
        auxiliary_loss = pooled_state.loss
        if projected_state.loss is not None:
            auxiliary_loss = auxiliary_loss + projected_state.loss
        return LayerState(hidden=projected_state.hidden, loss=auxiliary_loss)

    def __restore_token_positions(
        self, projected_state, valid_token_indices, batch_size, token_count
    ):
        flat_token_count = batch_size * token_count
        flat_output_shape = (flat_token_count, self.cfg.output_dim)
        flat_embeddings = projected_state.hidden.new_zeros(flat_output_shape)
        flat_embeddings[valid_token_indices] = projected_state.hidden
        token_embeddings = flat_embeddings.reshape(
            batch_size, token_count, self.cfg.output_dim
        )
        return LayerState(hidden=token_embeddings, loss=projected_state.loss)
