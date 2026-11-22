from __future__ import annotations

from torch import Tensor

from models.transformer._model import TranslationModel


class Model(TranslationModel):
    def _embed_source(self, source_ids: Tensor) -> Tensor:
        hidden = self.source_embedding(source_ids) * self.embedding_scale
        hidden = hidden + self.source_positional_embedding(source_ids)
        return self.embedding_dropout(hidden)

    def _embed_target(self, target_ids: Tensor) -> Tensor:
        hidden = self.target_embedding(target_ids) * self.embedding_scale
        hidden = hidden + self.target_positional_embedding(target_ids)
        return self.embedding_dropout(hidden)

    def _decoder_inputs(self, target_ids: Tensor) -> tuple[Tensor, Tensor]:
        padding_mask = target_ids.eq(self.pad_token_id)
        target_embedding = self._embed_target(target_ids)
        return target_embedding, padding_mask

    def _validate_ids(self, ids: object, maximum_length: int, name: str) -> None:
        super()._validate_ids(ids, maximum_length, name)


__all__ = ["Model"]
