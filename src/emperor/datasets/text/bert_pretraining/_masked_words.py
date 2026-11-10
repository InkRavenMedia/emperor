from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from emperor.datasets.text._bert_vocabulary import (
    BERT_CLS_TOKEN,
    BERT_MASK_TOKEN,
    BERT_PAD_TOKEN,
    BERT_SEP_TOKEN,
    BERT_SPECIAL_TOKENS,
    BERT_UNK_TOKEN,
)
from emperor.embedding.hierarchical.decoder import END_OF_TOKEN

from ._examples import _truncate_longest_first

BertWordPair = tuple[Sequence[str], Sequence[str], int]


@dataclass(frozen=True)
class BertMaskedWordBatch:
    """Corrupted words and packed decoder targets in supervised position order."""

    input_words: tuple[tuple[str, ...], ...]
    attention_mask: Tensor
    token_type_ids: Tensor
    prediction_mask: Tensor
    byte_prefix_ids: Tensor
    byte_lengths: Tensor
    byte_labels: Tensor
    next_sentence_labels: Tensor
    oversized_word_count: int

    def to(self, device, **kwargs):
        return replace(
            self,
            attention_mask=self.attention_mask.to(device, **kwargs),
            token_type_ids=self.token_type_ids.to(device, **kwargs),
            prediction_mask=self.prediction_mask.to(device, **kwargs),
            byte_prefix_ids=self.byte_prefix_ids.to(device, **kwargs),
            byte_lengths=self.byte_lengths.to(device, **kwargs),
            byte_labels=self.byte_labels.to(device, **kwargs),
            next_sentence_labels=self.next_sentence_labels.to(device, **kwargs),
        )

    @classmethod
    def collate(
        cls,
        pairs: Sequence[BertWordPair],
        *,
        sequence_length: int,
        max_token_bytes: int,
        mlm_probability: float,
        generator: torch.Generator | None = None,
    ):
        if not pairs or sequence_length < 5:
            raise ValueError(
                "Masked-word batches require sentence pairs and at least five positions."
            )
        if torch.utils.data.get_worker_info() is not None:
            generator = None  # DataLoader seeds each worker's torch RNG.
        shape = (len(pairs), sequence_length)
        attention_mask = torch.zeros(shape, dtype=torch.bool)
        token_type_ids = torch.zeros(shape, dtype=torch.long)
        prediction_mask = torch.zeros_like(attention_mask)
        input_words = []
        candidates = []
        next_sentence_labels = []
        oversized_word_count = 0
        for row, (first, second, nsp_label) in enumerate(pairs):
            first, second = list(first), list(second)
            _truncate_longest_first(first, second, sequence_length - 3)
            words = [BERT_CLS_TOKEN, *first, BERT_SEP_TOKEN, *second, BERT_SEP_TOKEN]
            attention_mask[row, : len(words)] = True
            token_type_ids[row, len(first) + 2 : len(words)] = 1
            next_sentence_labels.append(nsp_label)
            for column, word in enumerate(words):
                if word in BERT_SPECIAL_TOKENS:
                    continue
                encoded = word.encode("utf-8")
                is_word = any(character.isalnum() for character in word)
                if len(encoded) > max_token_bytes:
                    words[column] = BERT_UNK_TOKEN
                    oversized_word_count += int(is_word)
                elif is_word:
                    candidates.append((row, column, encoded))
            words.extend([BERT_PAD_TOKEN] * (sequence_length - len(words)))
            input_words.append(words)

        selected = (
            (torch.rand(len(candidates), generator=generator) < mlm_probability)
            .nonzero()
            .flatten()
        )
        if candidates and mlm_probability > 0 and selected.numel() == 0:
            selected = torch.randint(len(candidates), (1,), generator=generator)
        targets = []
        for index in selected.tolist():
            row, column, encoded = candidates[index]
            prediction_mask[row, column] = True
            input_words[row][column] = BERT_MASK_TOKEN
            targets.append(encoded)

        byte_prefix_ids = torch.zeros(
            (len(targets), 1, max_token_bytes), dtype=torch.long
        )
        byte_lengths = torch.tensor([len(word) for word in targets], dtype=torch.long)
        byte_labels = []
        for index, encoded in enumerate(targets):
            byte_prefix_ids[index, 0, : len(encoded)] = torch.tensor(list(encoded))
            byte_labels.extend(encoded)
            byte_labels.append(END_OF_TOKEN)
        return cls(
            input_words=tuple(tuple(words) for words in input_words),
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            prediction_mask=prediction_mask,
            byte_prefix_ids=byte_prefix_ids,
            byte_lengths=byte_lengths.unsqueeze(1),
            byte_labels=torch.tensor(byte_labels, dtype=torch.long),
            next_sentence_labels=torch.tensor(next_sentence_labels, dtype=torch.long),
            oversized_word_count=oversized_word_count,
        )
