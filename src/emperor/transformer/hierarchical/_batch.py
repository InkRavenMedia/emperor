from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from emperor.embedding.hierarchical.decoder import END_OF_DOCUMENT, END_OF_TOKEN

from ._validation import HierarchicalLanguageModelBatchValidator


@dataclass(frozen=True)
class HierarchicalLanguageModelBatch:
    VALIDATOR = HierarchicalLanguageModelBatchValidator

    context_texts: tuple[tuple[str | bytes, ...], ...]
    byte_prefix_ids: Tensor
    byte_lengths: Tensor
    labels: Tensor
    attention_mask: Tensor
    bos_mask: Tensor
    byte_count: int

    def to(self, device, **kwargs):
        tensor_field_names = (
            "byte_prefix_ids",
            "byte_lengths",
            "labels",
            "attention_mask",
            "bos_mask",
        )
        transferred_tensors = {
            name: getattr(self, name).to(device=device, **kwargs)
            for name in tensor_field_names
        }
        return replace(self, **transferred_tensors)

    @classmethod
    def collate(cls, examples: Sequence[dict]):
        maximum_tokens = cls.VALIDATOR.validate_examples(examples)
        encoded_targets, maximum_bytes = cls.__encode_target_rows(examples)
        return cls.__assemble_batch(
            examples, encoded_targets, maximum_tokens, maximum_bytes
        )

    @classmethod
    def __encode_target_rows(cls, examples):
        encoded_target_rows = []
        maximum_bytes = 0
        for example in examples:
            context_texts = example["context_texts"]
            target_texts = example["target_texts"]
            beginning_mask = example["bos_mask"]
            cls.VALIDATOR.validate_example(context_texts, target_texts, beginning_mask)
            encoded_targets = cls.VALIDATOR.validate_and_encode_targets(
                context_texts, target_texts
            )
            encoded_target_rows.append(encoded_targets)
            target_byte_lengths = (
                len(target_bytes)
                for target_bytes in encoded_targets
                if target_bytes is not None
            )
            maximum_row_bytes = max(target_byte_lengths, default=0)
            maximum_bytes = max(maximum_bytes, maximum_row_bytes)
        return encoded_target_rows, maximum_bytes

    @classmethod
    def __assemble_batch(
        cls, examples, encoded_target_rows, maximum_tokens, maximum_bytes
    ):
        import torch

        token_shape = (len(examples), maximum_tokens)
        prefix_shape = (*token_shape, maximum_bytes)
        byte_prefix_ids = torch.zeros(prefix_shape, dtype=torch.long)
        byte_lengths = torch.zeros(token_shape, dtype=torch.long)
        attention_mask = torch.zeros(token_shape, dtype=torch.bool)
        beginning_mask = torch.zeros_like(attention_mask)
        context_rows, packed_labels = [], []
        total_byte_count = 0
        for row_index, (example, encoded_targets) in enumerate(
            zip(examples, encoded_target_rows, strict=True)
        ):
            target_count = len(encoded_targets)
            padding_count = maximum_tokens - target_count
            context_tokens = tuple(example["context_texts"])
            padded_context_tokens = context_tokens + ("",) * padding_count
            context_rows.append(padded_context_tokens)
            attention_mask[row_index, :target_count] = True
            beginning_flags = torch.tensor(example["bos_mask"], dtype=torch.bool)
            beginning_mask[row_index, :target_count] = beginning_flags
            row_labels, row_byte_count = cls.__pack_target_row(
                encoded_targets, byte_prefix_ids[row_index], byte_lengths[row_index]
            )
            packed_labels.extend(row_labels)
            total_byte_count += row_byte_count
        batched_contexts = tuple(context_rows)
        label_tensor = torch.tensor(packed_labels, dtype=torch.long)
        return cls(
            batched_contexts,
            byte_prefix_ids,
            byte_lengths,
            label_tensor,
            attention_mask,
            beginning_mask,
            total_byte_count,
        )

    @classmethod
    def __pack_target_row(cls, encoded_targets, prefix_ids, byte_lengths):
        import torch

        target_labels = []
        row_byte_count = 0
        for token_index, target_bytes in enumerate(encoded_targets):
            if target_bytes is None:
                target_labels.append(END_OF_DOCUMENT)
                continue
            target_byte_count = len(target_bytes)
            byte_lengths[token_index] = target_byte_count
            byte_values = list(target_bytes)
            target_byte_ids = torch.tensor(byte_values, dtype=torch.long)
            prefix_ids[token_index, :target_byte_count] = target_byte_ids
            target_labels.extend(target_bytes)
            target_labels.append(END_OF_TOKEN)
            row_byte_count += target_byte_count
        return target_labels, row_byte_count
