"""Disk-backed WikiText sentence pairs for hierarchical masked-word BERT."""

import hashlib
import random
import re
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from tokenizers import Regex
from tokenizers.pre_tokenizers import Split

from emperor.datasets._base import DataModule
from emperor.datasets.text._bert_vocabulary import BERT_SPECIAL_TOKENS
from emperor.datasets.text._wikitext103 import is_article_heading, load_raw_wikitext103
from emperor.embedding.hierarchical.decoder import OUTPUT_SYMBOLS

from ._masked_words import BertMaskedWordBatch
from ._next_sentence import sample_next_sentence

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")
_WORD_PATTERN = r"[\p{L}\p{N}\p{M}]+(?:['’\-‐‑][\p{L}\p{N}\p{M}]+)*|\S"


def _sentence_rows(source, max_words):
    # Keep case, combining marks and internal apostrophes/hyphens. Other
    # punctuation becomes context-only, and whitespace separates positions.
    pre_tokenizer = Split(Regex(_WORD_PATTERN), behavior="removed", invert=True)
    document = 0
    for row in source:
        for line in row["text"].splitlines():
            if is_article_heading(line):
                document += 1
                continue
            for sentence in _SENTENCE_BOUNDARY.split(line):
                pieces = pre_tokenizer.pre_tokenize_str(sentence)
                if pieces:
                    yield {
                        "document": document,
                        "words": [word for word, _ in pieces[:max_words]],
                    }


class _SentencePairs(torch.utils.data.Dataset):
    def __init__(self, sentences, seed, random_next_probability):
        self.sentences = sentences
        self.seed = seed
        self.random_next_probability = random_next_probability

    def __len__(self):
        return max(0, len(self.sentences) - 1)

    def __getitem__(self, index):
        first = self.sentences[index]
        next_sentence = self.sentences[index + 1]
        has_successor = first["document"] == next_sentence["document"]
        second_index, label = sample_next_sentence(
            len(self.sentences),
            index,
            has_successor,
            self.random_next_probability,
            random.Random(self.seed + index),
        )
        second = (
            next_sentence if second_index == index + 1 else self.sentences[second_index]
        )
        return first["words"], second["words"], label


class _WikiTextBertHierarchical(DataModule):
    """Raw word strings and masked byte targets, without a fixed word vocabulary."""

    flattened_input_dim = OUTPUT_SYMBOLS
    num_classes = OUTPUT_SYMBOLS
    sequence_length = 35
    hierarchical_masked_word_flag = True

    def __init__(
        self,
        batch_size=64,
        sequence_length=35,
        max_token_bytes=64,
        mlm_probability=0.15,
        random_next_probability=0.5,
        root="data",
        num_workers=4,
        seed=None,
    ):
        super().__init__(root=root, num_workers=num_workers)
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if type(sequence_length) is not int or sequence_length < 5:
            raise ValueError("sequence_length must be at least 5 for BERT pairs.")
        if type(max_token_bytes) is not int or max_token_bytes < max(
            map(len, BERT_SPECIAL_TOKENS)
        ):
            raise ValueError("max_token_bytes must fit all BERT control markers.")
        if not 0.0 <= mlm_probability <= 1.0:
            raise ValueError("mlm_probability must be between 0 and 1.")
        if not 0.0 <= random_next_probability <= 1.0:
            raise ValueError("random_next_probability must be between 0 and 1.")
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_token_bytes = max_token_bytes
        self.mlm_probability = mlm_probability
        self.random_next_probability = random_next_probability
        self.seed = seed

    def _dataset(self, split):
        raise NotImplementedError

    def prepare_data(self):
        for split in ("train", "validation", "test"):
            self._dataset(split)

    def _build_split(self, split, offset):
        source = self._dataset(split)
        identity = (
            f"masked-bert-v1:{split}:{source._fingerprint}:{self.sequence_length}"
        )
        sentences = Dataset.from_generator(
            _sentence_rows,
            gen_kwargs={"source": source, "max_words": self.sequence_length - 4},
            features=Features(
                {"document": Value("int64"), "words": Sequence(Value("string"))}
            ),
            fingerprint=hashlib.sha256(identity.encode()).hexdigest(),
            cache_dir=str(Path(self.root) / "bert_masked_word_sentences"),
            keep_in_memory=False,
        )
        seed = (
            torch.empty((), dtype=torch.long).random_().item()
            if self.seed is None
            else self.seed + offset
        )
        return _SentencePairs(sentences, seed, self.random_next_probability)

    def _setup_fit(self):
        self.train = self._build_split("train", 0)
        self._setup_validate()

    def _setup_validate(self):
        self.val = self._build_split("validation", 1)

    def _setup_test(self):
        self.test = self._build_split("test", 2)

    def _loader(self, data, train=False):
        generator = (
            torch.Generator().manual_seed(self.seed) if self.seed is not None else None
        )
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=train and len(data) > 0,
            num_workers=self.num_workers,
            generator=generator,
            collate_fn=partial(
                BertMaskedWordBatch.collate,
                sequence_length=self.sequence_length,
                max_token_bytes=self.max_token_bytes,
                mlm_probability=self.mlm_probability,
                generator=generator,
            ),
        )

    def get_dataloader(self, train):
        return self._loader(self.train if train else self.val, train)

    def _get_test_dataloader(self):
        return self._loader(self.test)


class WikiText103BertHierarchical(_WikiTextBertHierarchical):
    """Raw WikiText-103 for hierarchical masked-word pretraining."""

    def _dataset(self, split):
        return load_raw_wikitext103(split, self.root)


class WikiText2BertHierarchical(_WikiTextBertHierarchical):
    """Raw WikiText-2 for hierarchical masked-word pretraining."""

    def _dataset(self, split):
        return load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split=split,
            cache_dir=str(self.root),
        )
