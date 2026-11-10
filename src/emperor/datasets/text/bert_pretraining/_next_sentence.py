import random
from collections.abc import Iterable
from dataclasses import dataclass

from emperor.datasets.text.bert_pretraining._tokenizer import (
    _normalise_text_units,
)


@dataclass(frozen=True)
class BertNextSentencePair:
    sentence_a: str
    sentence_b: str
    next_sentence_label: int
    sentence_a_index: int
    sentence_b_index: int


def build_bert_next_sentence_pairs(
    text_units: Iterable[str],
    random_next_probability: float = 0.5,
    rng: random.Random | None = None,
) -> list[BertNextSentencePair]:
    if random_next_probability < 0.0 or random_next_probability > 1.0:
        raise ValueError("random_next_probability must be between 0.0 and 1.0.")

    units = list(_normalise_text_units(text_units))
    if len(units) < 2:
        return []

    rng = rng or random.Random()
    pairs = []
    for sentence_a_index in range(len(units) - 1):
        sentence_b_index, next_sentence_label = sample_next_sentence(
            len(units), sentence_a_index, True, random_next_probability, rng
        )

        pairs.append(
            BertNextSentencePair(
                sentence_a=units[sentence_a_index],
                sentence_b=units[sentence_b_index],
                next_sentence_label=next_sentence_label,
                sentence_a_index=sentence_a_index,
                sentence_b_index=sentence_b_index,
            )
        )
    return pairs


def sample_next_sentence(
    sentence_count: int,
    anchor: int,
    has_successor: bool,
    random_next_probability: float,
    rng: random.Random,
) -> tuple[int, int]:
    """Sample an NSP pair with adjacent anchor/successor excluded from negatives."""
    excluded_count = 1 + int(has_successor)
    negative_count = sentence_count - excluded_count
    if has_successor and (
        negative_count == 0 or rng.random() >= random_next_probability
    ):
        return anchor + 1, 0
    # Sampling the ordered complement preserves the legacy seeded choices.
    negative = rng.randrange(negative_count)
    if negative >= anchor:
        negative += excluded_count
    return negative, 1
