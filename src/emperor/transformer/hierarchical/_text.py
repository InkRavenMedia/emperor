import re
from collections.abc import Iterator

from ._validation import HierarchicalTextCodecValidator


class HierarchicalTextCodec:
    """HAT v2 whitespace spans; optional chunking is an explicit adaptation.

    Trailing whitespace stays with its preceding span; leading whitespace forms
    its own span. This is not the later hat-splitter's Unicode word-boundary rule.
    """

    VALIDATOR = HierarchicalTextCodecValidator

    def __init__(self, max_token_bytes: int, *, split_long_tokens: bool = False):
        self.VALIDATOR.validate_max_token_bytes(max_token_bytes)
        if type(split_long_tokens) is not bool:
            raise TypeError("split_long_tokens must be bool")
        self.max_token_bytes = max_token_bytes
        self.split_long_tokens = split_long_tokens

    def split_text(self, text: str) -> Iterator[str]:
        self.VALIDATOR.validate_text(text)
        for matched_span in re.finditer(r"\S+\s*|\s+", text):
            span_text = matched_span.group()
            if self.split_long_tokens:
                yield from self.__split_span(span_text)
            else:
                try:
                    byte_count = len(span_text.encode("utf-8"))
                except UnicodeEncodeError as error:
                    raise ValueError("text must be valid UTF-8") from error
                if byte_count > self.max_token_bytes:
                    raise ValueError(
                        "word exceeds max_token_bytes; increase the bound or explicitly enable split_long_tokens"
                    )
                yield span_text

    def __split_span(self, span_text):
        token_characters = []
        token_byte_count = 0
        for character in span_text:
            character_byte_count = self.VALIDATOR.validate_character_byte_count(
                character, self.max_token_bytes
            )
            extended_byte_count = token_byte_count + character_byte_count
            if extended_byte_count > self.max_token_bytes:
                yield "".join(token_characters)
                token_characters = []
                token_byte_count = 0
            token_characters.append(character)
            token_byte_count += character_byte_count
        if token_characters:
            yield "".join(token_characters)

    def training_windows(self, text: str, sequence_length: int) -> Iterator[dict]:
        self.VALIDATOR.validate_sequence_length(sequence_length)
        context_texts, target_texts, beginning_mask = [], [], []
        previous_token = ""
        is_first_token = True
        for target_token in self.split_text(text):
            context_texts.append(previous_token)
            target_texts.append(target_token)
            beginning_mask.append(is_first_token)
            previous_token, is_first_token = target_token, False
            if len(context_texts) == sequence_length:
                yield {
                    "context_texts": context_texts,
                    "target_texts": target_texts,
                    "bos_mask": beginning_mask,
                }
                context_texts, target_texts, beginning_mask = [], [], []
        context_texts.append(previous_token)
        target_texts.append(None)
        beginning_mask.append(is_first_token)
        yield {
            "context_texts": context_texts,
            "target_texts": target_texts,
            "bos_mask": beginning_mask,
        }
