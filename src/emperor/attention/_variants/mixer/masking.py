"""Exclude padding tokens from Mixer inputs and outputs."""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from emperor.attention._variants.mixer.reshaping import MixerReshaper


class MixerMask:
    def __init__(self, reshaper: "MixerReshaper") -> None:
        self.reshaper = reshaper

    def prepare_inputs(
        self, tokens: Tensor, key_padding_mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        if key_padding_mask is None:
            return tokens, None
        excluded_token_mask = self.__get_excluded_token_mask(key_padding_mask)
        masked_tokens = self.apply_to_tokens(tokens, excluded_token_mask)
        return masked_tokens, excluded_token_mask

    def __get_excluded_token_mask(self, key_padding_mask: Tensor) -> Tensor:
        if key_padding_mask.dtype == torch.bool:
            excluded_token_mask = key_padding_mask
        else:
            excluded_token_mask = key_padding_mask.isneginf()
        return self.reshaper.from_batch_first(excluded_token_mask)

    @staticmethod
    def apply_to_tokens(tokens: Tensor, excluded_token_mask: Tensor | None) -> Tensor:
        if excluded_token_mask is None:
            return tokens
        excluded_feature_mask = excluded_token_mask.unsqueeze(-1)
        return tokens.masked_fill(excluded_feature_mask, 0)
