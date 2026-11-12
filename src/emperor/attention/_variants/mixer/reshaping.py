"""Token layouts, fixed mixing capacity, and output restoration for Mixer."""

from typing import TYPE_CHECKING

from torch import Tensor
from torch.nn import functional as F

if TYPE_CHECKING:
    from emperor.attention._variants.mixer.config import MixerAttentionConfig


class MixerReshaper:
    def __init__(self, cfg: "MixerAttentionConfig") -> None:
        self.batch_first_flag = cfg.batch_first_flag
        self.sequence_length = cfg.sequence_length

    @property
    def sequence_axis(self) -> int:
        if self.batch_first_flag:
            return 1
        return 0

    def from_batch_first(self, values: Tensor) -> Tensor:
        if self.batch_first_flag:
            return values
        return values.transpose(0, 1)

    def prepare_token_axis_vectors(self, tokens: Tensor) -> Tensor:
        token_axis_vectors = tokens.movedim(self.sequence_axis, -1)
        padding_length = self.sequence_length - token_axis_vectors.size(-1)
        if padding_length:
            sequence_padding = (0, padding_length)
            token_axis_vectors = F.pad(token_axis_vectors, sequence_padding)
        return token_axis_vectors

    def prepare_memory_vectors(
        self, memory_tokens: Tensor, memory_sequence_length: int
    ) -> Tensor:
        memory_vectors = memory_tokens.movedim(self.sequence_axis, -1)
        padding_length = memory_sequence_length - memory_vectors.size(-1)
        memory_padding = (0, padding_length)
        return F.pad(memory_vectors, memory_padding)

    def restore_output_layout(
        self, mixed_token_axis_vectors: Tensor, input_sequence_length: int
    ) -> Tensor:
        mixed_tokens = mixed_token_axis_vectors.movedim(-1, self.sequence_axis)
        return mixed_tokens.narrow(self.sequence_axis, 0, input_sequence_length)
