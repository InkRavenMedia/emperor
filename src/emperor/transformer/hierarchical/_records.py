from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from emperor.embedding.hierarchical.decoder import ByteTokenGenerationOutput

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class HierarchicalTextGenerationOutput(ByteTokenGenerationOutput):
    new_tokens: int
    completed_tokens: tuple[bytes, ...]
    partial_token: bytes
    generator_state: "Tensor | None" = field(default=None, compare=False, repr=False)
