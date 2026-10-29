from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class HierarchicalByteDecoderOutput:
    """Packed predictions in batch/token/byte order; loss is auxiliary only."""

    logits: Tensor
    token_offsets: Tensor
    loss: Tensor


@dataclass(frozen=True)
class ByteGenerationOptions:
    """Sampling, when enabled, always uses the fixed byte/control alphabet."""

    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None
    constrain_utf8: bool = False


@dataclass(frozen=True)
class ByteTokenGenerationOutput:
    data: bytes
    stop_reason: str
    new_bytes: int

    @property
    def text(self) -> str | None:
        try:
            return self.data.decode("utf-8")
        except UnicodeDecodeError:
            return None

    @property
    def utf8_status(self) -> str:
        try:
            self.data.decode("utf-8")
        except UnicodeDecodeError as error:
            return (
                "incomplete" if error.reason == "unexpected end of data" else "invalid"
            )
        return "valid"
