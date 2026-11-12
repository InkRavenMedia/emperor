"""Interface consumed by owners of dynamic-memory modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.memory._config import MemoryPositionOptions


class MemoryInterface(Protocol):
    """Memory behavior required by layer execution pipelines."""

    memory_position_option: MemoryPositionOptions

    def __call__(self, hidden: Tensor) -> Tensor: ...  # pragma: no cover


class SourceMemoryInterface(Protocol):
    def generate_source_memory(
        self,
        source: Tensor,
        *,
        excluded_token_mask: Tensor | None = None,
        batch_first_flag: bool = True,
    ) -> tuple[Tensor, Tensor | None]: ...  # pragma: no cover
