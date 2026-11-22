from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from torch import Tensor

from emperor.layers._composition.residual.base import ResidualState
from emperor.layers._composition.residual.validation import AttentionResidualValidator


@dataclass(slots=True)
class AttentionResidualState(ResidualState):
    """Forward-local sources mixed once per physical residual-depth execution."""

    VALIDATOR: ClassVar[type[AttentionResidualValidator]] = AttentionResidualValidator

    initial_source: Tensor
    block_size: int
    _completed_blocks: list[Tensor] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _partial_block: Tensor | None = field(default=None, init=False, repr=False)
    _partial_count: int = field(default=0, init=False, repr=False)
    _next_source_padding: Tensor | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.VALIDATOR.validate_positive_integer(
            self.block_size,
            name="block_size",
        )

    @property
    def sources(self) -> tuple[Tensor, ...]:
        partial_sources = ()
        if self._partial_block is not None:
            partial_sources = (self._partial_block,)
        return (self.initial_source, *self._completed_blocks, *partial_sources)

    @property
    def next_source_padding(self) -> Tensor | None:
        """Padding for the next depth read, including its current output."""
        return getattr(self, "_next_source_padding", None)

    @next_source_padding.setter
    def next_source_padding(self, padding: Tensor | None) -> None:
        if padding is not None:
            # Include the initial source and the block receiving the next output.
            depth = len(self._completed_blocks) + 2
            self.VALIDATOR.validate_source_padding(padding, self.initial_source, depth)
        self._next_source_padding = padding

    def append(self, raw_output: Tensor) -> None:
        updated_partial_block = raw_output
        if self._partial_block is not None:
            updated_partial_block = self._partial_block + raw_output
        self._partial_block = updated_partial_block
        self._partial_count += 1
        if self._partial_count == self.block_size:
            self._completed_blocks.append(self._partial_block)
            self._partial_block = None
            self._partial_count = 0
        self._next_source_padding = None

    def fork(self) -> AttentionResidualState:
        forked = AttentionResidualState(
            self.initial_source,
            block_size=self.block_size,
        )
        forked._completed_blocks = list(self._completed_blocks)
        forked._partial_block = self._partial_block
        forked._partial_count = self._partial_count
        forked._next_source_padding = self.next_source_padding
        return forked
