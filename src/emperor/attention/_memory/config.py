"""Configuration for temporary chunk-generated attention sources."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters import GroupingConfig


@dataclass
class AttentionSourceMemoryConfig(ConfigBase):
    model_config: ConfigBase | None = optional_field(
        "Independent row-preserving memory generator. Defaults to a learned linear projection."
    )
    grouping_config: "GroupingConfig | None" = optional_field(
        "Per-sequence chunk_size grouping; generate one memory embedding per reduced chunk. Runtime inputs determine layout and length."
    )
