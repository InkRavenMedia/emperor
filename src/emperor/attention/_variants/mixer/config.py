"""Private mixer-attention configuration implementation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.attention._memory.config import AttentionSourceMemoryConfig
    from emperor.experts import MixtureOfExpertsModelConfig
    from emperor.layers import LayerStackConfig, RecurrentCompositionConfig
    from emperor.memory import DynamicMemoryConfig


@dataclass
class MixerAttentionConfig(ConfigBase):
    embedding_dim: int | None = optional_field(
        "Exact feature width of each token passed to the mixer."
    )
    sequence_length: int | None = optional_field(
        "Maximum runtime token count and fixed token-mixing width. "
        "Shorter inputs are padded internally and outputs retain the input length."
    )
    batch_first_flag: bool | None = optional_field(
        "Explicit input layout. True selects [batch, sequence, embedding]; "
        "False selects [sequence, batch, embedding]."
    )
    mixing_model_config: (
        "LayerStackConfig | MixtureOfExpertsModelConfig | "
        "RecurrentCompositionConfig | None"
    ) = optional_field(
        "Model configuration built with sequence_length as both its input and "
        "output dimensions, extended by the configured source-memory capacity."
    )
    source_memory_config: "DynamicMemoryConfig | AttentionSourceMemoryConfig | None" = optional_field(
        "Optional per-token dynamic memory or chunk-compressed attention source memory. "
        "All positions are mixed, then memory output positions are discarded."
    )

    def _registry_owner(self) -> type:
        from emperor.attention._variants.mixer.layer import MixerAttention

        return MixerAttention
