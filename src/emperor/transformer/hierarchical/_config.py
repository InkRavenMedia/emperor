from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field
from emperor.embedding.hierarchical import HierarchicalByteDecoderConfig


@dataclass
class HierarchicalLanguageModelConfig(ConfigBase):
    """Supplied encoder, causal backbone and byte head; no word vocabulary.

    Adds a learned beginning-of-document context. Architecture choices remain
    owned by the supplied configurations, not inferred from the HAT paper.
    """

    sequence_length: int | None = optional_field("Maximum rolling backbone context.")
    embedding_config: ConfigBase | None = optional_field(
        "Hierarchical byte encoder configuration."
    )
    backbone_config: ConfigBase | None = optional_field(
        "Causal Transformer decoder stack configuration."
    )
    decoding_config: HierarchicalByteDecoderConfig | None = optional_field(
        "Conditional byte decoder configuration."
    )
    position_config: ConfigBase | None = optional_field(
        "Positions on the backbone token sequence."
    )
    embedding_normalization_config: ConfigBase | None = optional_field(
        "Optional LayerConfig for input normalization."
    )
    output_normalization_config: ConfigBase | None = optional_field(
        "Optional LayerConfig for backbone output normalization."
    )
    dropout_probability: float | None = optional_field(
        "Dropout on backbone input embeddings."
    )

    def _registry_owner(self) -> type:
        from ._language_model import HierarchicalLanguageModel

        return HierarchicalLanguageModel
