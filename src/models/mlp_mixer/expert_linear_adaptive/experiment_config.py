from dataclasses import dataclass, field

from emperor.config import ConfigBase, optional_field
from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
from emperor.embedding.hierarchical import (
    HierarchicalByteDecoderConfig,
    HierarchicalByteEmbeddingConfig,
)
from emperor.layers import (
    LayerConfig,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentCompositionConfig,
)
from emperor.patch import PatchConfig


@dataclass
class ExperimentConfig(ConfigBase):
    encoder_output_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    patch_config: PatchConfig | None = optional_field(
        "Config for image patch projection and its optional class token."
    )
    encoder_config: LayerStackConfig | RecurrentCompositionConfig | None = (
        optional_field(
            "Config for the generic Transformer encoder stack of Mixer blocks."
        )
    )
    output_config: LayerConfig | None = optional_field(
        "Config for the classification head applied to the image readout."
    )


@dataclass(kw_only=True)
class MaskedWordExperimentConfig(ConfigBase):
    """Package-owned text boundary; independent of the image classifier."""

    embedding_config: HierarchicalByteEmbeddingConfig
    masked_word_decoder_config: HierarchicalByteDecoderConfig
    positional_embedding_config: TextLearnedPositionalEmbeddingConfig
    encoder_config: LayerStackConfig | RecurrentCompositionConfig
    encoder_output_normalization: NormalizationOptions
    embedding_dropout_probability: float
