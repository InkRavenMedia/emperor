from dataclasses import dataclass, field

from emperor.config import ConfigBase, optional_field
from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.layers import NormalizationOptions
from emperor.linears import LinearLayerConfig
from emperor.neuron import NeuronClusterConfig
from emperor.patch import PatchConfig


@dataclass
class ExperimentConfig(ConfigBase):
    encoder_output_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    patch_config: PatchConfig | None = optional_field(
        "Config for image patch extraction and projection."
    )
    positional_embedding_config: AbsolutePositionalEmbeddingConfig | None = (
        optional_field("Config for image positional embeddings added to patches.")
    )
    neuron_cluster_config: NeuronClusterConfig | None = optional_field(
        "Cluster with a complete ViT encoder inside each nucleus."
    )
    output_config: LinearLayerConfig | None = optional_field(
        "Config for the [CLS] classification head."
    )
