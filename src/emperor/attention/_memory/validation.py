from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import _adaptive_grouping_configs, _configuration_visits

if TYPE_CHECKING:
    from emperor.attention._config import MultiHeadAttentionConfig
    from emperor.attention._runtime import MultiHeadAttentionInputs
    from emperor.config import ConfigBase


class AttentionSourceMemoryValidator:
    @staticmethod
    def validate_config(cfg: "MultiHeadAttentionConfig") -> None:
        widths = (
            cfg.embedding_dim,
            cfg.query_key_projection_dim or cfg.embedding_dim,
            cfg.value_projection_dim or cfg.embedding_dim,
        )
        if len(set(widths)) != 1:
            raise ValueError(
                "Attention source memory requires equal input, key and value widths."
            )
        if cfg.source_memory_config is None:
            raise ValueError(
                "Attention source memory requires source_memory_config with chunk_size grouping."
            )

    @classmethod
    def validate_context_config(cls, config: object) -> None:
        from emperor.attention._config import AttentionSourceMemoryConfig
        from emperor.augmentations.adaptive_parameters import (
            AdaptiveParameterGroupingScopeOptions,
            GroupingConfig,
        )

        if not isinstance(config, AttentionSourceMemoryConfig):
            raise TypeError(
                "source_memory_config must be an AttentionSourceMemoryConfig."
            )
        grouping_config = config.grouping_config
        if not isinstance(grouping_config, GroupingConfig):
            raise TypeError("Attention memory requires a concrete grouping_config.")
        if grouping_config.group_count is not None:
            raise ValueError(
                "Attention memory supports chunk_size only, not group_count."
            )
        if grouping_config.scope not in (
            None,
            AdaptiveParameterGroupingScopeOptions.SEQUENCE,
        ):
            raise ValueError("Attention memory grouping must use SEQUENCE scope.")
        cls.validate_context_model(getattr(grouping_config, "model_config", None))

    @staticmethod
    def validate_context_model(config: "ConfigBase | None") -> None:
        """Scorers and generators must keep context rows independent."""
        from emperor.attention import MixerAttentionConfig, MultiHeadAttentionConfig
        from emperor.experts import MixtureOfExpertsConfig
        from emperor.layers import (
            LayerConfig,
            LayerStackConfig,
            RecurrentCompositionConfig,
        )
        from emperor.linears import LinearLayerConfig

        if config is not None and not isinstance(
            config,
            (
                LinearLayerConfig,
                LayerConfig,
                LayerStackConfig,
                RecurrentCompositionConfig,
            ),
        ):
            raise ValueError(
                "Attention memory models must preserve independent rows; use a linear, layer, layer-stack, or recurrent-layer configuration."
            )
        if any(_adaptive_grouping_configs(config, root="source_memory.model_config")):
            raise ValueError(
                "Attention memory models require independent rows; adaptive parameter grouping is unsupported inside their generators or scorers."
            )
        for visit in _configuration_visits(config, "source_memory.model_config"):
            if isinstance(
                visit.config,
                (
                    MultiHeadAttentionConfig,
                    MixerAttentionConfig,
                    MixtureOfExpertsConfig,
                ),
            ):
                raise ValueError(
                    "Attention memory generators and grouping models must preserve independent rows; attention and expert routing can mix contexts."
                )

    @classmethod
    def validate_generator_config(cls, config: object) -> None:
        from emperor.config import ConfigBase

        if not isinstance(config, ConfigBase):
            raise TypeError("Attention memory model_config must be a ConfigBase.")
        cls.validate_context_model(config)

    @staticmethod
    def validate_generator_output(
        memory_embeddings: Tensor, generator_inputs: Tensor
    ) -> None:
        if memory_embeddings.shape != generator_inputs.shape:
            raise ValueError(
                "Attention memory generator must preserve row count and embedding width."
            )


    @staticmethod
    def validate_inputs(inputs: "MultiHeadAttentionInputs") -> None:
        if inputs.static_key is not None or inputs.static_value is not None:
            raise ValueError(
                "Attention source memory does not support static key/value inputs."
            )
        if inputs.query is not inputs.key or inputs.key is not inputs.value:
            raise ValueError(
                "Attention source memory requires self-attention over the same input tensor."
            )


    @staticmethod
    def validate_projected_output(
        memory: Tensor, key_projection: Tensor, *, memory_length: int
    ) -> None:
        expected = (memory_length, *key_projection.shape[1:])
        if not isinstance(memory, Tensor) or tuple(memory.shape) != expected:
            raise ValueError(
                "Attention source memory projection must match the chunk count, "
                "batch size, key/value expert count and feature width."
            )
