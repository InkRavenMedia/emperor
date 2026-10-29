from collections.abc import Sequence
from dataclasses import fields

import torch
from torch import Tensor

from emperor.config import ConfigBase
from emperor.embedding.absolute import (
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.embedding.hierarchical import END_OF_DOCUMENT, END_OF_TOKEN
from emperor.embedding.hierarchical.encoder._config import (
    HierarchicalByteEmbeddingConfig,
)
from emperor.experts import MixtureOfExpertsLayerConfig, MixtureOfExpertsModelConfig
from emperor.layers import LayerConfig, LayerStackConfig, RecurrentCompositionConfig


def _config_items(config: object, path: str):
    if isinstance(config, ConfigBase):
        yield path, config
        for field in fields(config):
            yield from _config_items(
                getattr(config, field.name), f"{path}.{field.name}"
            )


def _byte_attention_paths(config, path: str):
    if not isinstance(config, ConfigBase):
        return
    if hasattr(config, "attention_config") and hasattr(config, "feed_forward_config"):
        yield f"{path}.attention_config"
        return
    for name in (
        "encoder_stack_config",
        "layer_config",
        "layer_model_config",
        "block_config",
        "high_block_config",
        "low_block_config",
    ):
        yield from _byte_attention_paths(getattr(config, name, None), f"{path}.{name}")


class HierarchicalByteEmbeddingValidator:
    @staticmethod
    def validate_and_encode_token(token, index, token_count, max_token_bytes) -> bytes:
        location = f"token_texts[{index // token_count}][{index % token_count}]"
        try:
            token_bytes = token.encode("utf-8") if isinstance(token, str) else token
        except UnicodeEncodeError as error:
            raise ValueError(f"{location} is not a valid UTF-8 string") from error
        token_byte_count = len(token_bytes)
        if END_OF_TOKEN in token_bytes or END_OF_DOCUMENT in token_bytes:
            raise ValueError(f"{location} must not contain reserved control symbols")
        if token_byte_count > max_token_bytes:
            raise ValueError(
                f"{location} contains {token_byte_count} UTF-8 bytes; "
                f"max_token_bytes is {max_token_bytes}"
            )
        return token_bytes

    @staticmethod
    def validate_token_group_sizes(token_groups, max_group_tokens) -> None:
        for byte_length, token_group in token_groups.items():
            group_token_count = len(token_group)
            if group_token_count > max_group_tokens:
                raise ValueError(
                    f"{group_token_count} tokens with {byte_length} UTF-8 bytes exceed encoder "
                    f"batch_size bound {max_group_tokens}"
                )

    @staticmethod
    def validate_config_types(cfg, overrides) -> None:
        if not isinstance(cfg, HierarchicalByteEmbeddingConfig):
            raise TypeError("cfg must be HierarchicalByteEmbeddingConfig")
        if overrides is not None and not isinstance(
            overrides, HierarchicalByteEmbeddingConfig
        ):
            raise TypeError("overrides must be HierarchicalByteEmbeddingConfig or None")

    @classmethod
    def resolve_config(cls, cfg: HierarchicalByteEmbeddingConfig) -> int:
        for name in ("byte_embedding_dim", "output_dim", "max_token_bytes"):
            cls._positive_integer(getattr(cfg, name), name)
        for name in ("encoder_config", "projection_config"):
            if not isinstance(getattr(cfg, name), ConfigBase):
                raise TypeError(f"{name} must be a supplied ConfigBase configuration")
        cls._validate_independence(cfg)
        cls._resolve_positions(cfg)
        batch_bound = cls._resolve_encoder(cfg)
        cls._resolve_projection(cfg)
        return batch_bound

    @classmethod
    def _resolve_positions(cls, cfg: HierarchicalByteEmbeddingConfig) -> None:
        position = cfg.byte_position_config
        if position is None:
            attentions = [
                child
                for _, child in _config_items(cfg.encoder_config, "encoder")
                if hasattr(child, "batch_first_flag")
            ]
            if attentions and all(
                getattr(child, "rotary_base", None) is not None for child in attentions
            ):
                return
            raise TypeError(
                "byte_position_config may be None only with rotary byte attention"
            )
        if not isinstance(
            position,
            (
                TextLearnedPositionalEmbeddingConfig,
                TextSinusoidalPositionalEmbeddingConfig,
            ),
        ):
            raise TypeError(
                "byte_position_config must configure learned or sinusoidal text positions"
            )
        cls._dimension(
            position, "embedding_dim", cfg.byte_embedding_dim, "byte_position_config"
        )
        cls._positive_integer(
            position.num_embeddings, "byte_position_config.num_embeddings"
        )
        if position.num_embeddings < cfg.max_token_bytes + 1:
            raise ValueError(
                "byte_position_config.num_embeddings must cover max_token_bytes + 1"
            )
        if position.padding_idx is not None:
            raise ValueError(
                "byte_position_config.padding_idx must be None; every byte and [W] is real"
            )

    @classmethod
    def _resolve_encoder(cls, cfg: HierarchicalByteEmbeddingConfig) -> int:
        encoder_config = cfg.encoder_config
        encoder_stack = getattr(encoder_config, "encoder_stack_config", None)
        cls._validate_encoder_stack(encoder_config, encoder_stack)
        cls._resolve_encoder_stack_dimensions(encoder_stack, cfg.byte_embedding_dim)
        batch_bounds = []
        for config_path, child_config in _config_items(
            encoder_config, "encoder_config"
        ):
            if hasattr(child_config, "attention_config") and hasattr(
                child_config, "feed_forward_config"
            ):
                cls._dimension(
                    child_config, "embedding_dim", cfg.byte_embedding_dim, config_path
                )
            if not hasattr(child_config, "batch_first_flag"):
                continue
            batch_bound = cls._resolve_byte_attention(child_config, config_path, cfg)
            batch_bounds.append(batch_bound)
        if not batch_bounds:
            raise ValueError(
                "encoder_config requires a byte-attention encoder with explicit batch_first_flag"
            )
        return min(batch_bounds)

    @staticmethod
    def _validate_encoder_stack(encoder_config, encoder_stack):
        if not isinstance(encoder_stack, ConfigBase) or not hasattr(
            encoder_config, "decoder_stack_config"
        ):
            raise TypeError(
                "encoder_config must supply an encoder-only Transformer configuration"
            )
        if encoder_config.decoder_stack_config is not None:
            raise ValueError(
                "encoder_config must be encoder-only (decoder_stack_config=None)"
            )

    @classmethod
    def _resolve_encoder_stack_dimensions(cls, encoder_stack, byte_embedding_dim):
        stack_path = "encoder_config.encoder_stack_config"
        for dimension_name in ("input_dim", "output_dim"):
            cls._dimension(
                encoder_stack, dimension_name, byte_embedding_dim, stack_path
            )
        if isinstance(encoder_stack, LayerStackConfig):
            cls._dimension(encoder_stack, "hidden_dim", byte_embedding_dim, stack_path)

    @classmethod
    def _resolve_byte_attention(cls, attention_config, config_path, cfg):
        if attention_config.batch_first_flag is not True:
            raise ValueError(f"{config_path}.batch_first_flag must explicitly be True")
        if getattr(attention_config, "causal_attention_mask_flag", None) is not False:
            raise ValueError(
                f"{config_path}.causal_attention_mask_flag must be False for bidirectional byte attention"
            )
        cls._dimension(
            attention_config, "embedding_dim", cfg.byte_embedding_dim, config_path
        )
        batch_size_path = f"{config_path}.batch_size"
        cls._positive_integer(attention_config.batch_size, batch_size_path)
        required_positions = cfg.max_token_bytes + 1
        for dimension_name in ("source_sequence_length", "target_sequence_length"):
            position_limit = getattr(attention_config, dimension_name, None)
            dimension_path = f"{config_path}.{dimension_name}"
            cls._positive_integer(position_limit, dimension_path)
            if position_limit < required_positions:
                raise ValueError(f"{dimension_path} must cover max_token_bytes + 1")
        return attention_config.batch_size

    @classmethod
    def _resolve_projection(cls, cfg: HierarchicalByteEmbeddingConfig) -> None:
        projection = cfg.projection_config
        if not isinstance(
            projection,
            (
                LayerConfig,
                LayerStackConfig,
                RecurrentCompositionConfig,
                MixtureOfExpertsModelConfig,
            ),
        ) or isinstance(projection, MixtureOfExpertsLayerConfig):
            raise TypeError(
                "projection_config must build a LayerState model; wrap tensor layers in LayerConfig"
            )
        cls._dimension(
            projection, "input_dim", cfg.byte_embedding_dim, "projection_config"
        )
        cls._dimension(projection, "output_dim", cfg.output_dim, "projection_config")
        if (
            isinstance(projection, MixtureOfExpertsModelConfig)
            and projection.stack_config is not None
        ):
            cls._dimension(
                projection.stack_config,
                "input_dim",
                cfg.byte_embedding_dim,
                "projection_config.stack_config",
            )
            cls._dimension(
                projection.stack_config,
                "output_dim",
                cfg.output_dim,
                "projection_config.stack_config",
            )

    @staticmethod
    def _validate_independence(cfg: ConfigBase) -> None:
        attention_paths = set(
            _byte_attention_paths(cfg.encoder_config, "hierarchical.encoder_config")
        )
        for path, child in _config_items(cfg, "hierarchical"):
            if hasattr(child, "batch_first_flag") and path not in attention_paths:
                raise ValueError(
                    f"{path}: attention is only supported on the byte encoder sequence axis"
                )
            for name in (
                "memory_config",
                "shared_memory_config",
                "context_config",
                "grouping_config",
                "token_sampler_config",
            ):
                if getattr(child, name, None) is not None:
                    raise ValueError(
                        f"{path}.{name} must be None to preserve independent tokens"
                    )
            if hasattr(child, "capacity_factor") and child.capacity_factor != 0.0:
                raise ValueError(
                    f"{path}.capacity_factor must be 0.0 to disable batch-dependent dropping"
                )
            if isinstance(child, RecurrentCompositionConfig):
                if getattr(child, "sampler_config", None) is not None:
                    raise ValueError(
                        f"{path}.sampler_config must be None; token selection can couple independent rows"
                    )

    @staticmethod
    def _positive_integer(value, path: str) -> None:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{path} must be a supplied positive integer")

    @staticmethod
    def _dimension(config, name: str, expected: int, path: str) -> None:
        if not hasattr(config, name):
            raise TypeError(f"{path} must declare {name}")
        value = getattr(config, name)
        if value is None:
            setattr(config, name, expected)
        elif type(value) is not int or value != expected:
            raise ValueError(f"{path}.{name} must equal {expected}, received {value}")

    @staticmethod
    def validate_forward_inputs(token_texts, attention_mask) -> tuple[int, int]:
        if not isinstance(token_texts, Sequence) or isinstance(
            token_texts, (str, bytes)
        ):
            raise TypeError("token_texts must be a rectangular sequence of token rows")
        if not token_texts:
            raise ValueError(
                "token_texts must have nonempty batch and token dimensions"
            )
        token_count = None
        for batch_index, row in enumerate(token_texts):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError(
                    f"token_texts row {batch_index} must be a sequence of str"
                )
            if not row:
                raise ValueError("token_texts must have nonempty token rows")
            if token_count is None:
                token_count = len(row)
            if len(row) != token_count:
                raise ValueError("token_texts must be rectangular")
            for token_index, token in enumerate(row):
                if not isinstance(token, (str, bytes)):
                    raise TypeError(
                        f"token_texts[{batch_index}][{token_index}] must be str or raw bytes"
                    )
        shape = (len(token_texts), token_count)
        if attention_mask is not None:
            if not isinstance(attention_mask, Tensor):
                raise TypeError("attention_mask must be a Tensor or None")
            if attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must use torch.bool (True means valid)")
            if tuple(attention_mask.shape) != shape:
                raise ValueError(f"attention_mask must have shape {shape}")
        return shape
