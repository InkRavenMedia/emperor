import math
from dataclasses import fields

import torch
from torch import Tensor

from emperor.config import ConfigBase
from emperor.embedding.absolute import (
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.embedding.hierarchical import END_OF_DOCUMENT, END_OF_TOKEN, OUTPUT_SYMBOLS
from emperor.experts import MixtureOfExpertsLayerConfig, MixtureOfExpertsModelConfig
from emperor.layers import LayerConfig, LayerStackConfig, RecurrentCompositionConfig

from ._config import HierarchicalByteDecoderConfig
from ._records import (
    ByteGenerationOptions,
)

_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _config_items(config, path):
    if isinstance(config, ConfigBase):
        yield path, config
        for field in fields(config):
            yield from _config_items(
                getattr(config, field.name), f"{path}.{field.name}"
            )


class HierarchicalByteGenerationValidator:
    @staticmethod
    def validate_options(options):
        if not isinstance(options, ByteGenerationOptions):
            raise TypeError("options must be ByteGenerationOptions")
        if type(options.do_sample) is not bool:
            raise TypeError("do_sample must be a bool")
        if type(options.constrain_utf8) is not bool:
            raise TypeError("constrain_utf8 must be a bool")
        if not math.isfinite(options.temperature) or options.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if options.top_k is not None and (
            type(options.top_k) is not int or not 1 <= options.top_k <= OUTPUT_SYMBOLS
        ):
            raise ValueError("top_k must be None or an integer from 1 through 256")
        if not math.isfinite(options.top_p) or not 0 < options.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if options.seed is not None and type(options.seed) is not int:
            raise TypeError("seed must be an integer or None")

    @staticmethod
    def validate_and_encode_prefix(prefix, max_token_bytes):
        if not isinstance(prefix, (str, bytes)):
            raise TypeError("prefix must be a UTF-8 string or raw bytes")
        try:
            encoded_prefix = (
                prefix.encode("utf-8") if isinstance(prefix, str) else prefix
            )
            prefix_bytes = bytearray(encoded_prefix)
        except UnicodeEncodeError as error:
            raise ValueError("prefix must be valid UTF-8") from error
        if len(prefix_bytes) > max_token_bytes:
            raise ValueError("prefix exceeds max_token_bytes")
        if END_OF_TOKEN in prefix_bytes or END_OF_DOCUMENT in prefix_bytes:
            raise ValueError("prefix must not contain reserved control symbols")
        return prefix_bytes

    @staticmethod
    def validate_byte_budget(max_new_bytes):
        if type(max_new_bytes) is not int or max_new_bytes < 0:
            raise ValueError("max_new_bytes must be a non-negative integer")

    @staticmethod
    def validate_conditioning(conditioning, conditioning_dim):
        if not isinstance(conditioning, Tensor) or conditioning.shape != (
            conditioning_dim,
        ):
            raise ValueError("conditioning must have shape [conditioning_dim]")

    @staticmethod
    def validate_logits(logits):
        if not torch.isfinite(logits).all():
            raise ValueError("generation requires finite decoder logits")


class HierarchicalByteDecoderValidator(HierarchicalByteGenerationValidator):
    @staticmethod
    def validate_prefix_group_sizes(selected_byte_lengths, max_group_tokens):
        for byte_length in selected_byte_lengths.unique().tolist():
            group_token_count = int((selected_byte_lengths == byte_length).sum())
            if group_token_count > max_group_tokens:
                raise ValueError(
                    f"{group_token_count} prefixes with {byte_length} bytes exceed decoder "
                    f"batch_size bound {max_group_tokens}"
                )

    @staticmethod
    def validate_config_types(cfg, overrides):
        if not isinstance(cfg, HierarchicalByteDecoderConfig):
            raise TypeError("cfg must be HierarchicalByteDecoderConfig")
        if overrides is not None and not isinstance(
            overrides, HierarchicalByteDecoderConfig
        ):
            raise TypeError("overrides must be HierarchicalByteDecoderConfig or None")

    @classmethod
    def resolve_config(cls, cfg):
        for name in ("conditioning_dim", "byte_embedding_dim", "max_token_bytes"):
            cls._positive_integer(getattr(cfg, name), name)
        for name in (
            "decoder_config",
            "conditioning_projection_config",
            "output_projection_config",
        ):
            if not isinstance(getattr(cfg, name), ConfigBase):
                raise TypeError(f"{name} must be a supplied ConfigBase configuration")
        cls._resolve_positions(cfg)
        cls._resolve_projection(
            cfg.conditioning_projection_config,
            cfg.conditioning_dim,
            cfg.byte_embedding_dim,
            "conditioning_projection_config",
        )
        cls._resolve_projection(
            cfg.output_projection_config,
            cfg.byte_embedding_dim,
            OUTPUT_SYMBOLS,
            "output_projection_config",
        )
        cls._validate_independence(cfg)
        return cls._resolve_decoder(cfg)

    @classmethod
    def _resolve_positions(cls, cfg):
        position = cfg.byte_position_config
        if position is None:
            attentions = [
                child
                for _, child in _config_items(cfg.decoder_config, "decoder")
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
                "byte_position_config must supply learned or sinusoidal text positions"
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
                "byte_position_config.padding_idx must be None; all positions are real"
            )

    @classmethod
    def _resolve_projection(cls, projection, input_dim, output_dim, path):
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
                f"{path} must build a LayerState model; wrap tensor layers in LayerConfig"
            )
        cls._dimension(projection, "input_dim", input_dim, path)
        cls._dimension(projection, "output_dim", output_dim, path)
        if (
            isinstance(projection, MixtureOfExpertsModelConfig)
            and projection.stack_config is not None
        ):
            cls._dimension(
                projection.stack_config, "input_dim", input_dim, f"{path}.stack_config"
            )
            cls._dimension(
                projection.stack_config,
                "output_dim",
                output_dim,
                f"{path}.stack_config",
            )

    @classmethod
    def _resolve_decoder(cls, cfg):
        decoder_config = cfg.decoder_config
        decoder_stack = getattr(decoder_config, "decoder_stack_config", None)
        cls._validate_decoder_stack(decoder_config, decoder_stack)
        cls._resolve_decoder_stack_dimensions(decoder_stack, cfg.byte_embedding_dim)
        allowed_attention_paths = cls._resolve_decoder_attention_paths(
            decoder_config, cfg
        )
        batch_bounds = []
        for config_path, child_config in _config_items(cfg, "hierarchical"):
            if not hasattr(child_config, "batch_first_flag"):
                continue
            batch_bound = cls._resolve_byte_attention(
                child_config, config_path, allowed_attention_paths, cfg
            )
            batch_bounds.append(batch_bound)
        if not batch_bounds:
            raise ValueError(
                "decoder_config requires causal byte attention with explicit batch_first_flag"
            )
        return min(batch_bounds)

    @staticmethod
    def _validate_decoder_stack(decoder_config, decoder_stack):
        if not isinstance(decoder_stack, ConfigBase) or not hasattr(
            decoder_config, "encoder_stack_config"
        ):
            raise TypeError(
                "decoder_config must supply a decoder-only Transformer configuration"
            )
        if decoder_config.encoder_stack_config is not None:
            raise ValueError(
                "decoder_config must be decoder-only (encoder_stack_config=None)"
            )

    @classmethod
    def _resolve_decoder_stack_dimensions(cls, decoder_stack, byte_embedding_dim):
        stack_path = "decoder_config.decoder_stack_config"
        for dimension_name in ("input_dim", "output_dim"):
            cls._dimension(
                decoder_stack, dimension_name, byte_embedding_dim, stack_path
            )
        if isinstance(decoder_stack, LayerStackConfig):
            cls._dimension(decoder_stack, "hidden_dim", byte_embedding_dim, stack_path)

    @classmethod
    def _resolve_decoder_attention_paths(cls, decoder_config, cfg):
        allowed_attention_paths = set()
        for config_path, child_config in _config_items(
            decoder_config, "decoder_config"
        ):
            if hasattr(child_config, "self_attention_config") and hasattr(
                child_config, "feed_forward_config"
            ):
                cls._dimension(
                    child_config, "embedding_dim", cfg.byte_embedding_dim, config_path
                )
                if child_config.cross_attention_config is not None:
                    raise ValueError(
                        f"{config_path}.cross_attention_config must be None; conditioning is a prefix"
                    )
                allowed_attention_paths.add(f"{config_path}.self_attention_config")
        return allowed_attention_paths

    @classmethod
    def _resolve_byte_attention(cls, attention_config, config_path, allowed_paths, cfg):
        relative_path = config_path.removeprefix("hierarchical.")
        if relative_path not in allowed_paths:
            raise ValueError(
                f"{config_path}: attention is only supported on the decoder byte sequence axis"
            )
        if attention_config.batch_first_flag is not True:
            raise ValueError(f"{config_path}.batch_first_flag must explicitly be True")
        if getattr(attention_config, "causal_attention_mask_flag", None) is not True:
            raise ValueError(f"{config_path}.causal_attention_mask_flag must be True")
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

    @staticmethod
    def _validate_independence(cfg):
        for path, child in _config_items(cfg, "hierarchical"):
            for name in (
                "memory_config",
                "shared_memory_config",
                "context_config",
                "grouping_config",
                "token_sampler_config",
            ):
                if getattr(child, name, None) is not None:
                    raise ValueError(
                        f"{path}.{name} must be None to preserve independent causal tokens"
                    )
            if hasattr(child, "capacity_factor") and child.capacity_factor != 0.0:
                raise ValueError(
                    f"{path}.capacity_factor must be 0.0 to disable batch-dependent dropping"
                )
            if isinstance(child, RecurrentCompositionConfig):
                if getattr(child, "sampler_config", None) is not None:
                    raise ValueError(f"{path}.sampler_config must be None")

    @staticmethod
    def _positive_integer(value, path):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{path} must be a supplied positive integer")

    @staticmethod
    def _dimension(config, name, expected, path):
        if not hasattr(config, name):
            raise TypeError(f"{path} must declare {name}")
        value = getattr(config, name)
        if value is None:
            setattr(config, name, expected)
        elif type(value) is not int or value != expected:
            raise ValueError(f"{path}.{name} must equal {expected}, received {value}")

    @classmethod
    def validate_forward_inputs(
        cls, cfg, conditioning, prefixes, lengths, mask, device
    ):
        cls._validate_conditioning_tensor(cfg, conditioning, device)
        batch_shape = conditioning.shape[:2]
        cls._validate_prefix_tensor(prefixes, batch_shape)
        cls._validate_length_tensor(lengths, batch_shape)
        mask = cls._resolve_attention_mask(mask, lengths.shape, device)
        byte_lengths = lengths.to(device=device, dtype=torch.long)
        byte_prefix_ids = prefixes.to(device=device, dtype=torch.long)
        attention_mask = mask.to(device=device)
        cls._validate_active_prefixes(
            cfg, byte_prefix_ids, byte_lengths, attention_mask, device
        )
        return byte_prefix_ids, byte_lengths, attention_mask

    @staticmethod
    def _validate_conditioning_tensor(cfg, conditioning, device):
        if (
            not isinstance(conditioning, Tensor)
            or conditioning.ndim != 3
            or not conditioning.is_floating_point()
        ):
            raise TypeError(
                "conditioning must be a floating [batch, tokens, conditioning_dim] tensor"
            )
        if (
            min(conditioning.shape[:2]) <= 0
            or conditioning.shape[-1] != cfg.conditioning_dim
        ):
            raise ValueError(
                "conditioning must have nonempty batch/token dimensions and configured conditioning_dim"
            )
        if conditioning.device != device:
            raise ValueError("conditioning must be on the decoder device")

    @staticmethod
    def _validate_prefix_tensor(byte_prefix_ids, batch_shape):
        if (
            not isinstance(byte_prefix_ids, Tensor)
            or byte_prefix_ids.ndim != 3
            or byte_prefix_ids.dtype not in _INTEGER_DTYPES
        ):
            raise TypeError(
                "byte_prefix_ids must be an integer [batch, tokens, bytes] tensor"
            )
        if byte_prefix_ids.shape[:2] != batch_shape:
            raise ValueError(
                "byte_prefix_ids must share conditioning batch/token dimensions"
            )

    @staticmethod
    def _validate_length_tensor(byte_lengths, batch_shape):
        if (
            not isinstance(byte_lengths, Tensor)
            or byte_lengths.dtype not in _INTEGER_DTYPES
        ):
            raise TypeError("byte_lengths must be an integer tensor")
        if byte_lengths.shape != batch_shape:
            raise ValueError("byte_lengths must have shape [batch, tokens]")

    @staticmethod
    def _resolve_attention_mask(attention_mask, batch_shape, device):
        if attention_mask is None:
            return torch.ones(batch_shape, dtype=torch.bool, device=device)
        if not isinstance(attention_mask, Tensor) or attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be a bool tensor (True means valid)")
        if attention_mask.shape != batch_shape:
            raise ValueError("attention_mask must have shape [batch, tokens]")
        return attention_mask

    @staticmethod
    def _validate_active_prefixes(
        cfg, byte_prefix_ids, byte_lengths, attention_mask, device
    ):
        valid_lengths = byte_lengths[attention_mask]
        maximum_prefix_length = min(byte_prefix_ids.shape[-1], cfg.max_token_bytes)
        invalid_lengths = (valid_lengths < 0) | (valid_lengths > maximum_prefix_length)
        if bool(invalid_lengths.any()):
            raise ValueError(
                "valid byte_lengths must fit byte_prefix_ids and max_token_bytes"
            )
        byte_positions = torch.arange(byte_prefix_ids.shape[-1], device=device)
        valid_byte_positions = byte_positions < byte_lengths.unsqueeze(-1)
        valid_byte_mask = attention_mask.unsqueeze(-1) & valid_byte_positions
        valid_bytes = byte_prefix_ids[valid_byte_mask]
        invalid_bytes = (
            (valid_bytes < 0)
            | (valid_bytes >= OUTPUT_SYMBOLS)
            | (valid_bytes == END_OF_TOKEN)
            | (valid_bytes == END_OF_DOCUMENT)
        )
        if bool(invalid_bytes.any()):
            raise ValueError(
                "valid byte_prefix_ids must contain byte values excluding reserved controls"
            )
