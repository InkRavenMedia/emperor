import math
from collections.abc import Sequence
from dataclasses import fields

import torch
from torch import Tensor

from emperor.config import ConfigBase
from emperor.embedding.absolute import (
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.embedding.hierarchical import (
    END_OF_DOCUMENT,
    END_OF_TOKEN,
    HierarchicalByteDecoderConfig,
    HierarchicalByteEmbeddingConfig,
)
from emperor.embedding.hierarchical.decoder import OUTPUT_SYMBOLS, ByteGenerationOptions
from emperor.layers import LayerConfig, LayerStackConfig, RecurrentCompositionConfig

from ._config import HierarchicalLanguageModelConfig


def _config_items(config, path):
    if isinstance(config, ConfigBase):
        yield path, config
        for field in fields(config):
            yield from _config_items(
                getattr(config, field.name), f"{path}.{field.name}"
            )


class HierarchicalLanguageModelValidator:
    @staticmethod
    def validate_config_types(cfg, overrides):
        if not isinstance(cfg, HierarchicalLanguageModelConfig):
            raise TypeError("cfg must be HierarchicalLanguageModelConfig")
        if overrides is not None and not isinstance(
            overrides, HierarchicalLanguageModelConfig
        ):
            raise TypeError("overrides must be HierarchicalLanguageModelConfig or None")

    @classmethod
    def resolve_config(cls, cfg):
        cls._positive_integer(cfg.sequence_length, "sequence_length")
        if not isinstance(cfg.embedding_config, HierarchicalByteEmbeddingConfig):
            raise TypeError("embedding_config must be HierarchicalByteEmbeddingConfig")
        if not isinstance(cfg.decoding_config, HierarchicalByteDecoderConfig):
            raise TypeError("decoding_config must be HierarchicalByteDecoderConfig")
        context_dimension = cfg.embedding_config.output_dim
        cls._dimension(
            cfg.decoding_config,
            "byte_embedding_dim",
            cfg.embedding_config.byte_embedding_dim,
            "decoding_config",
        )
        cls._positive_integer(context_dimension, "embedding_config.output_dim")
        cls._dimension(
            cfg.decoding_config,
            "conditioning_dim",
            context_dimension,
            "decoding_config",
        )
        if cfg.embedding_config.max_token_bytes != cfg.decoding_config.max_token_bytes:
            raise ValueError("encoder and decoder max_token_bytes must agree")
        if not isinstance(
            cfg.backbone_config, (LayerStackConfig, RecurrentCompositionConfig)
        ):
            raise TypeError("backbone_config must supply a Transformer decoder stack")
        cls._validate_independence(cfg)
        batch_bounds = cls._resolve_backbone(cfg, context_dimension)
        cls._resolve_backbone_positions(cfg, context_dimension)
        if cfg.dropout_probability is None or not 0 <= cfg.dropout_probability < 1:
            raise ValueError("dropout_probability must be in [0, 1)")
        cls._resolve_normalizations(cfg, context_dimension)
        return min(batch_bounds)

    @classmethod
    def _resolve_backbone(cls, cfg, context_dimension):
        for dimension_name in ("input_dim", "output_dim"):
            cls._dimension(
                cfg.backbone_config,
                dimension_name,
                context_dimension,
                "backbone_config",
            )
        batch_bounds = []
        for config_path, child_config in _config_items(
            cfg.backbone_config, "backbone_config"
        ):
            if getattr(child_config, "cross_attention_config", None) is not None:
                raise ValueError("backbone cross-attention is not supported")
            if not hasattr(child_config, "batch_first_flag"):
                continue
            batch_bound = cls._resolve_backbone_attention(
                child_config, config_path, cfg, context_dimension
            )
            batch_bounds.append(batch_bound)
        if not batch_bounds:
            raise ValueError("backbone_config requires causal self-attention")
        return batch_bounds

    @classmethod
    def _resolve_backbone_attention(
        cls, attention_config, config_path, cfg, context_dimension
    ):
        if (
            not config_path.endswith(".self_attention_config")
            or attention_config.batch_first_flag is not True
            or attention_config.causal_attention_mask_flag is not True
        ):
            raise ValueError(
                f"{config_path} requires batch-first causal self-attention"
            )
        cls._dimension(
            attention_config, "embedding_dim", context_dimension, config_path
        )
        batch_size_path = f"{config_path}.batch_size"
        cls._positive_integer(attention_config.batch_size, batch_size_path)
        for dimension_name in ("source_sequence_length", "target_sequence_length"):
            position_limit = getattr(attention_config, dimension_name)
            if position_limit < cfg.sequence_length:
                raise ValueError(
                    f"{config_path}.{dimension_name} must cover sequence_length"
                )
        return attention_config.batch_size

    @classmethod
    def _resolve_backbone_positions(cls, cfg, context_dimension):
        if cfg.position_config is None:
            attentions = [
                child
                for _, child in _config_items(cfg.backbone_config, "backbone")
                if hasattr(child, "batch_first_flag")
            ]
            if attentions and all(
                getattr(child, "rotary_base", None) is not None for child in attentions
            ):
                return
            raise TypeError(
                "position_config may be None only with rotary backbone attention"
            )
        if not isinstance(
            cfg.position_config,
            (
                TextLearnedPositionalEmbeddingConfig,
                TextSinusoidalPositionalEmbeddingConfig,
            ),
        ):
            raise TypeError("position_config must supply text positions")
        cls._dimension(
            cfg.position_config, "embedding_dim", context_dimension, "position_config"
        )
        if cfg.position_config.num_embeddings < cfg.sequence_length:
            raise ValueError("position_config must cover all real backbone positions")

    @classmethod
    def _resolve_normalizations(cls, cfg, context_dimension):
        for config_name in (
            "embedding_normalization_config",
            "output_normalization_config",
        ):
            normalization_config = getattr(cfg, config_name)
            if normalization_config is not None:
                if not isinstance(normalization_config, LayerConfig):
                    raise TypeError(f"{config_name} must be a LayerConfig or None")
                cls._dimension(
                    normalization_config, "input_dim", context_dimension, config_name
                )
                cls._dimension(
                    normalization_config, "output_dim", context_dimension, config_name
                )

    @staticmethod
    def validate_batch(batch, sequence_length):
        from ._batch import HierarchicalLanguageModelBatch

        if not isinstance(batch, HierarchicalLanguageModelBatch):
            raise TypeError("hierarchical mode requires HierarchicalLanguageModelBatch")
        mask, bos = batch.attention_mask, batch.bos_mask
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.ndim != 2:
            raise TypeError("attention_mask must be a rank-2 bool tensor")
        if min(mask.shape) <= 0 or mask.shape[1] > sequence_length:
            raise ValueError(
                "batch dimensions must be nonempty and fit sequence_length"
            )
        if not isinstance(bos, Tensor):
            raise TypeError("bos_mask must be a bool tensor")
        if (
            bos.dtype != torch.bool
            or bos.shape != mask.shape
            or bool((bos.to(mask.device) & ~mask).any())
        ):
            raise ValueError(
                "bos_mask must be bool, match attention_mask, and select valid positions"
            )
        if bool((~mask[:, :-1] & mask[:, 1:]).any()):
            raise ValueError("attention_mask must use right padding")
        if bool(bos[:, 1:].any()):
            raise ValueError("BOS is only valid at the beginning of a window")
        contexts = batch.context_texts
        if (
            not isinstance(contexts, Sequence)
            or isinstance(contexts, str)
            or len(contexts) != mask.shape[0]
            or any(
                not isinstance(row, Sequence)
                or isinstance(row, str)
                or len(row) != mask.shape[1]
                for row in contexts
            )
        ):
            raise ValueError("context_texts must match the batch/token mask shape")
        for row, column in bos.nonzero().tolist():
            if contexts[row][column] not in ("", b""):
                raise ValueError("BOS context must use the empty string placeholder")

    @staticmethod
    def validate_backbone_batch_size(group_row_count, max_batch):
        if group_row_count > max_batch:
            raise ValueError("backbone batch_size bound exceeded")

    @staticmethod
    def validate_continuation(result, prompt_is_complete):
        if prompt_is_complete:
            raise ValueError("continuation already records its token boundaries")
        if result.stop_reason == "end_of_document":
            raise ValueError("cannot continue a terminated document")
        if (
            not isinstance(result.completed_tokens, tuple)
            or any(not isinstance(token, bytes) for token in result.completed_tokens)
            or not isinstance(result.partial_token, bytes)
        ):
            raise TypeError("continuation tokens must be raw bytes")
        if b"".join(result.completed_tokens) + result.partial_token != result.data:
            raise ValueError(
                "continuation bytes must agree with its recorded token boundaries"
            )

    @staticmethod
    def validate_text_generation_limits(
        max_new_tokens, max_new_bytes, prompt_is_complete
    ):
        for budget_name, budget in (
            ("max_new_tokens", max_new_tokens),
            ("max_new_bytes", max_new_bytes),
        ):
            if type(budget) is not int or budget < 0:
                raise ValueError(f"{budget_name} must be a non-negative integer")
        if type(prompt_is_complete) is not bool:
            raise TypeError("prompt_is_complete must be bool")

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


class HierarchicalLanguageModelBatchValidator:
    @staticmethod
    def validate_examples(examples):
        if not examples:
            raise ValueError("collation requires at least one example")
        maximum_tokens = max(len(example["context_texts"]) for example in examples)
        if not maximum_tokens:
            raise ValueError("examples require at least one context/target pair")
        return maximum_tokens

    @staticmethod
    def validate_example(context_texts, target_texts, beginning_mask):
        if (
            not context_texts
            or len(context_texts) != len(target_texts)
            or len(context_texts) != len(beginning_mask)
        ):
            raise ValueError(
                "context_texts, target_texts, and bos_mask must have equal nonzero lengths"
            )
        if any(not isinstance(text, (str, bytes)) for text in context_texts) or any(
            type(is_beginning) is not bool for is_beginning in beginning_mask
        ):
            raise TypeError(
                "contexts must be strings or bytes and BOS values must be bools"
            )

    @staticmethod
    def validate_and_encode_targets(context_texts, target_texts):
        try:
            encoded_targets = [
                None
                if text is None
                else text
                if isinstance(text, bytes)
                else text.encode("utf-8")
                for text in target_texts
            ]
            encoded_contexts = [
                text if isinstance(text, bytes) else text.encode("utf-8")
                for text in context_texts
            ]
        except (AttributeError, UnicodeEncodeError) as error:
            raise ValueError(
                "contexts and targets must be valid UTF-8 strings (None targets mean EOS)"
            ) from error
        if any(
            END_OF_TOKEN in value or END_OF_DOCUMENT in value
            for value in [*encoded_contexts, *encoded_targets]
            if value is not None
        ):
            raise ValueError(
                "contexts and target bytes must not contain reserved control symbols"
            )
        return encoded_targets


class HierarchicalTextCodecValidator:
    @staticmethod
    def validate_max_token_bytes(max_token_bytes):
        if type(max_token_bytes) is not int or max_token_bytes <= 0:
            raise ValueError("max_token_bytes must be a positive integer")

    @staticmethod
    def validate_text(text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")

    @staticmethod
    def validate_character_byte_count(character, max_token_bytes):
        try:
            encoded_character = character.encode("utf-8")
            character_byte_count = len(encoded_character)
        except UnicodeEncodeError as error:
            raise ValueError("text must be valid UTF-8") from error
        if character_byte_count > max_token_bytes:
            raise ValueError("a Unicode character exceeds max_token_bytes")
        return character_byte_count

    @staticmethod
    def validate_sequence_length(sequence_length):
        if type(sequence_length) is not int or sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")
