"""Configuration, tensor, and transformer-block contracts for sequence expansion."""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor._validation import _configuration_visits, _first_adaptive_grouping_path
from emperor.layers import Layer

if TYPE_CHECKING:
    from emperor.layers import LayerState
    from emperor.transformer.sequence_expansion._config import SequenceExpansionConfig


class SequenceExpansionValidator:
    @staticmethod
    def validate_fixed_token_output(config, operation: str) -> None:
        for visit in _configuration_visits(config, operation):
            if getattr(visit.config, "sequence_expansion_config", None) is not None:
                raise ValueError(
                    f"{operation} requires unchanged token positions and sequence "
                    "lengths, but Sequence Expansion changes them. Disable expansion "
                    "on this output stack or use an expansion-aware task."
                )

    @staticmethod
    def validate_recurrent_placement(config, *, path: str, recurrent_config) -> None:
        from emperor.layers import AdditiveResidualConfig

        for name in ("halting_config", "memory_config", "gate_config"):
            if getattr(recurrent_config, name, None) is not None:
                raise ValueError(
                    f"SequenceExpansion does not support recurrent {name} at {path}."
                )
        residual = recurrent_config.residual_config
        if residual is not None and type(residual) is not AdditiveResidualConfig:
            raise ValueError(
                "SequenceExpansion supports additive recurrent residual connections only."
            )

    @staticmethod
    def validate_config(cfg: "SequenceExpansionConfig") -> None:
        from emperor.config import ConfigBase
        from emperor.transformer.sequence_expansion._config import (
            SequenceExpansionConfig,
        )

        if not isinstance(cfg, SequenceExpansionConfig):
            raise TypeError("SequenceExpansion requires SequenceExpansionConfig.")
        for name in ("embedding_dim", "max_new_tokens"):
            if type(getattr(cfg, name)) is not int or getattr(cfg, name) <= 0:
                raise ValueError(
                    f"SequenceExpansion {name} must be a positive integer."
                )
        if cfg.model_config is None:
            raise ValueError(
                "SequenceExpansion model_config is required to predict token counts."
            )
        if not isinstance(cfg.model_config, ConfigBase):
            raise TypeError("SequenceExpansion model_config must be a ConfigBase.")
        SequenceExpansionValidator.__validate_position_config(
            cfg.positional_embedding_config
        )

    @staticmethod
    def __validate_position_config(config) -> None:
        from emperor.embedding.absolute import (
            TextLearnedPositionalEmbeddingConfig,
            TextSinusoidalPositionalEmbeddingConfig,
        )

        if config is not None and not isinstance(
            config,
            (
                TextLearnedPositionalEmbeddingConfig,
                TextSinusoidalPositionalEmbeddingConfig,
            ),
        ):
            raise TypeError(
                "SequenceExpansion positional_embedding_config must supply learned "
                "or sinusoidal text positions, or None."
            )

    @staticmethod
    def validate_model(model: object) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "SequenceExpansion model_config must build a torch.nn.Module."
            )

    @staticmethod
    def validate_prediction(
        cfg: "SequenceExpansionConfig", state: "LayerState", context: Tensor
    ) -> None:
        from emperor.layers import LayerState

        if not isinstance(state, LayerState):
            raise TypeError("SequenceExpansion count predictor must return LayerState.")
        logits = state.hidden
        if (
            not isinstance(logits, torch.Tensor)
            or logits.shape != (context.size(0), cfg.max_new_tokens)
            or not logits.is_floating_point()
            or logits.device != context.device
            or not logits.isfinite().all()
        ):
            raise ValueError(
                "SequenceExpansion count predictor must return finite floating [batch, max_new_tokens] logits on the context device."
            )
        if state.loss is not None and (
            not isinstance(state.loss, torch.Tensor)
            or not state.loss.is_floating_point()
            or state.loss.device != context.device
            or not state.loss.isfinite().all()
        ):
            raise ValueError(
                "SequenceExpansion count predictor loss must be a finite floating tensor on the context device."
            )

    @staticmethod
    def validate_inputs(
        cfg: "SequenceExpansionConfig", hidden: Tensor, padding: Tensor | None
    ) -> None:
        SequenceExpansionValidator.__validate_hidden(cfg, hidden)
        SequenceExpansionValidator.__validate_padding_mask(hidden, padding)

    @staticmethod
    def __validate_hidden(cfg: "SequenceExpansionConfig", hidden: Tensor) -> None:
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or not hidden.is_floating_point()
            or hidden.size(-1) != cfg.embedding_dim
            or hidden.size(0) == 0
            or hidden.size(1) == 0
        ):
            raise ValueError(
                "SequenceExpansion requires nonempty floating [batch, sequence, embedding_dim] input."
            )
        if not hidden.isfinite().all():
            raise ValueError("SequenceExpansion hidden input must be finite.")

    @staticmethod
    def __validate_padding_mask(hidden: Tensor, padding: Tensor | None) -> None:
        if padding is None:
            return
        if (
            not isinstance(padding, torch.Tensor)
            or padding.shape != hidden.shape[:2]
            or padding.device != hidden.device
        ):
            raise ValueError(
                "SequenceExpansion padding mask must match input shape and device."
            )
        if padding.dtype != torch.bool and (
            not padding.is_floating_point()
            or not ((padding == 0) | padding.isneginf()).all()
        ):
            raise ValueError(
                "SequenceExpansion padding mask must be Boolean or contain only 0/-inf."
            )
        if (padding != 0).all(dim=1).any():
            raise ValueError(
                "SequenceExpansion requires a valid context position in every sample."
            )

    @staticmethod
    def validate_position_offset(
        hidden: Tensor, padding_mask: Tensor, position_offset: Tensor | None
    ) -> None:
        if position_offset is not None and (
            not isinstance(position_offset, torch.Tensor)
            or position_offset.shape != (hidden.size(0),)
            or position_offset.dtype != torch.long
            or position_offset.device != hidden.device
            or (position_offset < (~padding_mask).sum(1)).any()
        ):
            raise ValueError(
                "position_offset must contain valid full-sequence token counts."
            )


class _TransformerExpansionBlockValidator(Layer.VALIDATOR):
    OPTIONAL_FIELDS = Layer.VALIDATOR.OPTIONAL_FIELDS | {"sequence_expansion_config"}
    DECODER = False

    @classmethod
    def validate(cls, model) -> None:
        super().validate(model)
        block_config = model.cfg
        expansion_config = getattr(block_config, "sequence_expansion_config", None)
        if expansion_config is None:
            return

        from emperor.transformer.sequence_expansion._config import (
            SequenceExpansionConfig,
        )

        if not isinstance(expansion_config, SequenceExpansionConfig):
            raise TypeError(
                "sequence_expansion_config must be SequenceExpansionConfig or None."
            )
        cls.__validate_transformer_layer(block_config)
        transformer_config = block_config.layer_model_config
        attention_config = (
            transformer_config.self_attention_config
            if cls.DECODER
            else transformer_config.attention_config
        )
        cls.__validate_self_attention(attention_config)
        cls.__validate_feed_forward(transformer_config.feed_forward_config)
        cross_attention_config = (
            transformer_config.cross_attention_config if cls.DECODER else None
        )
        cls.__validate_cross_attention(cross_attention_config)
        cls.__validate_token_stream(block_config)
        cls.__validate_transformer_residual(transformer_config)

    @classmethod
    def __validate_transformer_layer(cls, block_config) -> None:
        from emperor.transformer._config import (
            TransformerDecoderLayerConfig,
            TransformerEncoderLayerConfig,
        )

        transformer_config = block_config.layer_model_config
        config_type = (
            TransformerDecoderLayerConfig
            if cls.DECODER
            else TransformerEncoderLayerConfig
        )
        if not isinstance(transformer_config, config_type):
            raise ValueError(f"SequenceExpansion requires {config_type.__name__}.")
        if (
            block_config.input_dim != block_config.output_dim
            or block_config.input_dim != transformer_config.embedding_dim
        ):
            raise ValueError(
                "SequenceExpansion requires a consistent transformer embedding dimension."
            )

    @classmethod
    def __validate_self_attention(cls, attention_config) -> None:
        from emperor.attention import (
            MixerAttentionConfig,
            MixtureOfAttentionHeadsConfig,
            SelfAttentionConfig,
        )

        if isinstance(attention_config, MixerAttentionConfig):
            if (
                cls.DECODER
                or not attention_config.batch_first_flag
                or not attention_config.variable_sequence_length_flag
            ):
                raise ValueError(
                    "SequenceExpansion requires batch-first variable-length MixerAttention in an encoder."
                )
            return

        if (
            not isinstance(
                attention_config, (SelfAttentionConfig, MixtureOfAttentionHeadsConfig)
            )
            or attention_config.batch_first_flag is not True
            or (not cls.DECODER and attention_config.causal_attention_mask_flag)
        ):
            raise ValueError(
                "SequenceExpansion requires batch-first SelfAttentionConfig or MixtureOfAttentionHeadsConfig (noncausal for encoders)."
            )
        if (
            attention_config.relative_positional_embedding_config is not None
            or getattr(attention_config, "rotary_base", None) is not None
        ):
            raise ValueError(
                "SequenceExpansion does not yet support relative or rotary attention positions."
            )

    @staticmethod
    def __validate_feed_forward(feed_forward_config) -> None:
        from emperor.transformer._feed_forward import (
            FeedForwardConfig,
            SwiGLUFeedForwardConfig,
        )

        if not isinstance(
            feed_forward_config, (FeedForwardConfig, SwiGLUFeedForwardConfig)
        ):
            raise ValueError(
                "SequenceExpansion requires a pointwise feed-forward module."
            )

    @staticmethod
    def __validate_cross_attention(cross_attention_config) -> None:
        if cross_attention_config is None:
            return
        from emperor.attention import (
            IndependentAttentionConfig,
            MixtureOfAttentionHeadsConfig,
        )

        if (
            not isinstance(
                cross_attention_config,
                (IndependentAttentionConfig, MixtureOfAttentionHeadsConfig),
            )
            or cross_attention_config.batch_first_flag is not True
        ):
            raise ValueError(
                "SequenceExpansion requires batch-first IndependentAttentionConfig or MixtureOfAttentionHeadsConfig for cross-attention."
            )
        if (
            cross_attention_config.relative_positional_embedding_config is not None
            or getattr(cross_attention_config, "rotary_base", None) is not None
        ):
            raise ValueError(
                "SequenceExpansion does not yet support relative or rotary cross-attention positions."
            )

    @classmethod
    def __validate_token_stream(cls, block_config) -> None:
        # These constraints protect the expanding token stream. The predictor
        # consumes one pooled vector per sample and owns its own model validation.
        token_block_config = replace(block_config, sequence_expansion_config=None)
        grouping_path = _first_adaptive_grouping_path(
            token_block_config, root=type(block_config).__name__
        )
        if grouping_path is not None:
            raise ValueError(
                f"SequenceExpansion does not support adaptive parameter grouping at {grouping_path}."
            )
        for visit in _configuration_visits(
            token_block_config, type(block_config).__name__
        ):
            cls.__validate_token_component(visit.config, token_block_config, visit.path)

    @classmethod
    def __validate_token_component(cls, config, token_block_config, path: str) -> None:
        from emperor.experts import (
            MixtureOfExpertsConfig,
            MixtureOfExpertsLayerConfig,
            MixtureOfExpertsModelConfig,
        )
        from emperor.layers import LayerConfig, RecurrentCompositionConfig
        from emperor.linears import LinearLayerConfig
        from emperor.transformer._feed_forward import FeedForwardConfig

        if isinstance(config, RecurrentCompositionConfig):
            raise ValueError(
                "SequenceExpansion does not support recurrent attention/feed-forward components."
            )
        if isinstance(config, LayerConfig):
            cls.__validate_layer_composition(config, path)
        if isinstance(config, FeedForwardConfig):
            from emperor.layers import LayerStackConfig

            if not isinstance(
                config.stack_config, (LayerStackConfig, MixtureOfExpertsModelConfig)
            ):
                raise ValueError(
                    "SequenceExpansion requires a pointwise feed-forward stack or mixture of experts."
                )
        if isinstance(config, LayerConfig) and config is not token_block_config:
            allowed_model_types = (
                (LinearLayerConfig, MixtureOfExpertsConfig)
                if isinstance(config, MixtureOfExpertsLayerConfig)
                else (LinearLayerConfig,)
            )
            if config.layer_model_config is not None and not isinstance(
                config.layer_model_config, allowed_model_types
            ):
                raise ValueError(
                    "SequenceExpansion requires linear, adaptive linear, or expert projection/feed-forward stacks."
                )
        for name in (
            "shared_halting_config",
            "shared_memory_config",
            "shared_gate_config",
        ):
            if getattr(config, name, None) is not None:
                raise ValueError(
                    f"SequenceExpansion does not support {name} at {path}."
                )

    @staticmethod
    def __validate_layer_composition(config, path: str) -> None:
        from emperor.layers import AdditiveResidualConfig

        for name in ("halting_config", "memory_config", "gate_config"):
            if getattr(config, name) is not None:
                raise ValueError(
                    f"SequenceExpansion does not support {name} at {path}."
                )
        if (
            config.residual_config is not None
            and type(config.residual_config) is not AdditiveResidualConfig
        ):
            raise ValueError(
                "SequenceExpansion supports additive residual connections only."
            )

    @staticmethod
    def __validate_transformer_residual(transformer_config) -> None:
        from emperor.layers import AdditiveResidualConfig

        if (
            transformer_config.residual_config is not None
            and type(transformer_config.residual_config) is not AdditiveResidualConfig
        ):
            raise ValueError(
                "SequenceExpansion supports additive residual connections only."
            )

    @classmethod
    def validate_expansion_state(cls, state: "LayerState") -> None:
        from emperor.transformer._state import (
            TransformerDecoderLayerState,
            TransformerEncoderLayerState,
        )

        state_type = (
            TransformerDecoderLayerState
            if cls.DECODER
            else TransformerEncoderLayerState
        )
        if not isinstance(state, state_type):
            raise TypeError(
                f"SequenceExpansion requires {state_type.__name__} to preserve expansion records."
            )
        if cls.DECODER:
            cls._validate_decoder_masks(state)
        elif state.attention_mask is not None:
            raise ValueError(
                "SequenceExpansion supports padding masks only, not pairwise attention masks."
            )
        if state.halting_state is not None or state.residual_state is not None:
            raise ValueError(
                "SequenceExpansion cannot resize existing halting or residual history state."
            )

    @staticmethod
    def _validate_decoder_masks(state) -> None:
        if not isinstance(state.hidden, torch.Tensor) or state.hidden.ndim != 3:
            raise ValueError(
                "SequenceExpansion requires floating [batch, sequence, embedding_dim] input."
            )
        target_sequence_length = state.hidden.size(1)
        _TransformerExpansionBlockValidator.__validate_target_attention_mask(
            state, target_sequence_length
        )
        _TransformerExpansionBlockValidator.__validate_cross_attention_mask(
            state, target_sequence_length
        )

    @staticmethod
    def __validate_target_attention_mask(state, target_sequence_length: int) -> None:
        target_mask = state.target_attention_mask
        if target_mask is not None:
            if (
                not isinstance(target_mask, torch.Tensor)
                or target_mask.ndim not in (2, 3)
                or target_mask.shape[-2:]
                != (target_sequence_length, target_sequence_length)
            ):
                raise ValueError(
                    "SequenceExpansion target_attention_mask must match the current target length."
                )
            causal_mask = torch.ones(
                target_sequence_length,
                target_sequence_length,
                device=target_mask.device,
                dtype=torch.bool,
            ).triu(1)
            expected_causal_mask = (
                causal_mask
                if target_mask.dtype == torch.bool
                else torch.zeros_like(
                    causal_mask, dtype=state.hidden.dtype
                ).masked_fill(causal_mask, -torch.inf)
            )
            if (
                target_mask.dtype != torch.bool and not target_mask.is_floating_point()
            ) or not torch.equal(
                target_mask, expected_causal_mask.expand_as(target_mask)
            ):
                raise ValueError(
                    "SequenceExpansion can resize only standard causal target_attention_mask values; use None for configured attention masking."
                )

    @staticmethod
    def __validate_cross_attention_mask(state, target_sequence_length: int) -> None:
        cross_mask = state.cross_attention_mask
        if cross_mask is not None:
            encoder_output = state.encoder_output
            if (
                not isinstance(cross_mask, torch.Tensor)
                or cross_mask.ndim not in (2, 3)
                or encoder_output is None
                or cross_mask.shape[-2:]
                != (target_sequence_length, encoder_output.size(1))
            ):
                raise ValueError(
                    "SequenceExpansion cross_attention_mask must match target and encoder lengths."
                )
            if (
                cross_mask.dtype != torch.bool and not cross_mask.is_floating_point()
            ) or not torch.equal(
                cross_mask, cross_mask[..., :1, :].expand_as(cross_mask)
            ):
                raise ValueError(
                    "SequenceExpansion cross_attention_mask must use identical source restrictions for every target query."
                )
            if (
                cross_mask.is_floating_point()
                and not (cross_mask.isfinite() | cross_mask.isneginf()).all()
            ):
                raise ValueError(
                    "SequenceExpansion cross_attention_mask must contain finite values or -inf."
                )


class TransformerEncoderBlockValidator(_TransformerExpansionBlockValidator):
    pass


class TransformerDecoderBlockValidator(_TransformerExpansionBlockValidator):
    DECODER = True
