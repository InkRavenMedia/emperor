from __future__ import annotations

from functools import partial
from typing import Literal, overload

import torch

from emperor.attention import (
    IndependentAttentionConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
)
from emperor.transformer import (
    FeedForwardConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from models import _attention_source_memory as _source_memory
from models import _sequence_expansion
from models._residual_stack import residual_source_from_settings
from models.transformer._adaptive_augmentation import AdaptiveAugmentationFactory
from models.transformer._experiment_builder import (
    assemble_experiment,
    expanded_stacks,
    position_builder,
)
from models.transformer._stack import (
    build_controlled_stack,
    wrap_recurrent_stack,
)
from models.transformer._stack import build_gate as _gate
from models.transformer.linear_adaptive._adaptive_linear_template import (
    AdaptiveLinearLayerTemplateConfig,
)

from ._residual import (
    ResidualStackOptions,
    ResidualStackSource,
    build_residual_config,
    resolve_residual_stack_options,
)
from ._transformer_submodule import (
    configure_transformer_submodule,
)
from .experiment_config import ExperimentConfig
from .runtime_options import (
    AdaptiveParameterOptions,
    RuntimeOptions,
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
    TransformerStackOptions,
    resolve_controller_stack_options,
)


def _adaptive_augmentation(
    options: AdaptiveParameterOptions,
    residual_stack_options: ResidualStackOptions,
):
    factory = AdaptiveAugmentationFactory(
        resolve_controller_stack_options, residual_stack_options
    )
    return factory.build(options)


def _residual_stack(runtime: RuntimeOptions) -> ResidualStackOptions:
    return resolve_residual_stack_options(
        residual_source_from_settings(runtime, ResidualStackSource),
        runtime.encoder_attention_options.stack_options,
    )


def _adaptive_stack(
    *,
    stack_options,
    adaptive_options: AdaptiveParameterOptions,
    residual_stack_options: ResidualStackOptions,
) -> LayerStackConfig:
    residual_config = build_residual_config(
        stack_options.residual_connection_option,
        stack_options.residual_model_flag,
        residual_stack_options,
        residual_block_size=stack_options.residual_block_size,
        residual_rms_norm_epsilon=stack_options.residual_rms_norm_epsilon,
    )
    adaptive_augmentation_config = _adaptive_augmentation(
        adaptive_options,
        residual_stack_options,
    )
    layer_model_config = AdaptiveLinearLayerTemplateConfig(
        bias_flag=stack_options.bias_flag,
        adaptive_augmentation_config=adaptive_augmentation_config,
    )
    layer_config = LayerConfig(
        activation=stack_options.activation,
        residual_config=residual_config,
        dropout_probability=stack_options.dropout_probability,
        layer_norm_position=stack_options.layer_norm_position,
        normalization=stack_options.normalization,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=layer_model_config,
    )
    return LayerStackConfig(
        hidden_dim=stack_options.hidden_dim,
        num_layers=stack_options.num_layers,
        apply_output_postprocessing_flag=(
            stack_options.apply_output_postprocessing_flag
        ),
        last_layer_bias_option=stack_options.last_layer_bias_option,
        layer_config=layer_config,
    )


@overload
def _attention_config(
    runtime: RuntimeOptions,
    options: TransformerAttentionOptions,
    adaptive_options: AdaptiveParameterOptions,
    *,
    target_length: int,
    source_length: int,
    causal: bool,
    independent: Literal[False] = False,
) -> SelfAttentionConfig: ...


@overload
def _attention_config(
    runtime: RuntimeOptions,
    options: TransformerAttentionOptions,
    adaptive_options: AdaptiveParameterOptions,
    *,
    target_length: int,
    source_length: int,
    causal: bool,
    independent: Literal[True],
) -> IndependentAttentionConfig: ...


def _attention_config(
    runtime: RuntimeOptions,
    options: TransformerAttentionOptions,
    adaptive_options: AdaptiveParameterOptions,
    *,
    target_length: int,
    source_length: int,
    causal: bool,
    independent: bool = False,
) -> SelfAttentionConfig | IndependentAttentionConfig:
    if options.source_memory_flag and independent:
        raise ValueError(
            "Source memory is supported only for self-attention, not decoder cross-attention."
        )
    residual_stack_options = _residual_stack(runtime)
    projection_stack = _adaptive_stack(
        stack_options=options.stack_options,
        adaptive_options=adaptive_options,
        residual_stack_options=residual_stack_options,
    )
    projection_config = configure_transformer_submodule(
        projection_stack,
        control_stack=projection_stack,
        path_options=options,
        model_dim=runtime.model_dim,
        residual_stack_options=residual_stack_options,
    )
    source_memory_config = _source_memory.build_source_memory_config(
        options, projection_stack
    )
    config_type = IndependentAttentionConfig
    if not independent:
        config_type = partial(
            SelfAttentionConfig,
            projection_strategy=SelfAttentionProjectionStrategy.SEPARATE,
        )
    return config_type(
        batch_size=runtime.batch_size,
        num_heads=options.num_heads,
        embedding_dim=runtime.model_dim,
        query_key_projection_dim=runtime.model_dim,
        value_projection_dim=runtime.model_dim,
        target_sequence_length=target_length,
        source_sequence_length=source_length,
        target_dtype=torch.float32,
        dropout_probability=runtime.dropout_probability,
        zero_attention_flag=options.zero_attention_flag,
        causal_attention_mask_flag=causal,
        add_key_value_bias_flag=options.add_key_value_bias_flag,
        average_attention_weights_flag=True,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=projection_config,
        relative_positional_embedding_config=None,
        source_memory_flag=options.source_memory_flag,
        source_memory_config=source_memory_config,
    )


def _feed_forward(
    runtime: RuntimeOptions,
    options: TransformerFeedForwardOptions,
    adaptive_options: AdaptiveParameterOptions,
):
    residual_stack_options = _residual_stack(runtime)
    stack = _adaptive_stack(
        stack_options=options.stack_options,
        adaptive_options=adaptive_options,
        residual_stack_options=residual_stack_options,
    )
    stack_config = configure_transformer_submodule(
        stack,
        control_stack=stack,
        path_options=options,
        model_dim=runtime.model_dim,
        residual_stack_options=residual_stack_options,
    )
    return FeedForwardConfig(
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        stack_config=stack_config,
    )


def _controlled_stack(
    runtime: RuntimeOptions, options: TransformerStackOptions, layer_config: LayerConfig
):
    return build_controlled_stack(
        runtime,
        options,
        layer_config,
        resolve_residual_stack=_residual_stack,
        wrap_recurrent=partial(wrap_recurrent_stack, runtime, options),
    )


def _encoder(runtime: RuntimeOptions):
    options = runtime.encoder_options
    residual_config = AdditiveResidualConfig()
    attention_config = _attention_config(
        runtime,
        runtime.encoder_attention_options,
        runtime.encoder_attention_adaptive_options,
        target_length=runtime.source_sequence_length,
        source_length=runtime.source_sequence_length,
        causal=False,
    )
    feed_forward_config = _feed_forward(
        runtime,
        runtime.encoder_feed_forward_options,
        runtime.encoder_feed_forward_adaptive_options,
    )
    inner = TransformerEncoderLayerConfig(
        embedding_dim=runtime.model_dim,
        layer_norm_position=options.layer_norm_position,
        normalization=options.normalization,
        dropout_probability=runtime.dropout_probability,
        residual_config=residual_config,
        attention_config=attention_config,
        feed_forward_config=feed_forward_config,
    )
    residual_stack_options = _residual_stack(runtime)
    encoder_residual_config = build_residual_config(
        options.stack_residual_connection_option,
        options.stack_residual_model_flag,
        residual_stack_options,
        residual_block_size=options.stack_residual_block_size,
        residual_rms_norm_epsilon=options.stack_residual_rms_norm_epsilon,
    )
    gate_config = _gate(runtime.model_dim, options.stack_gate_flag)
    sequence_expansion_config = _sequence_expansion.build_sequence_expansion_config(
        options, runtime.model_dim
    )
    layer = TransformerEncoderBlockLayerConfig(
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        activation=ActivationOptions.DISABLED,
        residual_config=encoder_residual_config,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        normalization=NormalizationOptions.RMS_NORM,
        gate_config=gate_config,
        memory_config=None,
        layer_model_config=inner,
        sequence_expansion_config=sequence_expansion_config,
    )
    return _controlled_stack(runtime, options, layer)


def _decoder(runtime: RuntimeOptions):
    options = runtime.decoder_options
    residual_config = AdditiveResidualConfig()
    self_attention_config = _attention_config(
        runtime,
        runtime.decoder_self_attention_options,
        runtime.decoder_self_attention_adaptive_options,
        target_length=runtime.target_sequence_length,
        source_length=runtime.target_sequence_length,
        causal=True,
    )
    cross_attention_config = _attention_config(
        runtime,
        runtime.decoder_cross_attention_options,
        runtime.decoder_cross_attention_adaptive_options,
        target_length=runtime.target_sequence_length,
        source_length=runtime.source_sequence_length,
        causal=False,
        independent=True,
    )
    feed_forward_config = _feed_forward(
        runtime,
        runtime.decoder_feed_forward_options,
        runtime.decoder_feed_forward_adaptive_options,
    )
    inner = TransformerDecoderLayerConfig(
        embedding_dim=runtime.model_dim,
        layer_norm_position=options.layer_norm_position,
        normalization=options.normalization,
        dropout_probability=runtime.dropout_probability,
        residual_config=residual_config,
        self_attention_config=self_attention_config,
        cross_attention_config=cross_attention_config,
        feed_forward_config=feed_forward_config,
    )
    residual_stack_options = _residual_stack(runtime)
    decoder_residual_config = build_residual_config(
        options.stack_residual_connection_option,
        options.stack_residual_model_flag,
        residual_stack_options,
        residual_block_size=options.stack_residual_block_size,
        residual_rms_norm_epsilon=options.stack_residual_rms_norm_epsilon,
    )
    gate_config = _gate(runtime.model_dim, options.stack_gate_flag)
    sequence_expansion_config = _sequence_expansion.build_sequence_expansion_config(
        options, runtime.model_dim
    )
    layer = TransformerDecoderBlockLayerConfig(
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        activation=ActivationOptions.DISABLED,
        residual_config=decoder_residual_config,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        normalization=NormalizationOptions.RMS_NORM,
        gate_config=gate_config,
        memory_config=None,
        layer_model_config=inner,
        sequence_expansion_config=sequence_expansion_config,
    )
    return _controlled_stack(runtime, options, layer)


def build_experiment_config(runtime: RuntimeOptions) -> ExperimentConfig:
    build_position = position_builder(runtime)
    encoder, decoder = expanded_stacks(runtime, _encoder, _decoder)
    source_position = build_position(runtime.source_sequence_length)
    target_position = build_position(runtime.target_sequence_length)
    return assemble_experiment(
        runtime, ExperimentConfig, encoder, decoder, source_position, target_position
    )
