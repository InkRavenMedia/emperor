from __future__ import annotations

from dataclasses import replace
from functools import partial

import torch

from emperor.attention import MixtureOfAttentionHeadsConfig
from emperor.experts import (
    MixtureOfExpertsLayerConfig,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
)
from emperor.sampler import RouterConfig, SamplerConfig
from emperor.transformer import (
    FeedForwardConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from models import _attention_source_memory as _source_memory
from models import _sequence_expansion
from models._expert_mixture import build_expert_mixture, build_expert_model
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
from models.transformer.expert_linear_adaptive._adaptive_linear_template import (
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
    ExpertOptions,
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
    hidden_dim: int | None = None,
    num_layers: int = 1,
    bias_flag: bool = True,
    activation: ActivationOptions = ActivationOptions.RELU,
    dropout_probability: float = 0.0,
    adaptive_options: AdaptiveParameterOptions,
    residual_stack_options: ResidualStackOptions,
    stack_options=None,
):
    if stack_options is not None:
        hidden_dim = stack_options.hidden_dim
        num_layers = stack_options.num_layers
        bias_flag = stack_options.bias_flag
        activation = stack_options.activation
        dropout_probability = stack_options.dropout_probability
        apply_output_postprocessing_flag = (
            stack_options.apply_output_postprocessing_flag
        )
        last_layer_bias_option = stack_options.last_layer_bias_option
        residual_connection_option = stack_options.residual_connection_option
        residual_model_flag = stack_options.residual_model_flag
        residual_block_size = stack_options.residual_block_size
        residual_rms_norm_epsilon = stack_options.residual_rms_norm_epsilon
        layer_norm_position = stack_options.layer_norm_position
        normalization = stack_options.normalization
    else:
        apply_output_postprocessing_flag = False
        last_layer_bias_option = LastLayerBiasOptions.DEFAULT
        residual_connection_option = None
        residual_model_flag = False
        residual_block_size = None
        residual_rms_norm_epsilon = None
        layer_norm_position = LayerNormPositionOptions.DISABLED
        normalization = NormalizationOptions.RMS_NORM
    residual_config = build_residual_config(
        residual_connection_option,
        residual_model_flag,
        residual_stack_options,
        residual_block_size=residual_block_size,
        residual_rms_norm_epsilon=residual_rms_norm_epsilon,
    )
    adaptive_augmentation_config = _adaptive_augmentation(
        adaptive_options,
        residual_stack_options,
    )
    layer_model_config = AdaptiveLinearLayerTemplateConfig(
        bias_flag=bias_flag,
        adaptive_augmentation_config=adaptive_augmentation_config,
    )
    layer_config = LayerConfig(
        activation=activation,
        residual_config=residual_config,
        dropout_probability=dropout_probability,
        layer_norm_position=layer_norm_position,
        normalization=normalization,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=layer_model_config,
    )
    return LayerStackConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        apply_output_postprocessing_flag=apply_output_postprocessing_flag,
        last_layer_bias_option=last_layer_bias_option,
        layer_config=layer_config,
    )


def _sampler(
    runtime: RuntimeOptions,
    input_dim: int,
    options: ExpertOptions,
) -> SamplerConfig:
    residual_stack_options = _residual_stack(runtime)
    router_stack = _adaptive_stack(
        stack_options=options.router_path_options.stack_options,
        adaptive_options=runtime.router_adaptive_options,
        residual_stack_options=residual_stack_options,
    )
    model_config = configure_transformer_submodule(
        router_stack,
        control_stack=router_stack,
        path_options=options.router_path_options,
        model_dim=runtime.model_dim,
        residual_stack_options=residual_stack_options,
    )
    router = RouterConfig(
        input_dim=input_dim,
        num_experts=options.num_experts,
        noisy_topk_flag=options.router_noisy_topk_flag,
        model_config=model_config,
    )
    return SamplerConfig(
        top_k=options.top_k,
        threshold=options.sampler_threshold,
        filter_above_threshold=options.sampler_filter_above_threshold,
        num_topk_samples=options.sampler_num_topk_samples,
        normalize_probabilities_flag=options.normalize_probabilities_flag,
        noisy_topk_flag=options.sampler_noisy_topk_flag,
        num_experts=options.num_experts,
        coefficient_of_variation_loss_weight=(
            options.coefficient_of_variation_loss_weight
        ),
        switch_loss_weight=options.switch_loss_weight,
        zero_centred_loss_weight=options.zero_centred_loss_weight,
        mutual_information_loss_weight=options.mutual_information_loss_weight,
        router_config=router,
    )


def _experts_config(
    runtime: RuntimeOptions,
    input_dim: int,
    output_dim: int,
    options: ExpertOptions,
    adaptive_options: AdaptiveParameterOptions,
    *,
    bias_flag: bool = True,
):
    expert_options = options.expert_path_options.stack_options
    stack_options = replace(
        expert_options, bias_flag=bias_flag and expert_options.bias_flag
    )
    residual_stack_options = _residual_stack(runtime)
    expert_stack = _adaptive_stack(
        stack_options=stack_options,
        adaptive_options=adaptive_options,
        residual_stack_options=residual_stack_options,
    )
    sampler_config = _sampler(runtime, input_dim, options)
    expert_model_config = configure_transformer_submodule(
        expert_stack,
        control_stack=expert_stack,
        path_options=options.expert_path_options,
        model_dim=runtime.model_dim,
        residual_stack_options=residual_stack_options,
    )
    return build_expert_mixture(
        options,
        input_dim=input_dim,
        output_dim=output_dim,
        sampler_config=sampler_config,
        expert_model_config=expert_model_config,
    )


def _attention(
    runtime: RuntimeOptions,
    attention_options: TransformerAttentionOptions,
    projection_adaptive_options: AdaptiveParameterOptions,
    attention_expert_adaptive_options: AdaptiveParameterOptions,
    *,
    target_length: int,
    source_length: int,
    causal: bool,
    self_attention: bool,
):
    if attention_options.source_memory_flag and not (self_attention):
        raise ValueError(
            "Source memory is supported only for self-attention, not decoder cross-attention."
        )
    experts = runtime.attention_expert_options
    residual_stack_options = _residual_stack(runtime)
    projection_stack = _adaptive_stack(
        stack_options=attention_options.stack_options,
        adaptive_options=projection_adaptive_options,
        residual_stack_options=residual_stack_options,
    )
    projection_config = configure_transformer_submodule(
        projection_stack,
        control_stack=projection_stack,
        path_options=attention_options,
        model_dim=runtime.model_dim,
        residual_stack_options=residual_stack_options,
    )
    experts_config = _experts_config(
        runtime,
        runtime.model_dim,
        runtime.model_dim,
        experts,
        attention_expert_adaptive_options,
    )
    source_memory_config = _source_memory.build_source_memory_config(
        attention_options, experts_config
    )
    use_kv_expert_models = self_attention
    if experts.use_kv_expert_models_flag is not None:
        use_kv_expert_models = experts.use_kv_expert_models_flag
    return MixtureOfAttentionHeadsConfig(
        batch_size=runtime.batch_size,
        num_heads=attention_options.num_heads,
        embedding_dim=runtime.model_dim,
        query_key_projection_dim=runtime.model_dim,
        value_projection_dim=runtime.model_dim,
        target_sequence_length=target_length,
        source_sequence_length=source_length,
        target_dtype=torch.float32,
        dropout_probability=runtime.dropout_probability,
        zero_attention_flag=attention_options.zero_attention_flag,
        causal_attention_mask_flag=causal,
        add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
        average_attention_weights_flag=True,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=projection_config,
        relative_positional_embedding_config=None,
        source_memory_flag=attention_options.source_memory_flag,
        source_memory_config=source_memory_config,
        experts_config=experts_config,
        use_kv_expert_models_flag=use_kv_expert_models,
    )


def _expert_feed_forward(
    runtime: RuntimeOptions,
    feed_forward_options: TransformerFeedForwardOptions,
    adaptive_options: AdaptiveParameterOptions,
):
    options = runtime.feed_forward_expert_options
    stack_options = feed_forward_options.stack_options
    leaf = _experts_config(
        runtime,
        runtime.model_dim,
        runtime.model_dim,
        options,
        adaptive_options,
        bias_flag=stack_options.bias_flag,
    )
    residual_stack_options = _residual_stack(runtime)
    residual_config = build_residual_config(
        stack_options.residual_connection_option,
        stack_options.residual_model_flag,
        residual_stack_options,
        residual_block_size=stack_options.residual_block_size,
        residual_rms_norm_epsilon=stack_options.residual_rms_norm_epsilon,
    )
    layer_config = MixtureOfExpertsLayerConfig(
        activation=stack_options.activation,
        residual_config=residual_config,
        dropout_probability=stack_options.dropout_probability,
        layer_norm_position=stack_options.layer_norm_position,
        normalization=stack_options.normalization,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=leaf,
    )
    stack = LayerStackConfig(
        input_dim=runtime.model_dim,
        hidden_dim=stack_options.hidden_dim,
        output_dim=runtime.model_dim,
        num_layers=stack_options.num_layers,
        apply_output_postprocessing_flag=(
            stack_options.apply_output_postprocessing_flag
        ),
        last_layer_bias_option=stack_options.last_layer_bias_option,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=layer_config,
    )
    sampler_config = _sampler(runtime, runtime.model_dim, options)
    model = build_expert_model(
        options,
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        sampler_config=sampler_config,
        stack_config=stack,
    )
    configured = configure_transformer_submodule(
        model,
        control_stack=stack,
        path_options=feed_forward_options,
        model_dim=runtime.model_dim,
        residual_stack_options=residual_stack_options,
    )
    return FeedForwardConfig(
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        stack_config=configured,
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
    attention_config = _attention(
        runtime,
        runtime.encoder_attention_options,
        runtime.encoder_attention_adaptive_options,
        runtime.encoder_attention_expert_adaptive_options
        or runtime.attention_expert_adaptive_options,
        target_length=runtime.source_sequence_length,
        source_length=runtime.source_sequence_length,
        causal=False,
        self_attention=True,
    )
    feed_forward_config = _expert_feed_forward(
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
    self_attention_config = _attention(
        runtime,
        runtime.decoder_self_attention_options,
        runtime.decoder_self_attention_adaptive_options,
        runtime.decoder_self_attention_expert_adaptive_options
        or runtime.attention_expert_adaptive_options,
        target_length=runtime.target_sequence_length,
        source_length=runtime.target_sequence_length,
        causal=True,
        self_attention=True,
    )
    cross_attention_config = _attention(
        runtime,
        runtime.decoder_cross_attention_options,
        runtime.decoder_cross_attention_adaptive_options,
        runtime.decoder_cross_attention_expert_adaptive_options
        or runtime.attention_expert_adaptive_options,
        target_length=runtime.target_sequence_length,
        source_length=runtime.source_sequence_length,
        causal=False,
        self_attention=False,
    )
    feed_forward_config = _expert_feed_forward(
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
