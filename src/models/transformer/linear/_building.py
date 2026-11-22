from __future__ import annotations

from emperor.attention import (
    IndependentAttentionConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.halting import HaltingConfig
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    GateConfig,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentCompositionConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
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
from models.transformer._experiment_builder import (
    assemble_experiment,
    expanded_stacks,
    position_builder,
)
from models.transformer._stack import (
    build_controlled_stack,
    build_gate,
    plain_projection_stack,
)

from ._recurrent_composition import build_recurrent_composition
from ._residual import (
    ResidualStackSource,
    build_residual_config,
    resolve_residual_stack_options,
)
from ._transformer_submodule import (
    configure_transformer_submodule,
)
from .experiment_config import ExperimentConfig
from .runtime_options import (
    RuntimeOptions,
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
    TransformerStackOptions,
)


def _gate(model_dim: int, enabled: bool):
    return build_gate(model_dim, enabled, activation=ActivationOptions.SIGMOID)


def _residual_stack(runtime: RuntimeOptions):
    return resolve_residual_stack_options(
        residual_source_from_settings(runtime, ResidualStackSource),
        runtime.encoder_attention_options.stack_options,
    )


def _projection_stack(runtime: RuntimeOptions, options: TransformerAttentionOptions):
    stack_options = options.stack_options
    residual_stack_options = _residual_stack(runtime)
    stack = plain_projection_stack(stack_options, residual_stack_options)
    return configure_transformer_submodule(
        stack,
        control_stack=stack,
        path_options=options,
        model_dim=runtime.model_dim,
        residual_stack_options=residual_stack_options,
    )


def _self_attention(
    runtime: RuntimeOptions,
    options: TransformerAttentionOptions,
    *,
    maximum_length: int,
    causal: bool,
):
    target_dtype = __import__("torch").float32
    projection_model_config = _projection_stack(runtime, options)
    source_memory_config = _source_memory.build_source_memory_config(options)
    return SelfAttentionConfig(
        batch_size=runtime.batch_size,
        num_heads=options.num_heads,
        embedding_dim=runtime.model_dim,
        query_key_projection_dim=runtime.model_dim,
        value_projection_dim=runtime.model_dim,
        target_sequence_length=maximum_length,
        source_sequence_length=maximum_length,
        target_dtype=target_dtype,
        dropout_probability=runtime.dropout_probability,
        zero_attention_flag=options.zero_attention_flag,
        causal_attention_mask_flag=causal,
        add_key_value_bias_flag=options.add_key_value_bias_flag,
        average_attention_weights_flag=True,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=projection_model_config,
        relative_positional_embedding_config=None,
        source_memory_flag=options.source_memory_flag,
        source_memory_config=source_memory_config,
        projection_strategy=SelfAttentionProjectionStrategy.SEPARATE,
    )


def _cross_attention(
    runtime: RuntimeOptions,
    options: TransformerAttentionOptions,
):
    if options.source_memory_flag:
        raise ValueError(
            "Source memory is supported only for self-attention, not decoder cross-attention."
        )
    target_dtype = __import__("torch").float32
    projection_model_config = _projection_stack(runtime, options)
    source_memory_config = _source_memory.build_source_memory_config(options)
    return IndependentAttentionConfig(
        batch_size=runtime.batch_size,
        num_heads=options.num_heads,
        embedding_dim=runtime.model_dim,
        query_key_projection_dim=runtime.model_dim,
        value_projection_dim=runtime.model_dim,
        target_sequence_length=runtime.target_sequence_length,
        source_sequence_length=runtime.source_sequence_length,
        target_dtype=target_dtype,
        dropout_probability=runtime.dropout_probability,
        zero_attention_flag=options.zero_attention_flag,
        causal_attention_mask_flag=False,
        add_key_value_bias_flag=options.add_key_value_bias_flag,
        average_attention_weights_flag=True,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=projection_model_config,
        relative_positional_embedding_config=None,
        source_memory_flag=options.source_memory_flag,
        source_memory_config=source_memory_config,
    )


def _feed_forward(
    runtime: RuntimeOptions,
    options: TransformerFeedForwardOptions,
):
    stack_options = options.stack_options
    residual_stack_options = _residual_stack(runtime)
    residual_config = build_residual_config(
        stack_options.residual_connection_option,
        stack_options.residual_model_flag,
        residual_stack_options,
        residual_block_size=stack_options.residual_block_size,
        residual_rms_norm_epsilon=stack_options.residual_rms_norm_epsilon,
    )
    layer_model_config = LinearLayerConfig(bias_flag=stack_options.bias_flag)
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
    stack = LayerStackConfig(
        hidden_dim=stack_options.hidden_dim,
        num_layers=stack_options.num_layers,
        apply_output_postprocessing_flag=(
            stack_options.apply_output_postprocessing_flag
        ),
        last_layer_bias_option=stack_options.last_layer_bias_option,
        layer_config=layer_config,
    )
    configured = configure_transformer_submodule(
        stack,
        control_stack=stack,
        path_options=options,
        model_dim=runtime.model_dim,
        residual_stack_options=residual_stack_options,
    )
    return FeedForwardConfig(
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        stack_config=configured,
    )


def _controlled_stack(
    runtime: RuntimeOptions,
    options: TransformerStackOptions,
    layer_config: LayerConfig,
):
    def wrap_recurrent(
        stack: LayerStackConfig,
        gate_config: GateConfig | None,
        residual_config: ResidualConfig | None,
        halting_config: HaltingConfig | None,
    ) -> RecurrentCompositionConfig:
        return build_recurrent_composition(
            option=options.recurrent_composition_option,
            input_dim=runtime.model_dim,
            output_dim=runtime.model_dim,
            block_config=stack,
            max_steps=options.recurrent_max_steps,
            gradient_transition_count=options.recurrent_gradient_transition_count,
            initial_iterations=options.recurrent_initial_iterations,
            iteration_increment=options.recurrent_iteration_increment,
            forward_calls_before_iteration_increment=(
                options.recurrent_forward_calls_before_iteration_increment
            ),
            smooth_iteration_growth_flag=(
                options.recurrent_smooth_iteration_growth_flag
            ),
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            recurrent_normalization=NormalizationOptions.LAYER_NORM,
            gate_config=gate_config,
            residual_config=residual_config,
            halting_config=halting_config,
            memory_config=None,
            no_gradient_transition_count=(
                options.recurrent_no_gradient_transition_count
            ),
            reinject_original_hidden_flag=(
                options.recurrent_reinject_original_hidden_flag
            ),
            latent_updates_per_answer_update=(
                options.recurrent_latent_updates_per_answer_update
            ),
            answer_update_count=options.recurrent_answer_update_count,
            high_cycles=options.recurrent_high_cycles,
            low_cycles=options.recurrent_low_cycles,
            initialization_standard_deviation=(
                options.recurrent_initialization_standard_deviation
            ),
        )

    return build_controlled_stack(
        runtime,
        options,
        layer_config,
        resolve_residual_stack=_residual_stack,
        wrap_recurrent=wrap_recurrent,
        gate_activation=ActivationOptions.SIGMOID,
    )


def _encoder(runtime: RuntimeOptions):
    options = runtime.encoder_options
    residual_config = AdditiveResidualConfig()
    attention_config = _self_attention(
        runtime,
        runtime.encoder_attention_options,
        maximum_length=runtime.source_sequence_length,
        causal=False,
    )
    feed_forward_config = _feed_forward(runtime, runtime.encoder_feed_forward_options)
    transformer_layer = TransformerEncoderLayerConfig(
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
        layer_model_config=transformer_layer,
        sequence_expansion_config=sequence_expansion_config,
    )
    return _controlled_stack(runtime, options, layer)


def _decoder(runtime: RuntimeOptions):
    options = runtime.decoder_options
    residual_config = AdditiveResidualConfig()
    self_attention_config = _self_attention(
        runtime,
        runtime.decoder_self_attention_options,
        maximum_length=runtime.target_sequence_length,
        causal=True,
    )
    cross_attention_config = _cross_attention(
        runtime, runtime.decoder_cross_attention_options
    )
    feed_forward_config = _feed_forward(runtime, runtime.decoder_feed_forward_options)
    transformer_layer = TransformerDecoderLayerConfig(
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
        layer_model_config=transformer_layer,
        sequence_expansion_config=sequence_expansion_config,
    )
    return _controlled_stack(runtime, options, layer)


def build_experiment_config(runtime: RuntimeOptions) -> ExperimentConfig:
    build_position = position_builder(runtime)
    source_position = build_position(runtime.source_sequence_length)
    target_position = build_position(runtime.target_sequence_length)
    encoder, decoder = expanded_stacks(runtime, _encoder, _decoder)
    return assemble_experiment(
        runtime, ExperimentConfig, encoder, decoder, source_position, target_position
    )
