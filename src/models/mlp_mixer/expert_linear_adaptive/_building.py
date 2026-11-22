from __future__ import annotations

from functools import partial
from typing import cast

from emperor.attention import AttentionSourceMemoryConfig, MixerAttentionConfig
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions,
)
from emperor.experts import (
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
)
from emperor.layers import (
    LayerStackConfig,
    MirroredLayerStackConfig,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.patch import LinearPatchEmbeddingConfig
from emperor.sampler import RouterConfig, SamplerConfig
from models import _projection_model
from models._adaptive_generator_stack import build_generator_layers
from models._expert_mixture import build_expert_mixture, build_expert_model
from models._residual_stack import residual_source_from_settings
from models.mlp_mixer._adaptive_layers import AdaptiveLayerFactory
from models.mlp_mixer._component_factory import MixerComponentFactory
from models.mlp_mixer._control_options import HaltingOptions as BaseHaltingOptions
from models.mlp_mixer._encoder import (
    build_encoder_stack,
    build_token_attention,
    patch_sequence_length,
)
from models.mlp_mixer._encoder import output_config as output_config

from ._adaptive_linear_template import AdaptiveLinearLayerTemplateConfig
from ._control_options import (
    ControlOptions,
    HaltingOptions,
    StackOptions,
    channel_mixer_control_options,
    channel_mixer_stack_options,
    expert_control_options,
    main_control_options,
    submodule_stack_options,
    token_mixer_control_options,
    token_mixer_stack_options,
)
from ._residual import (
    ResidualStackSource,
    build_residual_config,
    resolve_residual_stack_options,
)
from .runtime_options import RuntimeOptions


def sequence_length(runtime: RuntimeOptions) -> int:
    if runtime.hierarchical_masked_word_flag:
        if type(runtime.sequence_length) is not int or runtime.sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")
        return runtime.sequence_length
    return patch_sequence_length(runtime)


def _residual(
    runtime: RuntimeOptions,
    option: type[ResidualConfig] | None,
    model_flag: bool,
    *,
    residual_block_size: int | None = None,
    residual_rms_norm_epsilon: float | None = None,
):
    residual_stack_source = residual_source_from_settings(runtime, ResidualStackSource)
    submodule_defaults = submodule_stack_options(runtime)
    residual_stack_options = resolve_residual_stack_options(
        residual_stack_source,
        submodule_defaults,
    )
    residual_config = build_residual_config(
        option,
        model_flag,
        residual_stack_options,
        residual_block_size=residual_block_size,
        residual_rms_norm_epsilon=residual_rms_norm_epsilon,
    )
    return residual_config


def _plain_linear_config(*, bias_flag: bool) -> LinearLayerConfig:
    return LinearLayerConfig(bias_flag=bias_flag)


def _generator_stack(
    runtime: RuntimeOptions,
    options: StackOptions,
) -> LayerStackConfig:
    residual_config = _residual(
        runtime,
        options.residual_connection_option,
        options.residual_model_flag,
        residual_block_size=options.residual_block_size,
        residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
    )
    return build_generator_layers(options, residual_config)


def _backend_linear_config(
    runtime: RuntimeOptions,
    *,
    bias_flag: bool,
    grouping_prefix: str = "",
):
    factory = AdaptiveLayerFactory(
        runtime,
        partial(_generator_stack, runtime),
        template_type=AdaptiveLinearLayerTemplateConfig,
    )
    return factory.build(bias_flag=bias_flag, grouping_prefix=grouping_prefix)


def _affine_stack(
    runtime: RuntimeOptions,
    *,
    stack_options: StackOptions,
    input_dim: int | None,
    output_dim: int | None,
    mirrored: bool = False,
    backend: bool = True,
    grouping_prefix: str = "",
    control_options: ControlOptions | None = None,
    control_name: str | None = None,
):
    build_layer = None
    if backend:
        build_layer = partial(
            _backend_linear_config, runtime, grouping_prefix=grouping_prefix
        )
    components = _component_factory()
    stack = components.build_affine_stack(
        runtime,
        stack_options=stack_options,
        input_dim=input_dim,
        output_dim=output_dim,
        mirrored=mirrored,
        build_layer=build_layer,
    )
    if control_options is None:
        return stack
    if input_dim is None or output_dim is None or input_dim != output_dim:
        raise ValueError(
            f"{control_name} controls require equal concrete input/output "
            f"dimensions, got input_dim={input_dim}, output_dim={output_dim}"
        )
    components = _component_factory()
    controlled_stack_config = components.configure_stack(
        runtime,
        stack,
        options=control_options,
        defaults=stack_options,
        model_dim=input_dim,
    )
    return controlled_stack_config


class _ExpertMixerComponentFactory(MixerComponentFactory[RuntimeOptions]):
    @staticmethod
    def _halting_min_steps(options: BaseHaltingOptions) -> int:
        return cast(HaltingOptions, options).min_steps

    @staticmethod
    def _halting_ponder_cost(options: BaseHaltingOptions) -> float:
        return cast(HaltingOptions, options).ponder_cost_weight


def _component_factory() -> MixerComponentFactory[RuntimeOptions]:
    return _ExpertMixerComponentFactory(_residual, _plain_linear_config)


def patch_config(runtime: RuntimeOptions) -> LinearPatchEmbeddingConfig:
    components = _component_factory()
    return components.build_patch(runtime)


def _router_stack(runtime: RuntimeOptions) -> LayerStackConfig:
    router_stack_options = StackOptions(
        hidden_dim=runtime.router_stack_hidden_dim,
        num_layers=runtime.router_stack_num_layers,
        activation=runtime.router_stack_activation,
        dropout_probability=runtime.router_stack_dropout_probability,
        layer_norm_position=runtime.router_stack_layer_norm_position,
        normalization=runtime.router_stack_normalization,
        residual_connection_option=runtime.router_stack_residual_connection_option,
        residual_block_size=runtime.router_stack_residual_block_size,
        residual_rms_norm_epsilon=runtime.router_stack_residual_rms_norm_epsilon,
        residual_model_flag=runtime.router_stack_residual_model_flag,
        last_layer_bias_option=runtime.router_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=runtime.router_stack_apply_output_postprocessing_flag,
        bias_flag=runtime.router_bias_flag,
    )
    components = _component_factory()
    router_stack_config = components.build_affine_stack(
        runtime,
        stack_options=router_stack_options,
        input_dim=None,
        output_dim=None,
    )
    return router_stack_config


def _sampler_config(runtime: RuntimeOptions) -> SamplerConfig:
    router_model_config = _router_stack(runtime)
    router_config = RouterConfig(
        input_dim=None,
        num_experts=runtime.num_experts,
        noisy_topk_flag=runtime.router_noisy_topk_flag,
        model_config=router_model_config,
    )
    return SamplerConfig(
        top_k=runtime.top_k,
        threshold=runtime.sampler_threshold,
        filter_above_threshold=runtime.sampler_filter_above_threshold,
        num_topk_samples=runtime.sampler_num_topk_samples,
        normalize_probabilities_flag=runtime.sampler_normalize_probabilities_flag,
        noisy_topk_flag=runtime.sampler_noisy_topk_flag,
        num_experts=runtime.num_experts,
        coefficient_of_variation_loss_weight=(
            runtime.sampler_coefficient_of_variation_loss_weight
        ),
        switch_loss_weight=runtime.sampler_switch_loss_weight,
        zero_centred_loss_weight=runtime.sampler_zero_centred_loss_weight,
        mutual_information_loss_weight=(runtime.sampler_mutual_information_loss_weight),
        router_config=router_config,
    )


def _expert_stack(
    runtime: RuntimeOptions,
    grouping_prefix: str = "",
) -> LayerStackConfig | RecurrentLayerConfig:
    expert_stack_options = StackOptions(
        hidden_dim=runtime.expert_stack_hidden_dim,
        num_layers=runtime.expert_stack_num_layers,
        activation=runtime.expert_stack_activation,
        dropout_probability=runtime.expert_stack_dropout_probability,
        layer_norm_position=runtime.expert_stack_layer_norm_position,
        normalization=runtime.expert_stack_normalization,
        residual_connection_option=runtime.expert_stack_residual_connection_option,
        residual_block_size=runtime.expert_stack_residual_block_size,
        residual_rms_norm_epsilon=runtime.expert_stack_residual_rms_norm_epsilon,
        residual_model_flag=runtime.expert_stack_residual_model_flag,
        last_layer_bias_option=runtime.expert_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=runtime.expert_stack_apply_output_postprocessing_flag,
        bias_flag=runtime.expert_bias_flag,
    )
    stack = cast(
        LayerStackConfig,
        _affine_stack(
            runtime,
            stack_options=expert_stack_options,
            grouping_prefix=grouping_prefix,
            input_dim=None,
            output_dim=None,
            backend=True,
        ),
    )
    defaults = StackOptions(
        hidden_dim=runtime.expert_stack_hidden_dim,
        num_layers=runtime.expert_stack_num_layers,
        activation=runtime.expert_stack_activation,
        dropout_probability=runtime.expert_stack_dropout_probability,
        layer_norm_position=runtime.expert_stack_layer_norm_position,
        normalization=runtime.expert_stack_normalization,
        residual_connection_option=runtime.expert_stack_residual_connection_option,
        residual_block_size=runtime.expert_stack_residual_block_size,
        residual_rms_norm_epsilon=runtime.expert_stack_residual_rms_norm_epsilon,
        residual_model_flag=runtime.expert_stack_residual_model_flag,
        last_layer_bias_option=runtime.expert_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=(
            runtime.expert_stack_apply_output_postprocessing_flag
        ),
        bias_flag=runtime.expert_bias_flag,
    )
    expert_controls = expert_control_options(runtime)
    components = _component_factory()
    expert_stack_config = components.configure_stack(
        runtime,
        stack,
        options=expert_controls,
        defaults=defaults,
        model_dim=None,
        shared_halting=False,
        shared_memory=False,
    )
    return expert_stack_config


def _mixture_model_config(
    runtime: RuntimeOptions,
    *,
    stack_options: StackOptions,
    grouping_prefix: str = "",
    input_dim: int,
    output_dim: int,
    mirrored: bool,
    control_options: ControlOptions,
) -> MixtureOfExpertsModelConfig:
    if runtime.routing_initialization_mode not in (
        RoutingInitializationMode.LAYER,
        RoutingInitializationMode.SHARED,
    ):
        raise ValueError(
            "routing_initialization_mode must be LAYER or SHARED for an "
            "MLP-Mixer expert branch."
        )
    layer_sampler_config = None
    if runtime.routing_initialization_mode == RoutingInitializationMode.LAYER:
        layer_sampler_config = _sampler_config(runtime)
    expert_model_config = _expert_stack(runtime, grouping_prefix)
    mixture_config = build_expert_mixture(
        runtime,
        input_dim=None,
        output_dim=None,
        sampler_config=layer_sampler_config,
        expert_model_config=expert_model_config,
    )
    stack_type = LayerStackConfig
    stack_depth = stack_options.num_layers
    if mirrored:
        stack_type = MirroredLayerStackConfig
        if stack_options.num_layers <= 0 or stack_options.num_layers % 2:
            raise ValueError(
                "channel_mixer_num_layers must be a positive even integer, got "
                f"{stack_options.num_layers}"
            )
        stack_depth = stack_options.num_layers // 2
    residual_config = _residual(
        runtime,
        stack_options.residual_connection_option,
        stack_options.residual_model_flag,
        residual_block_size=stack_options.residual_block_size,
        residual_rms_norm_epsilon=stack_options.residual_rms_norm_epsilon,
    )
    mixture_layer_config = MixtureOfExpertsLayerConfig(
        activation=stack_options.activation,
        residual_config=residual_config,
        dropout_probability=stack_options.dropout_probability,
        layer_norm_position=stack_options.layer_norm_position,
        normalization=stack_options.normalization,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=mixture_config,
    )
    mixture_stack = stack_type(
        input_dim=input_dim,
        hidden_dim=stack_options.hidden_dim,
        output_dim=output_dim,
        num_layers=stack_depth,
        apply_output_postprocessing_flag=stack_options.apply_output_postprocessing_flag,
        last_layer_bias_option=stack_options.last_layer_bias_option,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=mixture_layer_config,
    )
    sampler_config = None
    if runtime.routing_initialization_mode == RoutingInitializationMode.SHARED:
        sampler_config = _sampler_config(runtime)
    model_config = build_expert_model(
        runtime,
        input_dim=input_dim,
        output_dim=output_dim,
        sampler_config=sampler_config,
        stack_config=mixture_stack,
    )
    components = _component_factory()
    mixture_model_config = components.configure_controls(
        runtime,
        options=control_options,
        model_config=model_config,
        control_stack=mixture_stack,
        defaults=stack_options,
        model_dim=input_dim,
    )
    return mixture_model_config


def _token_mixing_model(runtime: RuntimeOptions, tokens: int):
    token_stack_options = token_mixer_stack_options(runtime)
    token_controls = token_mixer_control_options(runtime)
    token_mixing_config = _mixture_model_config(
        runtime,
        stack_options=token_stack_options,
        input_dim=tokens,
        output_dim=tokens,
        mirrored=False,
        control_options=token_controls,
    )
    return token_mixing_config


def _channel_mixing_model(runtime: RuntimeOptions):
    channel_stack_options = channel_mixer_stack_options(runtime)
    channel_controls = channel_mixer_control_options(runtime)
    channel_mixing_config = _mixture_model_config(
        runtime,
        stack_options=channel_stack_options,
        grouping_prefix="channel_mixer_",
        input_dim=runtime.hidden_dim,
        output_dim=runtime.hidden_dim,
        mirrored=not runtime.hierarchical_masked_word_flag,
        control_options=channel_controls,
    )
    return channel_mixing_config


def _source_memory_config(runtime: RuntimeOptions):
    if not runtime.token_mixer_source_memory_flag:
        return None
    if runtime.token_mixer_source_memory_chunk_size is not None:
        chunk_size = runtime.token_mixer_source_memory_chunk_size
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError(
                "token_mixer_source_memory_chunk_size must be a positive integer"
            )
        if (
            runtime.token_mixer_source_memory_test_time_training_learning_rate
            is not None
            or runtime.token_mixer_source_memory_test_time_training_num_inner_steps
            is not None
        ):
            raise ValueError(
                "Chunked source memory does not support per-token memory test-time training"
            )
        source_memory_model_config = _projection_model.resolve_projection_model(
            runtime.token_mixer_source_memory_model_config,
            runtime.token_mixer_source_memory_model_option,
            runtime.token_mixer_source_memory_model_bias_flag,
            stack_num_layers=runtime.token_mixer_source_memory_model_stack_num_layers,
            stack_hidden_dim=runtime.token_mixer_source_memory_model_stack_hidden_dim,
            recurrent_max_steps=runtime.token_mixer_source_memory_model_recurrent_max_steps,
        ) or LinearLayerConfig(bias_flag=True)
        grouping_config = runtime.token_mixer_source_memory_grouping_method(
            scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
            chunk_size=chunk_size,
        )
        return AttentionSourceMemoryConfig(
            model_config=source_memory_model_config,
            grouping_config=grouping_config,
        )
    submodule_defaults = submodule_stack_options(runtime)
    components = _component_factory()
    return components.build_source_memory(runtime, submodule_defaults)


def _mixer_config(runtime: RuntimeOptions, tokens: int) -> MixerAttentionConfig:
    if runtime.hierarchical_masked_word_flag:
        if (
            runtime.sequence_expansion_flag
            or runtime.sequence_expansion_config is not None
        ):
            raise ValueError("Masked-word Mixer does not support sequence expansion.")
    return build_token_attention(
        runtime,
        tokens,
        build_token_model=_token_mixing_model,
        build_source_memory=_source_memory_config,
    )


def encoder_config(runtime: RuntimeOptions, tokens: int):
    stack = build_encoder_stack(
        runtime,
        tokens,
        build_residual=_residual,
        build_attention=_mixer_config,
        build_channel_model=_channel_mixing_model,
    )
    main_controls = main_control_options(runtime)
    submodule_defaults = submodule_stack_options(runtime)
    model_dim = runtime.hidden_dim
    components = _component_factory()
    return components.configure_stack(
        runtime,
        stack,
        options=main_controls,
        defaults=submodule_defaults,
        model_dim=model_dim,
    )
