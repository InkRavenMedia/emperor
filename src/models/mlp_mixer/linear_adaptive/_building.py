from __future__ import annotations

from functools import partial

from emperor.attention import MixerAttentionConfig
from emperor.layers import (
    LayerStackConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import (
    DynamicMemoryConfig,
)
from emperor.patch import PatchConfig
from models._adaptive_generator_stack import build_generator_layers
from models._residual_stack import residual_source_from_settings
from models.mlp_mixer._adaptive_layers import AdaptiveLayerFactory
from models.mlp_mixer._component_factory import MixerComponentFactory
from models.mlp_mixer._encoder import (
    build_encoder_stack,
    build_token_attention,
    patch_sequence_length,
)
from models.mlp_mixer._encoder import output_config as output_config

from ._adaptive_linear_template import AdaptiveLinearLayerTemplateConfig
from ._control_options import (
    ControlOptions,
    StackOptions,
    channel_mixer_control_options,
    channel_mixer_stack_options,
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


def _component_factory() -> MixerComponentFactory[RuntimeOptions]:
    return MixerComponentFactory(_residual, _plain_linear_config)


def patch_config(runtime: RuntimeOptions) -> PatchConfig:
    components = _component_factory()
    return components.build_patch(runtime)


def _token_mixing_model(runtime: RuntimeOptions, tokens: int):
    token_stack_options = token_mixer_stack_options(runtime)
    token_controls = token_mixer_control_options(runtime)
    token_mixing_config = _affine_stack(
        runtime,
        stack_options=token_stack_options,
        input_dim=tokens,
        output_dim=tokens,
        control_options=token_controls,
        control_name="token_mixer",
    )
    return token_mixing_config


def _channel_mixing_model(runtime: RuntimeOptions):
    channel_stack_options = channel_mixer_stack_options(runtime)
    channel_controls = channel_mixer_control_options(runtime)
    channel_mixing_config = _affine_stack(
        runtime,
        stack_options=channel_stack_options,
        grouping_prefix="channel_mixer_",
        input_dim=runtime.hidden_dim,
        output_dim=runtime.hidden_dim,
        mirrored=True,
        control_options=channel_controls,
        control_name="channel_mixer",
    )
    return channel_mixing_config


def _source_memory_config(runtime: RuntimeOptions) -> DynamicMemoryConfig | None:
    if not runtime.token_mixer_source_memory_flag:
        return None
    submodule_defaults = submodule_stack_options(runtime)
    components = _component_factory()
    return components.build_source_memory(runtime, submodule_defaults)


def _mixer_config(runtime: RuntimeOptions, tokens: int) -> MixerAttentionConfig:
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
