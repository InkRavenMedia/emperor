from collections.abc import Callable
from typing import Final, Protocol, TypeVar

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from models._residual_stack import (
    ResidualStackSettings,
    StackDefaults,
    inherit_if_unset,
)


class StackSource(Protocol):
    independent_flag: Final[bool]
    hidden_dim: Final[int | None]
    num_layers: Final[int | None]
    activation: Final[ActivationOptions | None]
    layer_norm_position: Final[LayerNormPositionOptions | None]
    normalization: Final[NormalizationOptions | None]
    residual_connection_option: Final[type[ResidualConfig] | None]
    residual_block_size: Final[int | None]
    residual_rms_norm_epsilon: Final[float | None]
    residual_model_flag: Final[bool]
    dropout_probability: Final[float | None]
    last_layer_bias_option: Final[LastLayerBiasOptions | None]
    apply_output_postprocessing_flag: Final[bool | None]
    bias_flag: Final[bool | None]


class _ControllerDefaults(StackDefaults, Protocol):
    @property
    def residual_stack_options(self) -> ResidualStackSettings | None: ...


_Stack = TypeVar("_Stack", bound=StackDefaults)
_ControllerStack = TypeVar("_ControllerStack", bound=_ControllerDefaults)


def resolve_controller_options(
    source: StackSource,
    defaults: _ControllerStack,
    create_stack: Callable[..., _ControllerStack],
) -> _ControllerStack:
    if not source.independent_flag:
        return defaults
    hidden_dim = inherit_if_unset(source.hidden_dim, defaults.hidden_dim)
    num_layers = inherit_if_unset(source.num_layers, defaults.num_layers)
    last_layer_bias_option = inherit_if_unset(
        source.last_layer_bias_option, defaults.last_layer_bias_option
    )
    apply_output_postprocessing_flag = inherit_if_unset(
        source.apply_output_postprocessing_flag,
        defaults.apply_output_postprocessing_flag,
    )
    activation = inherit_if_unset(source.activation, defaults.activation)
    layer_norm_position = inherit_if_unset(
        source.layer_norm_position, defaults.layer_norm_position
    )
    normalization = inherit_if_unset(source.normalization, defaults.normalization)
    residual_connection_option = inherit_if_unset(
        source.residual_connection_option, defaults.residual_connection_option
    )
    dropout_probability = inherit_if_unset(
        source.dropout_probability, defaults.dropout_probability
    )
    bias_flag = inherit_if_unset(source.bias_flag, defaults.bias_flag)
    return create_stack(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_postprocessing_flag=apply_output_postprocessing_flag,
        activation=activation,
        layer_norm_position=layer_norm_position,
        normalization=normalization,
        residual_connection_option=residual_connection_option,
        residual_block_size=source.residual_block_size,
        residual_rms_norm_epsilon=source.residual_rms_norm_epsilon,
        residual_model_flag=source.residual_model_flag,
        residual_stack_options=defaults.residual_stack_options,
        dropout_probability=dropout_probability,
        bias_flag=bias_flag,
    )


def resolve_transformer_controller_options(
    source: StackSource,
    defaults: _Stack,
    create_stack: Callable[..., _Stack],
) -> _Stack:
    if not source.independent_flag:
        return defaults
    return create_stack(
        hidden_dim=inherit_if_unset(source.hidden_dim, defaults.hidden_dim),
        num_layers=inherit_if_unset(source.num_layers, defaults.num_layers),
        last_layer_bias_option=inherit_if_unset(
            source.last_layer_bias_option, defaults.last_layer_bias_option
        ),
        apply_output_postprocessing_flag=inherit_if_unset(
            source.apply_output_postprocessing_flag,
            defaults.apply_output_postprocessing_flag,
        ),
        activation=inherit_if_unset(source.activation, defaults.activation),
        layer_norm_position=inherit_if_unset(
            source.layer_norm_position, defaults.layer_norm_position
        ),
        normalization=inherit_if_unset(source.normalization, defaults.normalization),
        residual_connection_option=inherit_if_unset(
            source.residual_connection_option, defaults.residual_connection_option
        ),
        residual_block_size=source.residual_block_size,
        residual_rms_norm_epsilon=source.residual_rms_norm_epsilon,
        residual_model_flag=source.residual_model_flag,
        dropout_probability=inherit_if_unset(
            source.dropout_probability, defaults.dropout_probability
        ),
        bias_flag=inherit_if_unset(source.bias_flag, defaults.bias_flag),
    )


def resolve_expert_options(
    create_stack: Callable[..., _ControllerStack],
    defaults: _ControllerStack,
    *,
    residual_stack_last: bool,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
    last_layer_bias_option: LastLayerBiasOptions | None = None,
    apply_output_postprocessing_flag: bool | None = None,
    activation: ActivationOptions | None = None,
    layer_norm_position: LayerNormPositionOptions | None = None,
    normalization: NormalizationOptions | None = None,
    residual_connection_option: type[ResidualConfig] | None = None,
    residual_model_flag: bool | None = None,
    residual_block_size: int | None = None,
    residual_rms_norm_epsilon: float | None = None,
    dropout_probability: float | None = None,
    bias_flag: bool | None = None,
) -> _ControllerStack:
    hidden_dim = inherit_if_unset(hidden_dim, defaults.hidden_dim)
    num_layers = inherit_if_unset(num_layers, defaults.num_layers)
    last_layer_bias_option = inherit_if_unset(
        last_layer_bias_option, defaults.last_layer_bias_option
    )
    apply_output_postprocessing_flag = inherit_if_unset(
        apply_output_postprocessing_flag, defaults.apply_output_postprocessing_flag
    )
    activation = inherit_if_unset(activation, defaults.activation)
    layer_norm_position = inherit_if_unset(
        layer_norm_position, defaults.layer_norm_position
    )
    normalization = inherit_if_unset(normalization, defaults.normalization)
    residual_connection_option = inherit_if_unset(
        residual_connection_option, defaults.residual_connection_option
    )
    residual_block_size = inherit_if_unset(
        residual_block_size, defaults.residual_block_size
    )
    residual_rms_norm_epsilon = inherit_if_unset(
        residual_rms_norm_epsilon, defaults.residual_rms_norm_epsilon
    )
    residual_model_flag = inherit_if_unset(
        residual_model_flag, defaults.residual_model_flag
    )
    if residual_stack_last:
        default_dropout = defaults.dropout_probability
        default_bias = defaults.bias_flag
        residual_stack = defaults.residual_stack_options
    else:
        residual_stack = defaults.residual_stack_options
        default_dropout = defaults.dropout_probability
        default_bias = defaults.bias_flag
    return create_stack(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_postprocessing_flag=apply_output_postprocessing_flag,
        activation=activation,
        layer_norm_position=layer_norm_position,
        normalization=normalization,
        residual_connection_option=residual_connection_option,
        residual_block_size=residual_block_size,
        residual_rms_norm_epsilon=residual_rms_norm_epsilon,
        residual_model_flag=residual_model_flag,
        residual_stack_options=residual_stack,
        dropout_probability=inherit_if_unset(dropout_probability, default_dropout),
        bias_flag=inherit_if_unset(bias_flag, default_bias),
    )


def resolve_expert_controller(
    source: StackSource,
    defaults: _ControllerStack,
    resolver: Callable[..., _ControllerStack],
) -> _ControllerStack:
    if not source.independent_flag:
        return defaults
    return resolver(
        defaults,
        hidden_dim=source.hidden_dim,
        num_layers=source.num_layers,
        last_layer_bias_option=source.last_layer_bias_option,
        apply_output_postprocessing_flag=source.apply_output_postprocessing_flag,
        activation=source.activation,
        layer_norm_position=source.layer_norm_position,
        normalization=source.normalization,
        residual_connection_option=source.residual_connection_option,
        residual_block_size=source.residual_block_size,
        residual_rms_norm_epsilon=source.residual_rms_norm_epsilon,
        residual_model_flag=source.residual_model_flag,
        dropout_probability=source.dropout_probability,
        bias_flag=source.bias_flag,
    )
