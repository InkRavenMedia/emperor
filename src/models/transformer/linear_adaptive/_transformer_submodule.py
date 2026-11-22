from typing import TypeVar

from emperor.config import ConfigBase
from emperor.layers import LayerStackConfig, RecurrentLayerConfig

from .._submodule import TransformerSubmoduleFactory
from ._residual import ResidualStackOptions, build_residual_config
from .runtime_options import (
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
    resolve_controller_stack_options,
)

_Model = TypeVar("_Model", bound=ConfigBase)


def configure_transformer_submodule(
    model_config: _Model,
    *,
    control_stack: LayerStackConfig,
    path_options: TransformerAttentionOptions | TransformerFeedForwardOptions,
    model_dim: int,
    residual_stack_options: ResidualStackOptions,
) -> _Model | RecurrentLayerConfig:
    factory = TransformerSubmoduleFactory(
        resolve_controller_stack_options, build_residual_config
    )
    return factory.configure(
        model_config,
        control_stack=control_stack,
        path_options=path_options,
        model_dim=model_dim,
        residual_stack_options=residual_stack_options,
    )
