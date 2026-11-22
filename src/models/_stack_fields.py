from enum import Enum, auto
from types import ModuleType
from typing import Protocol, TypeVar

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)

_Source = TypeVar("_Source", covariant=True)


class _CreateSource(Protocol[_Source]):
    def __call__(
        self,
        *,
        independent_flag: bool,
        hidden_dim: int | None,
        num_layers: int | None,
        last_layer_bias_option: LastLayerBiasOptions | None,
        apply_output_postprocessing_flag: bool | None,
        activation: ActivationOptions | None,
        layer_norm_position: LayerNormPositionOptions | None,
        normalization: NormalizationOptions | None,
        residual_connection_option: type[ResidualConfig] | None,
        residual_model_flag: bool,
        residual_block_size: int | None,
        residual_rms_norm_epsilon: float | None,
        dropout_probability: float | None,
        bias_flag: bool | None,
    ) -> _Source: ...


class StackScope(Enum):
    MAIN = ""
    ATTENTION = "ATTN_"
    FEED_FORWARD = "FF_"
    EXPERT = "EXPERT_"
    ROUTER = "ROUTER_"


class ControllerStackFields(Enum):
    GATE = auto()
    HALTING = auto()
    MEMORY = auto()
    RECURRENT_GATE = auto()
    RECURRENT_HALTING = auto()
    RESIDUAL = auto()

    def read(
        self,
        config: ModuleType,
        create: _CreateSource[_Source],
        *,
        scope: StackScope = StackScope.MAIN,
    ) -> _Source:
        prefix = f"{scope.value}{self.name}_STACK"
        return create(
            independent_flag=getattr(config, f"{prefix}_INDEPENDENT_FLAG"),
            hidden_dim=getattr(config, f"{prefix}_HIDDEN_DIM"),
            num_layers=getattr(config, f"{prefix}_NUM_LAYERS"),
            last_layer_bias_option=getattr(config, f"{prefix}_LAST_LAYER_BIAS_OPTION"),
            apply_output_postprocessing_flag=getattr(
                config, f"{prefix}_APPLY_OUTPUT_POSTPROCESSING_FLAG"
            ),
            activation=getattr(config, f"{prefix}_ACTIVATION"),
            layer_norm_position=getattr(config, f"{prefix}_LAYER_NORM_POSITION"),
            normalization=getattr(config, f"{prefix}_NORMALIZATION"),
            residual_connection_option=getattr(
                config, f"{prefix}_RESIDUAL_CONNECTION_OPTION"
            ),
            residual_block_size=getattr(config, f"{prefix}_RESIDUAL_BLOCK_SIZE"),
            residual_rms_norm_epsilon=getattr(
                config, f"{prefix}_RESIDUAL_RMS_NORM_EPSILON"
            ),
            residual_model_flag=getattr(config, f"{prefix}_RESIDUAL_MODEL_FLAG"),
            dropout_probability=getattr(config, f"{prefix}_DROPOUT_PROBABILITY"),
            bias_flag=getattr(config, f"{prefix}_BIAS_FLAG"),
        )


class GeneratorStackFields(Enum):
    BIAS_GENERATOR = auto()
    DIAGONAL_GENERATOR = auto()
    MASK_GENERATOR = auto()
    WEIGHT_GENERATOR = auto()

    def read(
        self,
        config: ModuleType,
        create: _CreateSource[_Source],
        *,
        scope: StackScope = StackScope.MAIN,
    ) -> _Source:
        prefix = f"{scope.value}{self.name}_STACK"
        return create(
            independent_flag=getattr(config, f"{prefix}_INDEPENDENT_FLAG"),
            hidden_dim=getattr(config, f"{prefix}_HIDDEN_DIM"),
            layer_norm_position=getattr(config, f"{prefix}_LAYER_NORM_POSITION"),
            normalization=getattr(config, f"{prefix}_NORMALIZATION"),
            num_layers=getattr(config, f"{prefix}_NUM_LAYERS"),
            activation=getattr(config, f"{prefix}_ACTIVATION"),
            residual_connection_option=getattr(
                config, f"{prefix}_RESIDUAL_CONNECTION_OPTION"
            ),
            residual_block_size=getattr(config, f"{prefix}_RESIDUAL_BLOCK_SIZE"),
            residual_rms_norm_epsilon=getattr(
                config, f"{prefix}_RESIDUAL_RMS_NORM_EPSILON"
            ),
            residual_model_flag=getattr(config, f"{prefix}_RESIDUAL_MODEL_FLAG"),
            dropout_probability=getattr(config, f"{prefix}_DROPOUT_PROBABILITY"),
            last_layer_bias_option=getattr(config, f"{prefix}_LAST_LAYER_BIAS_OPTION"),
            apply_output_postprocessing_flag=getattr(
                config, f"{prefix}_APPLY_OUTPUT_POSTPROCESSING_FLAG"
            ),
            bias_flag=getattr(config, f"{prefix}_BIAS_FLAG"),
        )
