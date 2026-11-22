from abc import ABC, abstractmethod
from typing import Protocol

from emperor.config import ConfigBase
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from models._residual_stack import (
    ResidualStackSettings,
    StackDefaults,
    build_residual_config,
)


class LayerStackSettings(StackDefaults, Protocol):
    @property
    def residual_stack_options(self) -> ResidualStackSettings | None: ...


class LinearStackFactory(ABC):
    hidden_dim: int

    @property
    @abstractmethod
    def _default_activation(self) -> ActivationOptions: ...

    def build_backend_linear_layer_config(
        self, *, bias_flag: bool
    ) -> LinearLayerConfig:
        return self.build_plain_linear_layer_config(bias_flag=bias_flag)

    def build_plain_linear_layer_config(
        self,
        *,
        bias_flag: bool,
    ) -> LinearLayerConfig:
        return LinearLayerConfig(bias_flag=bias_flag)

    def build_backend_linear_stack_config(
        self,
        *,
        num_layers: int,
        bias_flag: bool,
        layer_norm_position: LayerNormPositionOptions,
        normalization: NormalizationOptions = NormalizationOptions.RMS_NORM,
        dropout_probability: float,
        input_dim: int | None = None,
        output_dim: int | None = None,
        apply_output_postprocessing_flag: bool = True,
    ) -> LayerStackConfig:
        layer_model_config = self.build_backend_linear_layer_config(
            bias_flag=bias_flag,
        )
        backend_linear_stack_config = self.build_linear_stack_config(
            layer_model_config=layer_model_config,
            num_layers=num_layers,
            layer_norm_position=layer_norm_position,
            normalization=normalization,
            dropout_probability=dropout_probability,
            input_dim=input_dim,
            output_dim=output_dim,
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
        )
        return backend_linear_stack_config

    def build_plain_linear_stack_config(
        self,
        *,
        num_layers: int,
        bias_flag: bool,
        layer_norm_position: LayerNormPositionOptions,
        normalization: NormalizationOptions = NormalizationOptions.RMS_NORM,
        dropout_probability: float,
        input_dim: int | None = None,
        output_dim: int | None = None,
        apply_output_postprocessing_flag: bool = True,
    ) -> LayerStackConfig:
        layer_model_config = self.build_plain_linear_layer_config(
            bias_flag=bias_flag,
        )
        plain_linear_stack_config = self.build_linear_stack_config(
            layer_model_config=layer_model_config,
            num_layers=num_layers,
            layer_norm_position=layer_norm_position,
            normalization=normalization,
            dropout_probability=dropout_probability,
            input_dim=input_dim,
            output_dim=output_dim,
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
        )
        return plain_linear_stack_config

    def build_submodule_stack_config(
        self, options: LayerStackSettings
    ) -> LayerStackConfig:
        model = self.build_backend_linear_layer_config(bias_flag=options.bias_flag)
        return self.build_linear_stack_config(
            layer_model_config=model,
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            activation=options.activation,
            residual_connection_option=options.residual_connection_option,
            residual_block_size=options.residual_block_size,
            residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
            residual_model_flag=options.residual_model_flag,
            residual_stack_options=options.residual_stack_options,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            dropout_probability=options.dropout_probability,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
        )

    def build_linear_stack_config(
        self,
        *,
        layer_model_config: ConfigBase,
        num_layers: int,
        layer_norm_position: LayerNormPositionOptions,
        normalization: NormalizationOptions = NormalizationOptions.RMS_NORM,
        dropout_probability: float,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        activation: ActivationOptions | None = None,
        residual_connection_option: type[ResidualConfig] | None = None,
        residual_model_flag: bool = False,
        residual_block_size: int | None = None,
        residual_rms_norm_epsilon: float | None = None,
        residual_stack_options: ResidualStackSettings | None = None,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag: bool = True,
    ) -> LayerStackConfig:
        residual_config = build_residual_config(
            residual_connection_option,
            residual_model_flag,
            residual_stack_options,
            residual_block_size=residual_block_size,
            residual_rms_norm_epsilon=residual_rms_norm_epsilon,
        )
        if activation is None:
            activation = self._default_activation
        layer_config = LayerConfig(
            activation=activation,
            layer_norm_position=layer_norm_position,
            normalization=normalization,
            residual_config=residual_config,
            dropout_probability=dropout_probability,
            gate_config=None,
            halting_config=None,
            layer_model_config=layer_model_config,
        )
        if hidden_dim is None:
            hidden_dim = self.hidden_dim
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
            layer_config=layer_config,
        )
