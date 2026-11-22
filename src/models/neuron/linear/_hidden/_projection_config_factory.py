from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
)
from emperor.linears import LinearLayerConfig
from models.neuron.linear._hidden.runtime_options import RuntimeOptions


class ProjectionConfigFactory:
    def __init__(self, runtime: RuntimeOptions) -> None:
        self.runtime = runtime

    def build_input_model_config(self) -> LayerConfig:
        linear_layer_config = LinearLayerConfig(
            bias_flag=True,
        )
        return LayerConfig(
            activation=self.runtime.stack.activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=linear_layer_config,
        )

    def build_output_model_config(self) -> LayerConfig:
        linear_layer_config = LinearLayerConfig(
            bias_flag=True,
        )
        return LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=linear_layer_config,
        )
