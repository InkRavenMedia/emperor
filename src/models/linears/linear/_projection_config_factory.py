from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
)
from emperor.linears import LinearLayerConfig
from models.linears.linear.runtime_options import LinearBuildSettings


class ProjectionConfigFactory:
    def __init__(self, build_settings: LinearBuildSettings) -> None:
        self.build_settings = build_settings

    def build_input_model_config(self) -> LayerConfig:
        input_model_config = self.__projection_layer(
            self.build_settings.stack.activation
        )
        return input_model_config

    def build_output_model_config(self) -> LayerConfig:
        output_model_config = self.__projection_layer(ActivationOptions.DISABLED)
        return output_model_config

    @staticmethod
    def __projection_layer(activation: ActivationOptions) -> LayerConfig:
        linear_config = LinearLayerConfig(bias_flag=True)
        return LayerConfig(
            activation=activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=linear_config,
        )
