from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
)
from emperor.linears import LinearLayerConfig
from models.neuron.expert_linear._hidden.runtime_options import ExpertsStackOptions


class ProjectionConfigFactory:
    def __init__(self, stack_options: ExpertsStackOptions) -> None:
        self.stack_options = stack_options

    def build_input_model_config(self) -> LayerConfig:
        linear_layer_config = LinearLayerConfig(bias_flag=True)
        return LayerConfig(
            activation=self.stack_options.activation,
            layer_norm_position=self.stack_options.layer_norm_position,
            normalization=self.stack_options.normalization,
            residual_config=None,
            dropout_probability=self.stack_options.dropout_probability,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=linear_layer_config,
        )

    def build_output_model_config(self) -> LayerConfig:
        linear_layer_config = LinearLayerConfig(bias_flag=True)
        return LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=linear_layer_config,
        )
