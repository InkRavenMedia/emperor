from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
)
from models.neuron.linear_adaptive._adaptive_linear_template import (
    AdaptiveLinearLayerTemplateConfig,
)
from models.neuron.linear_adaptive._hidden._adaptive_parameter_config_factory import (
    AdaptiveParameterConfigFactory,
)
from models.neuron.linear_adaptive._hidden.runtime_options import (
    AdaptiveProjectionOptions,
    RuntimeOptions,
)


class ProjectionConfigFactory:
    def __init__(self, runtime: RuntimeOptions) -> None:
        self._runtime = runtime
        self._adaptive_factory = AdaptiveParameterConfigFactory(runtime)

    def build_input_model_config(self) -> LayerConfig:
        input_model_config = self.__build_projection(
            self._runtime.input_projection,
            activation=self._runtime.stack.activation,
        )
        return input_model_config

    def build_output_model_config(self) -> LayerConfig:
        output_model_config = self.__build_projection(
            self._runtime.output_projection,
            activation=ActivationOptions.DISABLED,
        )
        return output_model_config

    def __build_projection(
        self,
        options: AdaptiveProjectionOptions,
        *,
        activation: ActivationOptions,
    ) -> LayerConfig:
        adaptive_augmentation_config = self._adaptive_factory.build_projection_config(
            options
        )
        linear_layer_config = AdaptiveLinearLayerTemplateConfig(
            bias_flag=self._runtime.stack.bias_flag,
            adaptive_augmentation_config=adaptive_augmentation_config,
        )
        return LayerConfig(
            activation=activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=linear_layer_config,
        )
