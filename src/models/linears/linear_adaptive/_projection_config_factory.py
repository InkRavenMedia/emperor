from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
)
from models.linears.linear_adaptive._adaptive_linear_template import (
    AdaptiveLinearLayerTemplateConfig,
)
from models.linears.linear_adaptive._adaptive_parameter_config_factory import (
    AdaptiveParameterConfigFactory,
)
from models.linears.linear_adaptive.runtime_options import (
    AdaptiveProjectionOptions,
    RuntimeOptions,
)


class ProjectionConfigFactory:
    def __init__(self, runtime: RuntimeOptions) -> None:
        self._runtime = runtime
        self._adaptive_factory = AdaptiveParameterConfigFactory(runtime)

    def build_input_model_config(self) -> LayerConfig:
        input_model_config = self._build(
            self._runtime.input_projection,
            activation=self._runtime.stack.activation,
        )
        return input_model_config

    def build_output_model_config(self) -> LayerConfig:
        output_model_config = self._build(
            self._runtime.output_projection,
            activation=ActivationOptions.DISABLED,
        )
        return output_model_config

    def _build(
        self,
        options: AdaptiveProjectionOptions,
        *,
        activation: ActivationOptions,
    ) -> LayerConfig:
        augmentation_config = self._adaptive_factory.build_projection_config(options)
        adaptive_linear_config = AdaptiveLinearLayerTemplateConfig(
            bias_flag=self._runtime.stack.bias_flag,
            adaptive_augmentation_config=augmentation_config,
        )
        return LayerConfig(
            activation=activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=adaptive_linear_config,
        )
