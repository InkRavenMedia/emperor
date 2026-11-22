from emperor.config import ModelConfig
from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
)
from emperor.linears import LinearLayerConfig
from models.parametric.parametric_matrix._control_config_factory import (
    build_parametric_stack_config,
)
from models.parametric.parametric_matrix.experiment_config import ExperimentConfig
from models.parametric.parametric_matrix.runtime_defaults import DEFAULT_RUNTIME
from models.parametric.parametric_matrix.runtime_options import (
    RuntimeOptions,
)


class ParametricMatrixConfigBuilder:
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        self.runtime = runtime

    @property
    def runtime(self) -> RuntimeOptions:
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: RuntimeOptions) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.parametric.parametric_matrix ParametricMatrixConfigBuilder runtime must be RuntimeOptions"
            )
        options = runtime.construction_options()
        self.batch_size = options.batch_size
        self.learning_rate = options.learning_rate
        self.input_dim = options.input_dim
        self.stack_options = options.stack
        self.hidden_dim = options.stack.hidden_dim
        self.output_dim = options.output_dim
        self.stack_num_layers = options.stack.num_layers
        self.stack_activation = options.stack.activation
        self.stack_residual_connection_option = options.stack.residual_connection_option
        self.stack_residual_model_flag = options.stack.residual_model_flag
        self.stack_residual_block_size = options.stack.residual_block_size
        self.stack_residual_rms_norm_epsilon = options.stack.residual_rms_norm_epsilon
        self.stack_dropout_probability = options.stack.dropout_probability
        self.residual_stack_options = options.residual_stack
        self.mixture_options = options.mixture
        self.adaptive_mixture_top_k = options.mixture.top_k
        self.adaptive_mixture_num_experts = options.mixture.num_experts
        self.adaptive_mixture_weighted_parameters_flag = (
            options.mixture.weighted_parameters_flag
        )
        self.adaptive_bias_option = options.adaptive_bias_option
        self.sampler_options = options.sampler
        self.sampler_threshold = options.sampler.threshold
        self.sampler_filter_above_threshold = options.sampler.filter_above_threshold
        self.sampler_num_topk_samples = options.sampler.num_topk_samples
        self.sampler_normalize_probabilities_flag = (
            options.sampler.normalize_probabilities_flag
        )
        self.sampler_noisy_topk_flag = options.sampler.noisy_topk_flag
        self.sampler_coefficient_of_variation_loss_weight = (
            options.sampler.coefficient_of_variation_loss_weight
        )
        self.sampler_switch_loss_weight = options.sampler.switch_loss_weight
        self.sampler_zero_centred_loss_weight = options.sampler.zero_centred_loss_weight
        self.sampler_mutual_information_loss_weight = (
            options.sampler.mutual_information_loss_weight
        )
        self.router_options = options.router
        self._runtime = runtime

    def build(self) -> ModelConfig:
        input_model_config = build_linear_layer_config(
            activation=self.stack_activation,
        )
        model_config = build_parametric_stack_config(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            stack_options=self.stack_options,
            mixture_options=self.mixture_options,
            sampler_options=self.sampler_options,
            router_options=self.router_options,
            adaptive_bias_option=self.adaptive_bias_option,
            residual_stack_options=self.residual_stack_options,
        )
        output_model_config = build_linear_layer_config(
            activation=ActivationOptions.DISABLED,
        )
        experiment_config = ExperimentConfig(
            input_model_config=input_model_config,
            model_config=model_config,
            output_model_config=output_model_config,
        )
        return ModelConfig(
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            learning_rate=self.learning_rate,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=experiment_config,
        )


def build_linear_layer_config(
    *,
    activation: ActivationOptions,
) -> LayerConfig:
    layer_model_config = LinearLayerConfig(
        bias_flag=True,
    )
    return LayerConfig(
        activation=activation,
        residual_config=None,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        normalization=NormalizationOptions.RMS_NORM,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=layer_model_config,
    )
