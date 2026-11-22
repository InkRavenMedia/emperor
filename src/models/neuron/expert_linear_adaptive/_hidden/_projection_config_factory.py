# ruff: noqa: E501

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    MatrixBiasMixtureConfig,
)
from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
)
from models.neuron.expert_linear_adaptive._adaptive_linear_template import (
    AdaptiveLinearLayerTemplateConfig,
)
from models.neuron.expert_linear_adaptive._generation import (
    build_mixture_generation_config,
    build_weight_generation_config,
)
from models.neuron.expert_linear_adaptive._hidden import runtime_options
from models.neuron.expert_linear_adaptive._hidden._adaptive_generator_stack_config_factory import (
    AdaptiveGeneratorStackConfigFactory,
)
from models.neuron.expert_linear_adaptive._hidden._adaptive_parameter_config_factory import (
    build_bias_config,
    build_diagonal_config,
    build_mask_config,
    build_weight_config,
)
from models.neuron.expert_linear_adaptive._hidden.runtime_options import (
    RuntimeOptions,
)


class BoundaryModelConfigFactory:
    def __init__(self, runtime: RuntimeOptions) -> None:
        self.runtime = runtime
        self.adaptive_generator_stack_config_factory = (
            AdaptiveGeneratorStackConfigFactory(
                runtime.adaptive_generator_stack_options
            )
        )
        self.shared_adaptive_generator_stack_config = (
            self.adaptive_generator_stack_config_factory.build_shared_config()
        )

    def build_input_model_config(self) -> LayerConfig:
        input_model_config = self.__build_boundary_layer_config(
            self.runtime.input_boundary_options,
            activation=self.runtime.stack_options.activation,
        )
        return input_model_config

    def build_output_model_config(self) -> LayerConfig:
        output_model_config = self.__build_boundary_layer_config(
            self.runtime.output_boundary_options,
            activation=ActivationOptions.DISABLED,
        )
        return output_model_config

    def __build_boundary_layer_config(
        self,
        options: runtime_options.AdaptiveBoundaryModelOptions,
        *,
        activation: ActivationOptions,
    ) -> LayerConfig:
        layer_model_config = self.__build_boundary_layer_model_config(options)
        return LayerConfig(
            activation=activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=layer_model_config,
        )

    def __build_boundary_layer_model_config(
        self,
        options: runtime_options.AdaptiveBoundaryModelOptions,
    ) -> AdaptiveLinearLayerConfig:
        weight_generation = build_weight_generation_config(
            options.weight_option,
            options.weight_generation,
            self.adaptive_generator_stack_config_factory,
        )
        weight_config = build_weight_config(
            generation=weight_generation,
            weight_option=options.weight_option,
            generator_depth=options.generator_depth,
            decay_schedule=options.weight_decay_schedule,
            decay_rate=options.weight_decay_rate,
            decay_warmup_batches=options.weight_decay_warmup_batches,
            normalization_option=options.weight_normalization_option,
            normalization_learnable_scale_flag=options.weight_normalization_learnable_scale_flag,
            normalization_position_option=(
                options.weight_normalization_position_option
            ),
            bank_expansion_factor=options.weight_bank_expansion_factor,
        )
        bias_generation = (
            build_mixture_generation_config(
                options.bias_generation,
                self.adaptive_generator_stack_config_factory,
            )
            if options.bias_option is MatrixBiasMixtureConfig
            else None
        )
        bias_config = build_bias_config(
            generation=bias_generation,
            bias_option=options.bias_option,
            decay_schedule=options.bias_decay_schedule,
            decay_rate=options.bias_decay_rate,
            decay_warmup_batches=options.bias_decay_warmup_batches,
            bank_expansion_factor=options.bias_bank_expansion_factor,
        )
        diagonal_config = build_diagonal_config(options.diagonal_option)
        mask_config = build_mask_config(options)
        augmentation = AdaptiveParameterAugmentationConfig(
            grouping_config=options.grouping_config,
            weight_config=weight_config,
            bias_config=bias_config,
            diagonal_config=diagonal_config,
            mask_config=mask_config,
            model_config=self.shared_adaptive_generator_stack_config,
        )
        return AdaptiveLinearLayerTemplateConfig(
            bias_flag=self.runtime.stack_options.bias_flag,
            adaptive_augmentation_config=augmentation,
        )
