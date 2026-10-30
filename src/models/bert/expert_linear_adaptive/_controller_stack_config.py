from emperor.config import ConfigBase
from emperor.layers import LayerConfig, LayerStackConfig
from emperor.linears import LinearLayerConfig
from emperor.memory import (
    AttentionDynamicMemoryConfig,
    DynamicMemoryConfig,
    MemoryPositionOptions,
)
from models.bert.expert_linear_adaptive.runtime_options import (
    ExpertsAdaptiveGeneratorStackOptions,
    ExpertsSubmoduleStackOptions,
    TransformerAttentionOptions,
)

from ._residual import build_residual_config


def build_controller_stack_config(
    options: ExpertsSubmoduleStackOptions | ExpertsAdaptiveGeneratorStackOptions,
    *,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
    layer_model_config: ConfigBase | None = None,
) -> LayerStackConfig:
    """Build a linear or adaptive controller stack for this Model Package."""

    return LayerStackConfig(
        hidden_dim=options.hidden_dim if hidden_dim is None else hidden_dim,
        output_dim=output_dim,
        num_layers=options.num_layers,
        last_layer_bias_option=options.last_layer_bias_option,
        apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
        layer_config=LayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            residual_config=build_residual_config(
                options.residual_connection_option,
                options.residual_model_flag,
                options.residual_stack_options,
                residual_block_size=options.residual_block_size,
                residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
            ),
            dropout_probability=options.dropout_probability,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=(
                LinearLayerConfig(bias_flag=options.bias_flag)
                if layer_model_config is None
                else layer_model_config
            ),
        ),
    )


def build_source_memory_config(
    options: TransformerAttentionOptions,
    stack_options: ExpertsSubmoduleStackOptions | None,
) -> DynamicMemoryConfig | None:
    """Build independent attention sources with the package's controller stack."""
    if not options.source_memory_flag:
        return None
    if stack_options is None:
        raise ValueError("Attention source memory requires submodule stack options.")
    config = options.source_memory_option(
        memory_position_option=MemoryPositionOptions.AFTER_AFFINE,
        model_config=build_controller_stack_config(stack_options),
        test_time_training_learning_rate=options.source_memory_test_time_training_learning_rate,
        test_time_training_num_inner_steps=options.source_memory_test_time_training_num_inner_steps,
    )
    if isinstance(config, AttentionDynamicMemoryConfig):
        config.num_memory_slots = options.source_memory_num_memory_slots
    return config
