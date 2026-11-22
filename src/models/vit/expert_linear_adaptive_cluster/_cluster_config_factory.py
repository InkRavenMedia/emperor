from emperor.config import ConfigBase
from emperor.halting import HaltingHiddenStateModeOptions, StickBreakingConfig
from emperor.layers import (
    ActivationOptions,
    AttentionResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.neuron import (
    AxonsConfig,
    NeuronClusterConfig,
    NeuronConfig,
    NucleusConfig,
    TerminalConfig,
    TerminalConnectionShapeOptions,
)
from emperor.sampler import RouterConfig, SamplerConfig

from .runtime_options import ClusterOptions


def _projection(
    width: int,
    outputs: int,
    last_bias: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
) -> LayerStackConfig:
    layer_model_config = LinearLayerConfig(bias_flag=True)
    layer_config = LayerConfig(
        activation=ActivationOptions.DISABLED,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        normalization=NormalizationOptions.RMS_NORM,
        layer_model_config=layer_model_config,
    )
    return LayerStackConfig(
        input_dim=width,
        hidden_dim=width,
        output_dim=outputs,
        num_layers=1,
        last_layer_bias_option=last_bias,
        apply_output_postprocessing_flag=False,
        layer_config=layer_config,
    )


def build_cluster_config(
    encoder: ConfigBase, width: int, options: ClusterOptions
) -> NeuronClusterConfig:
    xy_range = options.terminal_xy_axis_range
    z_range = options.terminal_z_axis_range
    connections = (2 * xy_range.value + 1) ** 2 * (2 * z_range.value + 1)
    model_config = _projection(width, connections)
    router_config = RouterConfig(
        input_dim=width,
        num_experts=connections,
        noisy_topk_flag=False,
        model_config=model_config,
    )
    sampler = SamplerConfig(
        num_experts=connections,
        top_k=options.terminal_top_k,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        normalize_probabilities_flag=options.terminal_top_k > 1,
        noisy_topk_flag=False,
        coefficient_of_variation_loss_weight=0.0,
        switch_loss_weight=0.0,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        router_config=router_config,
    )
    halting = None
    if options.halting_flag:
        halting_gate_config = _projection(width, 2, LastLayerBiasOptions.DISABLED)
        halting = StickBreakingConfig(
            input_dim=width,
            threshold=options.halting_threshold,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            dropout_probability=0.0,
            min_steps=options.halting_min_steps,
            ponder_cost_weight=options.halting_ponder_cost_weight,
            halting_gate_config=halting_gate_config,
        )
    nucleus_config = NucleusConfig(model_config=encoder)
    axons_config = AxonsConfig()
    sequence_residual_config = None
    if options.attention_residual_flag:
        sequence_residual_config = AttentionResidualConfig(
            residual_dim=width, block_size=1, rms_norm_epsilon=1e-06
        )
    terminal_config = TerminalConfig(
        input_dim=width,
        connection_shape=TerminalConnectionShapeOptions.BOX,
        xy_axis_range=xy_range,
        z_axis_range=z_range,
        sampler_config=sampler,
    )
    neuron_config = NeuronConfig(
        nucleus_config=nucleus_config,
        axons_config=axons_config,
        sequence_residual_config=sequence_residual_config,
        terminal_config=terminal_config,
    )
    return NeuronClusterConfig(
        sequence_routing_mode=options.routing_mode,
        x_axis_total_neurons=options.x_axis_total_neurons,
        y_axis_total_neurons=options.y_axis_total_neurons,
        z_axis_total_neurons=options.z_axis_total_neurons,
        initial_x_axis_total_neurons=options.initial_x_axis_total_neurons,
        initial_y_axis_total_neurons=options.initial_y_axis_total_neurons,
        initial_z_axis_total_neurons=options.initial_z_axis_total_neurons,
        max_steps=options.max_steps,
        halting_config=halting,
        growth_threshold=options.growth_threshold,
        growth_warmup_steps=options.growth_warmup_steps,
        pruning_threshold=options.pruning_threshold,
        neuron_config=neuron_config,
    )
