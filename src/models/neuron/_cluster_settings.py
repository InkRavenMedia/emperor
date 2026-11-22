from models._expert_mixture import SamplerSettings
from models._linear_stack import LayerStackSettings
from models.neuron._cluster_builder import (
    ClusterCapacitySettings,
    ClusterHaltingSettings,
    TerminalSettings,
)


class NeuronClusterSettings:
    cluster_capacity_options: ClusterCapacitySettings
    terminal_options: TerminalSettings
    terminal_router_options: LayerStackSettings
    terminal_sampler_options: SamplerSettings
    cluster_halting_options: ClusterHaltingSettings

    def _expose_flat_settings(self) -> None:
        self.cluster_x_axis_total_neurons = (
            self.cluster_capacity_options.x_axis_total_neurons
        )
        self.cluster_y_axis_total_neurons = (
            self.cluster_capacity_options.y_axis_total_neurons
        )
        self.cluster_z_axis_total_neurons = (
            self.cluster_capacity_options.z_axis_total_neurons
        )
        self.cluster_initial_x_axis_total_neurons = (
            self.cluster_capacity_options.initial_x_axis_total_neurons
        )
        self.cluster_initial_y_axis_total_neurons = (
            self.cluster_capacity_options.initial_y_axis_total_neurons
        )
        self.cluster_initial_z_axis_total_neurons = (
            self.cluster_capacity_options.initial_z_axis_total_neurons
        )
        self.cluster_max_steps = self.cluster_capacity_options.max_steps
        self.cluster_growth_threshold = self.cluster_capacity_options.growth_threshold
        self.cluster_terminal_xy_axis_range = self.terminal_options.xy_axis_range
        self.cluster_terminal_z_axis_range = self.terminal_options.z_axis_range
        self.cluster_terminal_top_k = self.terminal_options.top_k
        self.cluster_terminal_router_num_layers = (
            self.terminal_router_options.num_layers
        )
        self.cluster_terminal_router_hidden_dim = (
            self.terminal_router_options.hidden_dim
        )
        self.cluster_terminal_router_activation = (
            self.terminal_router_options.activation
        )
        self.cluster_terminal_router_layer_norm_position = (
            self.terminal_router_options.layer_norm_position
        )
        self.cluster_terminal_router_normalization = (
            self.terminal_router_options.normalization
        )
        self.cluster_terminal_router_residual_connection_option = (
            self.terminal_router_options.residual_connection_option
        )
        self.cluster_terminal_router_residual_model_flag = (
            self.terminal_router_options.residual_model_flag
        )
        self.cluster_terminal_router_residual_block_size = (
            self.terminal_router_options.residual_block_size
        )
        self.cluster_terminal_router_residual_rms_norm_epsilon = (
            self.terminal_router_options.residual_rms_norm_epsilon
        )
        self.cluster_terminal_router_dropout_probability = (
            self.terminal_router_options.dropout_probability
        )
        self.cluster_terminal_router_last_layer_bias_option = (
            self.terminal_router_options.last_layer_bias_option
        )
        self.cluster_terminal_router_apply_output_postprocessing_flag = (
            self.terminal_router_options.apply_output_postprocessing_flag
        )
        self.cluster_terminal_router_bias_flag = self.terminal_router_options.bias_flag
        self.cluster_terminal_sampler_threshold = (
            self.terminal_sampler_options.threshold
        )
        self.cluster_terminal_sampler_filter_above_threshold = (
            self.terminal_sampler_options.filter_above_threshold
        )
        self.cluster_terminal_sampler_num_topk_samples = (
            self.terminal_sampler_options.num_topk_samples
        )
        self.cluster_terminal_sampler_normalize_probabilities_flag = (
            self.terminal_sampler_options.normalize_probabilities_flag
        )
        self.cluster_terminal_sampler_noisy_topk_flag = (
            self.terminal_sampler_options.noisy_topk_flag
        )
        self.cluster_terminal_sampler_coefficient_of_variation_loss_weight = (
            self.terminal_sampler_options.coefficient_of_variation_loss_weight
        )
        self.cluster_terminal_sampler_switch_loss_weight = (
            self.terminal_sampler_options.switch_loss_weight
        )
        self.cluster_terminal_sampler_zero_centred_loss_weight = (
            self.terminal_sampler_options.zero_centred_loss_weight
        )
        self.cluster_terminal_sampler_mutual_information_loss_weight = (
            self.terminal_sampler_options.mutual_information_loss_weight
        )
        self.cluster_halting_flag = self.cluster_halting_options.enabled
        self.cluster_halting_threshold = self.cluster_halting_options.threshold
        self.cluster_halting_dropout = self.cluster_halting_options.dropout
        self.cluster_halting_hidden_state_mode = (
            self.cluster_halting_options.hidden_state_mode
        )
        self.cluster_halting_stack_options = self.cluster_halting_options.stack_options
        self.cluster_halting_stack_hidden_dim = (
            self.cluster_halting_stack_options.hidden_dim
        )
        self.cluster_halting_output_dim = self.cluster_halting_options.output_dim
        self.cluster_halting_stack_layer_norm_position = (
            self.cluster_halting_stack_options.layer_norm_position
        )
        self.cluster_halting_stack_normalization = (
            self.cluster_halting_stack_options.normalization
        )
        self.cluster_halting_stack_num_layers = (
            self.cluster_halting_stack_options.num_layers
        )
        self.cluster_halting_stack_activation = (
            self.cluster_halting_stack_options.activation
        )
        self.cluster_halting_stack_residual_connection_option = (
            self.cluster_halting_stack_options.residual_connection_option
        )
        self.cluster_halting_stack_residual_model_flag = (
            self.cluster_halting_stack_options.residual_model_flag
        )
        self.cluster_halting_stack_residual_block_size = (
            self.cluster_halting_stack_options.residual_block_size
        )
        self.cluster_halting_stack_residual_rms_norm_epsilon = (
            self.cluster_halting_stack_options.residual_rms_norm_epsilon
        )
        self.cluster_halting_stack_dropout_probability = (
            self.cluster_halting_stack_options.dropout_probability
        )
        self.cluster_halting_stack_last_layer_bias_option = (
            self.cluster_halting_stack_options.last_layer_bias_option
        )
        self.cluster_halting_stack_apply_output_postprocessing_flag = (
            self.cluster_halting_stack_options.apply_output_postprocessing_flag
        )
        self.cluster_halting_stack_bias_flag = (
            self.cluster_halting_stack_options.bias_flag
        )
