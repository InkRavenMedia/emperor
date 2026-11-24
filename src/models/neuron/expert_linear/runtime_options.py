from dataclasses import dataclass, field
from typing import TypedDict

from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.neuron import (
    TerminalRangeOptions,
    TerminalRoutingTreeDepthOptions,
)
from model_runtime.packages.runtime_values import ResolvedRuntimeOptions
from models._recurrent_variants import RecurrentResidualOptions
from models.neuron.expert_linear import SETTINGS_VALIDATION
from models.neuron.expert_linear._hidden.runtime_options import (
    RuntimeOptions as HiddenRuntimeOptions,
)
from models.neuron.expert_linear._residual import ResidualStackOptions


@dataclass(frozen=True)
class NeuronClusterCapacityOptions:
    x_axis_total_neurons: int
    y_axis_total_neurons: int
    z_axis_total_neurons: int
    initial_x_axis_total_neurons: int | None
    initial_y_axis_total_neurons: int | None
    initial_z_axis_total_neurons: int | None
    max_steps: int
    growth_threshold: int | None
    beam_width: int | None = None
    growth_cooldown_steps: int | None = None
    max_total_growths: int | None = None
    growth_warmup_steps: int | None = None
    pruning_threshold: int | None = None
    escape_driven_growth_flag: bool = False
    mitosis_initialization_flag: bool = False


@dataclass(frozen=True)
class NeuronTerminalRoutingTreeOptions:
    depth: TerminalRoutingTreeDepthOptions
    direction_branch_counts: tuple[int, ...]
    direction_top_k: tuple[int, ...]


@dataclass(frozen=True)
class NeuronTerminalOptions:
    xy_axis_range: TerminalRangeOptions
    z_axis_range: TerminalRangeOptions
    top_k: int
    routing_tree: NeuronTerminalRoutingTreeOptions | None = None


@dataclass(frozen=True)
class NeuronSubmoduleStackOptions:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float
    bias_flag: bool
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.positive_integer("stack_hidden_dim", self.hidden_dim)
        SETTINGS_VALIDATION.stack(self)


@dataclass(frozen=True)
class NeuronTerminalSamplerOptions:
    threshold: float
    filter_above_threshold: bool
    num_topk_samples: int
    normalize_probabilities_flag: bool
    noisy_topk_flag: bool
    coefficient_of_variation_loss_weight: float
    switch_loss_weight: float
    zero_centred_loss_weight: float
    mutual_information_loss_weight: float


@dataclass(frozen=True)
class ClusterRouteHaltingOptions:
    enabled: bool
    threshold: float
    dropout: float
    hidden_state_mode: HaltingHiddenStateModeOptions
    stack_options: NeuronSubmoduleStackOptions
    output_dim: int
    halting_option: type[HaltingConfig] = StickBreakingConfig

    def __post_init__(self) -> None:
        SETTINGS_VALIDATION.halting_threshold(
            "cluster_halting_threshold", self.threshold
        )
        SETTINGS_VALIDATION.dropout("cluster_halting_dropout", self.dropout)
        SETTINGS_VALIDATION.positive_integer(
            "cluster_halting_output_dim", self.output_dim
        )


@dataclass(frozen=True, slots=True)
class RuntimeOptions(ResolvedRuntimeOptions, RecurrentResidualOptions):
    """Resolved package-owned neuron construction roles."""

    def __post_init__(self) -> None:
        ResolvedRuntimeOptions.__post_init__(self)
        for name, expected in (
            ("hidden_runtime", HiddenRuntimeOptions),
            ("cluster_capacity_options", NeuronClusterCapacityOptions),
            ("terminal_options", NeuronTerminalOptions),
            ("terminal_router_options", NeuronSubmoduleStackOptions),
            ("terminal_sampler_options", NeuronTerminalSamplerOptions),
            ("cluster_halting_options", ClusterRouteHaltingOptions),
        ):
            if not isinstance(self._values.get(name), expected):
                raise TypeError(f"{name} must be a {expected.__name__} value")


class PresetValues(TypedDict, total=False):
    capacity_factor: float
    dropped_token_behavior: DroppedTokenOptions
    layer_norm_position: LayerNormPositionOptions
    memory_flag: bool
    recurrent_flag: bool
    recurrent_stack_gate_flag: bool
    recurrent_stack_halting_flag: bool
    router_noisy_topk_flag: bool
    routing_initialization_mode: RoutingInitializationMode
    sampler_coefficient_of_variation_loss_weight: float
    sampler_noisy_topk_flag: bool
    sampler_normalize_probabilities_flag: bool
    sampler_switch_loss_weight: float
    stack_gate_flag: bool
    stack_halting_flag: bool
    stack_residual_connection_option: type[ResidualConfig] | None
    top_k: int
    weighting_position_option: ExpertWeightingPositionOptions
