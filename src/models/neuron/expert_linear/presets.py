from typing import Unpack

from emperor.config import BaseOptions
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.layers import (
    AdditiveResidualConfig,
    LayerNormPositionOptions,
)
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase
from models.neuron.expert_linear.config_builder import NeuronExpertLinearConfigBuilder
from models.neuron.expert_linear.runtime_defaults import runtime_from_flat
from models.neuron.expert_linear.runtime_options import PresetValues


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    GATING = 2
    HALTING = 3
    GATING_HALTING = 4
    RECURRENT = 5
    RECURRENT_GATING = 6
    RECURRENT_HALTING = 7
    RECURRENT_GATING_HALTING = 8
    SHARED_ROUTER_AFTER_WEIGHT = 9
    TOP1_SWITCH_AUX = 10
    TOP2_BALANCED_AUX = 11
    CAPACITY_TOP1_ZERO = 12
    CAPACITY_TOP1_IDENTITY = 13
    NOISY_SHARED_ROUTER = 14
    RESIDUAL_SHARED_ROUTER = 15
    POST_NORM_AFTER_WEIGHT = 16
    MEMORY = 17
    GATING_MEMORY = 18
    HALTING_MEMORY = 19
    GATING_HALTING_MEMORY = 20
    RECURRENT_MEMORY = 21
    RECURRENT_GATING_MEMORY = 22
    RECURRENT_HALTING_MEMORY = 23
    RECURRENT_GATING_HALTING_MEMORY = 24
    RESIDUAL = 25
    POST_NORM = 26
    RESIDUAL_POST_NORM = 27
    RESIDUAL_GATING = 28
    RESIDUAL_HALTING = 29
    RESIDUAL_MEMORY = 30
    RECURRENT_RESIDUAL = 31
    RECURRENT_POST_NORM = 32


def _define_preset(
    *,
    description: str,
    **values: Unpack[PresetValues],
) -> PresetDefinition:
    return PresetDefinition(preset_values=values, description=description)


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: _define_preset(
        description=(
            "Default config: a mixture-of-experts classifier with linear "
            "expert and router stacks."
        ),
    ),
    ExperimentPreset.GATING: _define_preset(
        description="Default config with per-layer gating enabled in the expert stack.",
        stack_gate_flag=True,
    ),
    ExperimentPreset.HALTING: _define_preset(
        description="Default config with stack halting enabled in the expert stack.",
        stack_halting_flag=True,
    ),
    ExperimentPreset.GATING_HALTING: _define_preset(
        description=(
            "Default config with both per-layer gating and stack halting "
            "enabled in the expert stack."
        ),
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.MEMORY: _define_preset(
        description=(
            "Default config with shared stack memory enabled in the expert stack."
        ),
        memory_flag=True,
    ),
    ExperimentPreset.GATING_MEMORY: _define_preset(
        description=(
            "Default config with both per-layer gating and shared stack "
            "memory enabled in the expert stack."
        ),
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.HALTING_MEMORY: _define_preset(
        description=(
            "Default config with both stack halting and shared stack memory "
            "enabled in the expert stack."
        ),
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default config with per-layer gating, stack halting, and shared "
            "stack memory enabled in the expert stack."
        ),
        stack_gate_flag=True,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RESIDUAL: _define_preset(
        description=(
            "Default config with residual skip connections enabled in the expert stack."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.POST_NORM: _define_preset(
        description=(
            "Default config with layer norm applied after expert stack layers "
            "instead of before them."
        ),
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: _define_preset(
        description=(
            "Default config with residual skip connections and post-layer "
            "normalization enabled in the expert stack."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RESIDUAL_GATING: _define_preset(
        description=(
            "Default config with residual skip connections and per-layer "
            "gating enabled in the expert stack."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_gate_flag=True,
    ),
    ExperimentPreset.RESIDUAL_HALTING: _define_preset(
        description=(
            "Default config with residual skip connections and stack halting "
            "enabled in the expert stack."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_halting_flag=True,
    ),
    ExperimentPreset.RESIDUAL_MEMORY: _define_preset(
        description=(
            "Default config with residual skip connections and shared stack "
            "memory enabled in the expert stack."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT: _define_preset(
        description=(
            "Default config wrapped in fixed-step recurrence, reusing the expert "
            "stack for each recurrent step."
        ),
        recurrent_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING: _define_preset(
        description=(
            "Default recurrent config with step-level gating enabled after each "
            "recurrent update."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING: _define_preset(
        description=(
            "Default recurrent config with recurrent halting enabled, allowing "
            "early stopping before the max step count."
        ),
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: _define_preset(
        description=(
            "Default recurrent config with both step-level gating and recurrent "
            "halting enabled."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_MEMORY: _define_preset(
        description=(
            "Default recurrent config whose reused expert stack has shared "
            "memory enabled."
        ),
        recurrent_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: _define_preset(
        description=(
            "Default recurrent config with step-level gating and shared memory "
            "in the reused expert stack."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent config with recurrent halting and shared memory "
            "in the reused expert stack."
        ),
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent config with step-level gating, recurrent "
            "halting, and shared memory in the reused expert stack."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: _define_preset(
        description=(
            "Default recurrent config using a residual expert stack at each "
            "recurrent step."
        ),
        recurrent_flag=True,
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.RECURRENT_POST_NORM: _define_preset(
        description=(
            "Default recurrent config using a post-normalized expert stack at "
            "each recurrent step."
        ),
        recurrent_flag=True,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.SHARED_ROUTER_AFTER_WEIGHT: _define_preset(
        description=(
            "Default config with shared expert routing and expert weighting after "
            "expert outputs."
        ),
        routing_initialization_mode=RoutingInitializationMode.SHARED,
        weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
    ),
    ExperimentPreset.TOP1_SWITCH_AUX: _define_preset(
        description=(
            "Default config with top-1 switch routing and switch auxiliary loss "
            "enabled."
        ),
        top_k=1,
        sampler_normalize_probabilities_flag=False,
        sampler_switch_loss_weight=0.1,
    ),
    ExperimentPreset.TOP2_BALANCED_AUX: _define_preset(
        description="Default config with top-2 routing and balance auxiliary loss enabled.",
        top_k=2,
        sampler_coefficient_of_variation_loss_weight=0.1,
    ),
    ExperimentPreset.CAPACITY_TOP1_ZERO: _define_preset(
        description="Default config with top-1 capacity limiting and dropped tokens zeroed.",
        top_k=1,
        capacity_factor=1.0,
        dropped_token_behavior=DroppedTokenOptions.ZEROS,
        sampler_normalize_probabilities_flag=False,
    ),
    ExperimentPreset.CAPACITY_TOP1_IDENTITY: _define_preset(
        description=(
            "Default config with top-1 capacity limiting and dropped tokens "
            "preserved by identity."
        ),
        top_k=1,
        capacity_factor=1.0,
        dropped_token_behavior=DroppedTokenOptions.IDENTITY,
        sampler_normalize_probabilities_flag=False,
    ),
    ExperimentPreset.NOISY_SHARED_ROUTER: _define_preset(
        description=(
            "Default config with shared noisy top-k routing enabled for "
            "sampler and router."
        ),
        routing_initialization_mode=RoutingInitializationMode.SHARED,
        sampler_noisy_topk_flag=True,
        router_noisy_topk_flag=True,
    ),
    ExperimentPreset.RESIDUAL_SHARED_ROUTER: _define_preset(
        description=(
            "Default config with residual expert stack connections and shared "
            "expert routing."
        ),
        routing_initialization_mode=RoutingInitializationMode.SHARED,
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.POST_NORM_AFTER_WEIGHT: _define_preset(
        description=(
            "Default config with post-layer normalization and expert weighting "
            "after expert outputs."
        ),
        layer_norm_position=LayerNormPositionOptions.AFTER,
        weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=NeuronExpertLinearConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            runtime_factory=runtime_from_flat,
        )


class Experiment(ExperimentBase):
    """Package-local Run Experiment Adapter."""

    def __init__(
        self,
        preset: ExperimentPreset | None = None,
        experiment_task=None,
        *,
        model_package,
        run_artifacts=None,
    ) -> None:
        super().__init__(
            preset=preset,
            experiment_task=experiment_task,
            model_package=model_package,
            run_artifacts=run_artifacts,
        )
