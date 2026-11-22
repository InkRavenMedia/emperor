# ruff: noqa: E501

from typing import Unpack

from emperor.config import BaseOptions
from emperor.layers import (
    AdditiveResidualConfig,
    LayerNormPositionOptions,
)
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase
from models.neuron.linear.config_builder import NeuronLinearConfigBuilder
from models.neuron.linear.runtime_defaults import runtime_from_flat
from models.neuron.linear.runtime_options import PresetValues


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    GATING = 2
    HALTING = 3
    MEMORY = 4
    GATING_HALTING = 5
    GATING_MEMORY = 6
    HALTING_MEMORY = 7
    GATING_HALTING_MEMORY = 8
    RESIDUAL = 9
    POST_NORM = 10
    RESIDUAL_POST_NORM = 11
    RESIDUAL_GATING = 12
    RESIDUAL_HALTING = 13
    RESIDUAL_MEMORY = 14
    RECURRENT = 15
    RECURRENT_GATING = 16
    RECURRENT_HALTING = 17
    RECURRENT_MEMORY = 18
    RECURRENT_GATING_HALTING = 19
    RECURRENT_GATING_MEMORY = 20
    RECURRENT_HALTING_MEMORY = 21
    RECURRENT_GATING_HALTING_MEMORY = 22
    RECURRENT_RESIDUAL = 23
    RECURRENT_POST_NORM = 24


def _define_preset(
    *,
    description: str,
    **values: Unpack[PresetValues],
) -> PresetDefinition:
    return PresetDefinition(preset_values=values, description=description)


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: _define_preset(
        description=(
            "Default config: a GELU hidden linear stack with pre-layer norm and "
            "dropout."
        ),
    ),
    ExperimentPreset.GATING: _define_preset(
        description=(
            "Default config with per-layer gating enabled, so each hidden layer "
            "output is modulated by a learned sigmoid gate."
        ),
        stack_gate_flag=True,
    ),
    ExperimentPreset.HALTING: _define_preset(
        description=(
            "Default config with stack halting enabled, so examples can stop early "
            "as they move through the hidden stack."
        ),
        stack_halting_flag=True,
    ),
    ExperimentPreset.MEMORY: _define_preset(
        description=(
            "Default config with shared stack memory enabled across the hidden layers."
        ),
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING: _define_preset(
        description="Default config with both per-layer gating and stack halting enabled.",
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.GATING_MEMORY: _define_preset(
        description=(
            "Default config with both per-layer gating and shared stack memory enabled."
        ),
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.HALTING_MEMORY: _define_preset(
        description=(
            "Default config with both stack halting and shared stack memory enabled."
        ),
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default config with per-layer gating, stack halting, and shared stack "
            "memory enabled."
        ),
        stack_gate_flag=True,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RESIDUAL: _define_preset(
        description=(
            "Default config with residual skip connections enabled between "
            "same-width hidden layers."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.POST_NORM: _define_preset(
        description=(
            "Default config with layer norm applied after each layer instead of "
            "before it."
        ),
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: _define_preset(
        description=(
            "Default config with residual skip connections and post-layer "
            "normalization enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RESIDUAL_GATING: _define_preset(
        description=(
            "Default config with residual skip connections and per-layer gating "
            "enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_gate_flag=True,
    ),
    ExperimentPreset.RESIDUAL_HALTING: _define_preset(
        description=(
            "Default config with residual skip connections and stack halting enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_halting_flag=True,
    ),
    ExperimentPreset.RESIDUAL_MEMORY: _define_preset(
        description=(
            "Default config with residual skip connections and shared stack memory "
            "enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT: _define_preset(
        description=(
            "Default config wrapped in fixed-step recurrence, reusing the hidden "
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
    ExperimentPreset.RECURRENT_MEMORY: _define_preset(
        description=(
            "Default recurrent config whose reused hidden stack has shared memory "
            "enabled."
        ),
        recurrent_flag=True,
        memory_flag=True,
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
    ExperimentPreset.RECURRENT_GATING_MEMORY: _define_preset(
        description=(
            "Default recurrent config with step-level gating and shared memory in "
            "the reused hidden stack."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent config with recurrent halting and shared memory in "
            "the reused hidden stack."
        ),
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent config with step-level gating, recurrent halting, "
            "and shared memory in the reused hidden stack."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: _define_preset(
        description=(
            "Default recurrent config using a residual hidden stack at each "
            "recurrent step."
        ),
        recurrent_flag=True,
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.RECURRENT_POST_NORM: _define_preset(
        description=(
            "Default recurrent config using a post-normalized hidden stack at each "
            "recurrent step."
        ),
        recurrent_flag=True,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=NeuronLinearConfigBuilder,
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
