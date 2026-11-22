from typing import Unpack

from emperor.config import BaseOptions
from emperor.layers import (
    AdditiveResidualConfig,
    AttentionResidualConfig,
    LayerNormPositionOptions,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.memory import (
    ElementWiseWeightedDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,
)
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase
from models.linears.linear.config_builder import LinearConfigBuilder
from models.linears.linear.runtime_defaults import build_settings_from_flat
from models.linears.linear.runtime_options import LinearPresetValues


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
    WEIGHTED_RESIDUAL = 25
    WEIGHTED_BLEND_RESIDUAL = 26
    ATTENTION_RESIDUAL = 27
    RECURRENT_LAYER_GATING = 28
    RECURRENT_DUAL_GATING = 29
    RECURRENT_LAYER_HALTING = 30
    RECURRENT_DUAL_HALTING = 31
    WEIGHTED_MEMORY = 32
    ELEMENT_WISE_WEIGHTED_MEMORY = 33
    NO_NORM = 34
    PRE_ACTIVATION_NORM = 35


def _define_preset(
    *,
    description: str,
    **values: Unpack[LinearPresetValues],
) -> PresetDefinition:
    return PresetDefinition(preset_values=values, description=description)


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: _define_preset(
        description="Default config: a GELU hidden linear stack with pre-layer norm "
        "and dropout.",
    ),
    ExperimentPreset.GATING: _define_preset(
        description="Default config with per-layer gating enabled, so each hidden "
        "layer output is modulated by a learned sigmoid gate.",
        stack_gate_flag=True,
    ),
    ExperimentPreset.HALTING: _define_preset(
        description="Default config with stack halting enabled, so examples can stop "
        "early as they move through the hidden stack.",
        stack_halting_flag=True,
    ),
    ExperimentPreset.MEMORY: _define_preset(
        description="Default config with shared stack memory enabled across the hidden "
        "layers.",
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING: _define_preset(
        description="Default config with both per-layer gating and stack halting "
        "enabled.",
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.GATING_MEMORY: _define_preset(
        description="Default config with both per-layer gating and shared stack memory "
        "enabled.",
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.HALTING_MEMORY: _define_preset(
        description="Default config with both stack halting and shared stack memory "
        "enabled.",
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: _define_preset(
        description="Default config with per-layer gating, stack halting, and shared "
        "stack memory enabled.",
        stack_gate_flag=True,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RESIDUAL: _define_preset(
        description="Default config with residual skip connections enabled between "
        "same-width hidden layers.",
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.POST_NORM: _define_preset(
        description="Default config with layer norm applied after each layer instead "
        "of before it.",
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: _define_preset(
        description="Default config with residual skip connections and post-layer "
        "normalization enabled.",
        stack_residual_connection_option=AdditiveResidualConfig,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RESIDUAL_GATING: _define_preset(
        description="Default config with residual skip connections and per-layer "
        "gating enabled.",
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_gate_flag=True,
    ),
    ExperimentPreset.RESIDUAL_HALTING: _define_preset(
        description="Default config with residual skip connections and stack halting "
        "enabled.",
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_halting_flag=True,
    ),
    ExperimentPreset.RESIDUAL_MEMORY: _define_preset(
        description="Default config with residual skip connections and shared stack "
        "memory enabled.",
        stack_residual_connection_option=AdditiveResidualConfig,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT: _define_preset(
        description="Default config wrapped in fixed-step recurrence, reusing the "
        "hidden stack for each recurrent step.",
        recurrent_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING: _define_preset(
        description="Default recurrent config with step-level gating enabled after "
        "each recurrent update.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING: _define_preset(
        description="Default recurrent config with recurrent halting enabled, allowing "
        "early stopping before the max step count.",
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_MEMORY: _define_preset(
        description="Default recurrent config whose reused hidden stack has shared "
        "memory enabled.",
        recurrent_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: _define_preset(
        description="Default recurrent config with both step-level gating and "
        "recurrent halting enabled.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: _define_preset(
        description="Default recurrent config with step-level gating and shared memory "
        "in the reused hidden stack.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: _define_preset(
        description="Default recurrent config with recurrent halting and shared memory "
        "in the reused hidden stack.",
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: _define_preset(
        description="Default recurrent config with step-level gating, recurrent "
        "halting, and shared memory in the reused hidden stack.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: _define_preset(
        description="Default recurrent config using a residual hidden stack at each "
        "recurrent step.",
        recurrent_flag=True,
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.RECURRENT_POST_NORM: _define_preset(
        description="Default recurrent config using a post-normalized hidden stack at "
        "each recurrent step.",
        recurrent_flag=True,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.WEIGHTED_RESIDUAL: _define_preset(
        description="Default config with a learned sigmoid-scaled current contribution "
        "composed with the previous hidden state at each hidden layer.",
        stack_residual_connection_option=WeightedResidualConfig,
    ),
    ExperimentPreset.WEIGHTED_BLEND_RESIDUAL: _define_preset(
        description="Default config with a learned bounded convex blend between "
        "current and previous hidden states at each hidden layer.",
        stack_residual_connection_option=WeightedBlendResidualConfig,
    ),
    ExperimentPreset.ATTENTION_RESIDUAL: _define_preset(
        description="Default config with depth-local attention over compatible "
        "residual sources in the hidden stack.",
        stack_residual_connection_option=AttentionResidualConfig,
    ),
    ExperimentPreset.RECURRENT_LAYER_GATING: _define_preset(
        description="Recurrent config with per-layer gating inside the reused hidden "
        "stack and no outer recurrent-step gate.",
        recurrent_flag=True,
        stack_gate_flag=True,
    ),
    ExperimentPreset.RECURRENT_DUAL_GATING: _define_preset(
        description="Recurrent config with separate inner per-layer gates and an outer "
        "recurrent-step gate.",
        recurrent_flag=True,
        stack_gate_flag=True,
        recurrent_stack_gate_flag=True,
    ),
    ExperimentPreset.RECURRENT_LAYER_HALTING: _define_preset(
        description="Recurrent config with halting inside the reused hidden stack "
        "while the outer recurrence retains a fixed maximum step budget.",
        recurrent_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_DUAL_HALTING: _define_preset(
        description="Recurrent config with separate inner-stack and outer "
        "recurrent-step stick-breaking halting controllers.",
        recurrent_flag=True,
        stack_halting_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.WEIGHTED_MEMORY: _define_preset(
        description="Default config with shared stack memory using a sample- and "
        "position-level weighted merge.",
        memory_flag=True,
        memory_option=WeightedDynamicMemoryConfig,
    ),
    ExperimentPreset.ELEMENT_WISE_WEIGHTED_MEMORY: _define_preset(
        description="Default config with shared stack memory using a per-feature "
        "weighted merge.",
        memory_flag=True,
        memory_option=ElementWiseWeightedDynamicMemoryConfig,
    ),
    ExperimentPreset.NO_NORM: _define_preset(
        description="Default config with no layer normalization in the main hidden "
        "stack.",
        layer_norm_position=LayerNormPositionOptions.DISABLED,
    ),
    ExperimentPreset.PRE_ACTIVATION_NORM: _define_preset(
        description="Default config with normalization after affine and memory work "
        "but before activation, distinct from pre-layer and post-layer normalization.",
        layer_norm_position=LayerNormPositionOptions.DEFAULT,
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=LinearConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            runtime_factory=build_settings_from_flat,
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
