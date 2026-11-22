from typing import Unpack

from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
    DiagonalAxisMaskConfig,
    DualModelDynamicWeightConfig,
    GeneratorDynamicBiasConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    MultiplicativeDynamicBiasConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    SigmoidGatedDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    StandardDynamicDiagonalConfig,
    TanhGatedDynamicBiasConfig,
    TopSliceAxisMaskConfig,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
    WeightInformedScoreAxisMaskConfig,
    WeightNormalizationOptions,
)
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
from models.linears.linear_adaptive.config_builder import LinearAdaptiveConfigBuilder
from models.linears.linear_adaptive.runtime_defaults import runtime_from_flat
from models.linears.linear_adaptive.runtime_options import PresetValues


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
    SINGLE_MODEL_WEIGHT = 15
    DUAL_MODEL_WEIGHT = 16
    LOW_RANK_WEIGHT = 17
    HYPERNETWORK_WEIGHT = 18
    LAYERED_WEIGHTED_BANK_WEIGHT = 19
    SOFT_WEIGHTED_BANK_WEIGHT = 20
    AFFINE_TRANSFORM_BIAS = 21
    ADDITIVE_BIAS = 22
    GENERATOR_BIAS = 23
    MULTIPLICATIVE_BIAS = 24
    SIGMOID_GATED_BIAS = 25
    TANH_GATED_BIAS = 26
    WEIGHTED_BANK_BIAS = 27
    STANDARD_DIAGONAL = 28
    ANTI_DIAGONAL = 29
    COMBINED_DIAGONAL = 30
    DIAGONAL_AXIS_MASK = 31
    OUTER_PRODUCT_MASK = 32
    PER_AXIS_SCORE_MASK = 33
    TOP_SLICE_AXIS_MASK = 34
    WEIGHT_INFORMED_SCORE_MASK = 35
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 36
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 37
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 38
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 39
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 40
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 41
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 42
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 43
    DECAY_EXPONENTIAL_WEIGHT = 44
    NORM_L2_WEIGHT = 45
    DEEP_GENERATOR = 46
    FULL_STACK = 47
    DUAL_WEIGHT_GATING = 48
    DUAL_WEIGHT_HALTING = 49
    DUAL_WEIGHT_GATING_HALTING = 50
    DUAL_WEIGHT_MEMORY = 51
    DUAL_WEIGHT_GATING_MEMORY = 52
    DUAL_WEIGHT_HALTING_MEMORY = 53
    FULL_STACK_GATING = 54
    FULL_STACK_HALTING = 55
    FULL_STACK_MEMORY = 56
    FULL_STACK_GATING_HALTING = 57
    FULL_STACK_RECURRENT = 58
    BANK_WEIGHT_MASK = 59
    LOW_RANK_POST_NORM = 60
    RECURRENT = 61
    RECURRENT_GATING = 62
    RECURRENT_HALTING = 63
    RECURRENT_MEMORY = 64
    RECURRENT_GATING_HALTING = 65
    RECURRENT_GATING_MEMORY = 66
    RECURRENT_HALTING_MEMORY = 67
    RECURRENT_GATING_HALTING_MEMORY = 68
    RECURRENT_RESIDUAL = 69
    RECURRENT_POST_NORM = 70


_ADAPTIVE_OPTION_FLAGS = {
    "weight_option": "weight_option_flag",
    "bias_option": "bias_option_flag",
    "diagonal_option": "diagonal_option_flag",
    "row_mask_option": "mask_option_flag",
}


_FULL_STACK_VALUES = PresetValues(
    weight_option=DualModelDynamicWeightConfig,
    bias_option=AdditiveDynamicBiasConfig,
    diagonal_option=CombinedDynamicDiagonalConfig,
    row_mask_option=WeightInformedScoreAxisMaskConfig,
)


def _define_preset(
    *,
    description: str,
    base: PresetValues | None = None,
    **values: Unpack[PresetValues],
) -> PresetDefinition:
    if base is not None:
        values = {**base, **values}
    enabled_flags = {
        flag: True
        for option, flag in _ADAPTIVE_OPTION_FLAGS.items()
        if values.get(option) is not None
    }
    return PresetDefinition(
        preset_values={**enabled_flags, **values},
        description=description,
    )


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: _define_preset(
        description=(
            "Default config: a GELU adaptive linear stack with pre-layer norm and "
            "dropout."
        ),
    ),
    ExperimentPreset.GATING: _define_preset(
        description=(
            "Default adaptive config with per-layer gating enabled, so each hidden "
            "layer output is modulated by a learned sigmoid gate."
        ),
        stack_gate_flag=True,
    ),
    ExperimentPreset.HALTING: _define_preset(
        description=(
            "Default adaptive config with stack halting enabled, so examples can "
            "stop early as they move through the adaptive hidden stack."
        ),
        stack_halting_flag=True,
    ),
    ExperimentPreset.MEMORY: _define_preset(
        description=(
            "Default adaptive config with shared stack memory enabled across the "
            "hidden layers."
        ),
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING: _define_preset(
        description=(
            "Default adaptive config with both per-layer gating and stack halting "
            "enabled."
        ),
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.GATING_MEMORY: _define_preset(
        description=(
            "Default adaptive config with both per-layer gating and shared stack "
            "memory enabled."
        ),
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.HALTING_MEMORY: _define_preset(
        description=(
            "Default adaptive config with both stack halting and shared stack "
            "memory enabled."
        ),
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default adaptive config with per-layer gating, stack halting, and "
            "shared stack memory enabled."
        ),
        stack_gate_flag=True,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RESIDUAL: _define_preset(
        description=(
            "Default adaptive config with residual skip connections "
            "enabled between same-width hidden layers."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.POST_NORM: _define_preset(
        description=(
            "Default adaptive config with layer norm applied after each layer "
            "instead of before it."
        ),
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: _define_preset(
        description=(
            "Default adaptive config with residual skip connections and post-layer "
            "normalization enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RESIDUAL_GATING: _define_preset(
        description=(
            "Default adaptive config with residual skip connections and per-layer "
            "gating enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_gate_flag=True,
    ),
    ExperimentPreset.RESIDUAL_HALTING: _define_preset(
        description=(
            "Default adaptive config with residual skip connections and stack "
            "halting enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_halting_flag=True,
    ),
    ExperimentPreset.RESIDUAL_MEMORY: _define_preset(
        description=(
            "Default adaptive config with residual skip connections and shared "
            "stack memory enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        memory_flag=True,
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT: _define_preset(
        description=(
            "Default adaptive config with the single-model dynamic "
            "weight generator enabled."
        ),
        weight_option=SingleModelDynamicWeightConfig,
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT: _define_preset(
        description=(
            "Default adaptive config with the dual-model dynamic weight generator "
            "enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
    ),
    ExperimentPreset.LOW_RANK_WEIGHT: _define_preset(
        description=(
            "Default adaptive config with the low-rank dynamic weight generator "
            "enabled."
        ),
        weight_option=LowRankDynamicWeightConfig,
    ),
    ExperimentPreset.HYPERNETWORK_WEIGHT: _define_preset(
        description=(
            "Default adaptive config with the hypernetwork dynamic "
            "weight generator enabled."
        ),
        weight_option=HypernetworkDynamicWeightConfig,
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT: _define_preset(
        description=(
            "Default adaptive config with the layered weighted-bank dynamic weight "
            "generator enabled."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
    ),
    ExperimentPreset.SOFT_WEIGHTED_BANK_WEIGHT: _define_preset(
        description=(
            "Default adaptive config with the soft weighted-bank dynamic weight "
            "generator enabled."
        ),
        weight_option=SoftWeightedBankDynamicWeightConfig,
    ),
    ExperimentPreset.AFFINE_TRANSFORM_BIAS: _define_preset(
        description="Default adaptive config with affine-transform dynamic bias enabled.",
        bias_option=AffineTransformDynamicBiasConfig,
    ),
    ExperimentPreset.ADDITIVE_BIAS: _define_preset(
        description="Default adaptive config with additive dynamic bias enabled.",
        bias_option=AdditiveDynamicBiasConfig,
    ),
    ExperimentPreset.GENERATOR_BIAS: _define_preset(
        description="Default adaptive config with generator-based dynamic bias enabled.",
        bias_option=GeneratorDynamicBiasConfig,
    ),
    ExperimentPreset.MULTIPLICATIVE_BIAS: _define_preset(
        description="Default adaptive config with multiplicative dynamic bias enabled.",
        bias_option=MultiplicativeDynamicBiasConfig,
    ),
    ExperimentPreset.SIGMOID_GATED_BIAS: _define_preset(
        description="Default adaptive config with sigmoid-gated dynamic bias enabled.",
        bias_option=SigmoidGatedDynamicBiasConfig,
    ),
    ExperimentPreset.TANH_GATED_BIAS: _define_preset(
        description="Default adaptive config with tanh-gated dynamic bias enabled.",
        bias_option=TanhGatedDynamicBiasConfig,
    ),
    ExperimentPreset.WEIGHTED_BANK_BIAS: _define_preset(
        description="Default adaptive config with weighted-bank dynamic bias enabled.",
        bias_option=WeightedBankDynamicBiasConfig,
    ),
    ExperimentPreset.STANDARD_DIAGONAL: _define_preset(
        description="Default adaptive config with standard dynamic diagonal enabled.",
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.ANTI_DIAGONAL: _define_preset(
        description="Default adaptive config with anti dynamic diagonal enabled.",
        diagonal_option=AntiDynamicDiagonalConfig,
    ),
    ExperimentPreset.COMBINED_DIAGONAL: _define_preset(
        description="Default adaptive config with combined dynamic diagonal enabled.",
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.DIAGONAL_AXIS_MASK: _define_preset(
        description="Default adaptive config with diagonal-axis row masking enabled.",
        row_mask_option=DiagonalAxisMaskConfig,
    ),
    ExperimentPreset.OUTER_PRODUCT_MASK: _define_preset(
        description="Default adaptive config with outer-product row masking enabled.",
        row_mask_option=OuterProductMaskConfig,
    ),
    ExperimentPreset.PER_AXIS_SCORE_MASK: _define_preset(
        description="Default adaptive config with per-axis score row masking enabled.",
        row_mask_option=PerAxisScoreMaskConfig,
    ),
    ExperimentPreset.TOP_SLICE_AXIS_MASK: _define_preset(
        description="Default adaptive config with top-slice axis row masking enabled.",
        row_mask_option=TopSliceAxisMaskConfig,
    ),
    ExperimentPreset.WEIGHT_INFORMED_SCORE_MASK: _define_preset(
        description=(
            "Default adaptive config with weight-informed score axis row masking "
            "enabled."
        ),
        row_mask_option=WeightInformedScoreAxisMaskConfig,
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _define_preset(
        description=(
            "Default adaptive config with single-model dynamic weights, additive "
            "bias, and combined dynamic diagonal enabled."
        ),
        weight_option=SingleModelDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights, additive "
            "bias, and combined dynamic diagonal enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _define_preset(
        description=(
            "Default adaptive config with layered weighted-bank dynamic weights, "
            "additive bias, and combined dynamic diagonal enabled."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _define_preset(
        description=(
            "Default adaptive config with low-rank dynamic weights, additive bias, "
            "and combined dynamic diagonal enabled."
        ),
        weight_option=LowRankDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _define_preset(
        description=(
            "Default adaptive config with single-model dynamic weights, additive "
            "bias, and standard dynamic diagonal enabled."
        ),
        weight_option=SingleModelDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights, additive "
            "bias, and standard dynamic diagonal enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _define_preset(
        description=(
            "Default adaptive config with layered weighted-bank dynamic weights, "
            "additive bias, and standard dynamic diagonal enabled."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _define_preset(
        description=(
            "Default adaptive config with low-rank dynamic weights, additive bias, "
            "and standard dynamic diagonal enabled."
        ),
        weight_option=LowRankDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.DECAY_EXPONENTIAL_WEIGHT: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights decaying "
            "exponentially toward a static linear layer."
        ),
        weight_option=DualModelDynamicWeightConfig,
        weight_decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
        weight_decay_rate=0.001,
        weight_decay_warmup_batches=500,
    ),
    ExperimentPreset.NORM_L2_WEIGHT: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights and L2-scale "
            "weight normalization."
        ),
        weight_option=DualModelDynamicWeightConfig,
        weight_normalization_option=WeightNormalizationOptions.L2_SCALE,
    ),
    ExperimentPreset.DEEP_GENERATOR: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights produced by a "
            "depth-8 generator network."
        ),
        weight_option=DualModelDynamicWeightConfig,
        generator_depth=8,
    ),
    ExperimentPreset.FULL_STACK: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights, additive "
            "bias, combined dynamic diagonal, and weight-informed row masking "
            "enabled."
        ),
        base=_FULL_STACK_VALUES,
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights and per-layer "
            "gating enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_gate_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights and stack "
            "halting enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_halting_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING_HALTING: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights, per-layer "
            "gating, and stack halting enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_MEMORY: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights and shared "
            "stack memory enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        memory_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING_MEMORY: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights, per-layer "
            "gating, and shared stack memory enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING_MEMORY: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights, stack "
            "halting, and shared stack memory enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.FULL_STACK_GATING: _define_preset(
        description=(
            "Default adaptive config with full adaptive parameter controls and "
            "per-layer gating enabled."
        ),
        base=_FULL_STACK_VALUES,
        stack_gate_flag=True,
    ),
    ExperimentPreset.FULL_STACK_HALTING: _define_preset(
        description=(
            "Default adaptive config with full adaptive parameter controls and "
            "stack halting enabled."
        ),
        base=_FULL_STACK_VALUES,
        stack_halting_flag=True,
    ),
    ExperimentPreset.FULL_STACK_MEMORY: _define_preset(
        description=(
            "Default adaptive config with full adaptive parameter controls and "
            "shared stack memory enabled."
        ),
        base=_FULL_STACK_VALUES,
        memory_flag=True,
    ),
    ExperimentPreset.FULL_STACK_GATING_HALTING: _define_preset(
        description=(
            "Default adaptive config with full adaptive parameter controls, "
            "per-layer gating, and stack halting enabled."
        ),
        base=_FULL_STACK_VALUES,
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.FULL_STACK_RECURRENT: _define_preset(
        description=(
            "Default adaptive config with full adaptive parameter controls wrapped "
            "in fixed-step recurrence."
        ),
        base=_FULL_STACK_VALUES,
        recurrent_flag=True,
    ),
    ExperimentPreset.BANK_WEIGHT_MASK: _define_preset(
        description=(
            "Default adaptive config with layered weighted-bank dynamic "
            "weights and weight-informed row masking enabled."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
        row_mask_option=WeightInformedScoreAxisMaskConfig,
    ),
    ExperimentPreset.LOW_RANK_POST_NORM: _define_preset(
        description=(
            "Default adaptive config with low-rank dynamic weights and post-layer "
            "normalization enabled."
        ),
        weight_option=LowRankDynamicWeightConfig,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RECURRENT: _define_preset(
        description=(
            "Default adaptive config wrapped in fixed-step recurrence, reusing the "
            "adaptive linear stack for each recurrent step."
        ),
        recurrent_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING: _define_preset(
        description=(
            "Default recurrent adaptive config with step-level gating "
            "enabled after each recurrent update."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING: _define_preset(
        description=(
            "Default recurrent adaptive config with recurrent halting enabled, "
            "allowing early stopping before the max step count."
        ),
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_MEMORY: _define_preset(
        description=(
            "Default recurrent adaptive config whose reused hidden stack "
            "has shared memory enabled."
        ),
        recurrent_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: _define_preset(
        description=(
            "Default recurrent adaptive config with both step-level gating and "
            "recurrent halting enabled."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: _define_preset(
        description=(
            "Default recurrent adaptive config with step-level gating and shared "
            "memory in the reused adaptive stack."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent adaptive config with recurrent halting and shared "
            "memory in the reused adaptive stack."
        ),
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent adaptive config with step-level gating, recurrent "
            "halting, and shared memory in the reused adaptive stack."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: _define_preset(
        description=(
            "Default recurrent adaptive config using a residual hidden stack at "
            "each recurrent step."
        ),
        recurrent_flag=True,
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.RECURRENT_POST_NORM: _define_preset(
        description=(
            "Default recurrent adaptive config using a post-normalized "
            "hidden stack at each recurrent step."
        ),
        recurrent_flag=True,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=LinearAdaptiveConfigBuilder,
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
