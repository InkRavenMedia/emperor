# ruff: noqa: E501

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
from models.neuron.expert_linear_adaptive.config_builder import (
    NeuronExpertLinearAdaptiveConfigBuilder,
)
from models.neuron.expert_linear_adaptive.runtime_defaults import runtime_from_flat
from models.neuron.expert_linear_adaptive.runtime_options import PresetValues


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    GATING = 2
    HALTING = 3
    GATING_HALTING = 4
    RECURRENT = 5
    RECURRENT_GATING = 6
    RECURRENT_HALTING = 7
    RECURRENT_GATING_HALTING = 8
    ADAPTIVE_SHARED_ROUTER = 9
    ADAPTIVE_AFTER_WEIGHT = 10
    ADAPTIVE_TOP1_SWITCH = 11
    ADAPTIVE_FULL_SHARED = 12
    ADAPTIVE_FULL_CAPACITY = 13
    ADAPTIVE_BANK_ROUTER = 14
    MEMORY = 15
    GATING_MEMORY = 16
    HALTING_MEMORY = 17
    GATING_HALTING_MEMORY = 18
    RECURRENT_MEMORY = 19
    RECURRENT_GATING_MEMORY = 20
    RECURRENT_HALTING_MEMORY = 21
    RECURRENT_GATING_HALTING_MEMORY = 22
    RESIDUAL = 23
    POST_NORM = 24
    RESIDUAL_POST_NORM = 25
    RESIDUAL_GATING = 26
    RESIDUAL_HALTING = 27
    RESIDUAL_MEMORY = 28
    SINGLE_MODEL_WEIGHT = 29
    DUAL_MODEL_WEIGHT = 30
    LOW_RANK_WEIGHT = 31
    HYPERNETWORK_WEIGHT = 32
    LAYERED_WEIGHTED_BANK_WEIGHT = 33
    SOFT_WEIGHTED_BANK_WEIGHT = 34
    AFFINE_TRANSFORM_BIAS = 35
    ADDITIVE_BIAS = 36
    GENERATOR_BIAS = 37
    MULTIPLICATIVE_BIAS = 38
    SIGMOID_GATED_BIAS = 39
    TANH_GATED_BIAS = 40
    WEIGHTED_BANK_BIAS = 41
    STANDARD_DIAGONAL = 42
    ANTI_DIAGONAL = 43
    COMBINED_DIAGONAL = 44
    DIAGONAL_AXIS_MASK = 45
    OUTER_PRODUCT_MASK = 46
    PER_AXIS_SCORE_MASK = 47
    TOP_SLICE_AXIS_MASK = 48
    WEIGHT_INFORMED_SCORE_MASK = 49
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 50
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 51
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 52
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 53
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 54
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 55
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 56
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 57
    DECAY_EXPONENTIAL_WEIGHT = 58
    NORM_L2_WEIGHT = 59
    DEEP_GENERATOR = 60
    FULL_STACK = 61
    DUAL_WEIGHT_GATING = 62
    DUAL_WEIGHT_HALTING = 63
    DUAL_WEIGHT_GATING_HALTING = 64
    DUAL_WEIGHT_MEMORY = 65
    DUAL_WEIGHT_GATING_MEMORY = 66
    DUAL_WEIGHT_HALTING_MEMORY = 67
    FULL_STACK_GATING = 68
    FULL_STACK_HALTING = 69
    FULL_STACK_MEMORY = 70
    FULL_STACK_GATING_HALTING = 71
    FULL_STACK_RECURRENT = 72
    BANK_WEIGHT_MASK = 73
    LOW_RANK_POST_NORM = 74
    RECURRENT_RESIDUAL = 75
    RECURRENT_POST_NORM = 76


_ADAPTIVE_OPTION_FLAGS = {
    "weight_option": "weight_option_flag",
    "bias_option": "bias_option_flag",
    "diagonal_option": "diagonal_option_flag",
    "row_mask_option": "mask_option_flag",
    "router_weight_option": "router_weight_option_flag",
    "router_bias_option": "router_bias_option_flag",
    "router_diagonal_option": "router_diagonal_option_flag",
    "router_row_mask_option": "router_mask_option_flag",
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
            "Default config: a mixture-of-experts classifier with adaptive linear "
            "input, output, expert, and router stacks."
        ),
    ),
    ExperimentPreset.GATING: _define_preset(
        description=(
            "Default config with per-layer gating enabled in the adaptive expert stack."
        ),
        stack_gate_flag=True,
    ),
    ExperimentPreset.HALTING: _define_preset(
        description=(
            "Default config with stack halting enabled in the adaptive expert stack."
        ),
        stack_halting_flag=True,
    ),
    ExperimentPreset.GATING_HALTING: _define_preset(
        description=(
            "Default config with both per-layer gating and stack halting "
            "enabled in the adaptive expert stack."
        ),
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.MEMORY: _define_preset(
        description=(
            "Default config with shared stack memory enabled in the adaptive "
            "expert stack."
        ),
        memory_flag=True,
    ),
    ExperimentPreset.GATING_MEMORY: _define_preset(
        description=(
            "Default config with both per-layer gating and shared stack "
            "memory enabled in the adaptive expert stack."
        ),
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.HALTING_MEMORY: _define_preset(
        description=(
            "Default config with both stack halting and shared stack memory "
            "enabled in the adaptive expert stack."
        ),
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default config with per-layer gating, stack halting, and shared "
            "stack memory enabled in the adaptive expert stack."
        ),
        stack_gate_flag=True,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RESIDUAL: _define_preset(
        description="Default adaptive expert config with residual skip connections.",
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.POST_NORM: _define_preset(
        description="Default adaptive expert config with post-layer normalization.",
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: _define_preset(
        description=(
            "Default adaptive expert config with residual connections and "
            "post-layer normalization."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RESIDUAL_GATING: _define_preset(
        description=(
            "Default adaptive expert config with residual connections and "
            "per-layer gating."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_gate_flag=True,
    ),
    ExperimentPreset.RESIDUAL_HALTING: _define_preset(
        description=(
            "Default adaptive expert config with residual connections and "
            "stack halting."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_halting_flag=True,
    ),
    ExperimentPreset.RESIDUAL_MEMORY: _define_preset(
        description=(
            "Default adaptive expert config with residual connections and "
            "shared stack memory."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        memory_flag=True,
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT: _define_preset(
        description="Default adaptive expert config with single-model dynamic weights.",
        weight_option=SingleModelDynamicWeightConfig,
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT: _define_preset(
        description="Default adaptive expert config with dual-model dynamic weights.",
        weight_option=DualModelDynamicWeightConfig,
    ),
    ExperimentPreset.LOW_RANK_WEIGHT: _define_preset(
        description="Default adaptive expert config with low-rank dynamic weights.",
        weight_option=LowRankDynamicWeightConfig,
    ),
    ExperimentPreset.HYPERNETWORK_WEIGHT: _define_preset(
        description="Default adaptive expert config with hypernetwork dynamic weights.",
        weight_option=HypernetworkDynamicWeightConfig,
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT: _define_preset(
        description=(
            "Default adaptive expert config with layered weighted-bank dynamic weights."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
    ),
    ExperimentPreset.SOFT_WEIGHTED_BANK_WEIGHT: _define_preset(
        description=(
            "Default adaptive expert config with soft weighted-bank dynamic weights."
        ),
        weight_option=SoftWeightedBankDynamicWeightConfig,
    ),
    ExperimentPreset.AFFINE_TRANSFORM_BIAS: _define_preset(
        description="Default adaptive expert config with affine-transform dynamic bias.",
        bias_option=AffineTransformDynamicBiasConfig,
    ),
    ExperimentPreset.ADDITIVE_BIAS: _define_preset(
        description="Default adaptive expert config with additive dynamic bias.",
        bias_option=AdditiveDynamicBiasConfig,
    ),
    ExperimentPreset.GENERATOR_BIAS: _define_preset(
        description="Default adaptive expert config with generator dynamic bias.",
        bias_option=GeneratorDynamicBiasConfig,
    ),
    ExperimentPreset.MULTIPLICATIVE_BIAS: _define_preset(
        description="Default adaptive expert config with multiplicative dynamic bias.",
        bias_option=MultiplicativeDynamicBiasConfig,
    ),
    ExperimentPreset.SIGMOID_GATED_BIAS: _define_preset(
        description="Default adaptive expert config with sigmoid-gated dynamic bias.",
        bias_option=SigmoidGatedDynamicBiasConfig,
    ),
    ExperimentPreset.TANH_GATED_BIAS: _define_preset(
        description="Default adaptive expert config with tanh-gated dynamic bias.",
        bias_option=TanhGatedDynamicBiasConfig,
    ),
    ExperimentPreset.WEIGHTED_BANK_BIAS: _define_preset(
        description="Default adaptive expert config with weighted-bank dynamic bias.",
        bias_option=WeightedBankDynamicBiasConfig,
    ),
    ExperimentPreset.STANDARD_DIAGONAL: _define_preset(
        description="Default adaptive expert config with standard dynamic diagonal.",
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.ANTI_DIAGONAL: _define_preset(
        description="Default adaptive expert config with anti dynamic diagonal.",
        diagonal_option=AntiDynamicDiagonalConfig,
    ),
    ExperimentPreset.COMBINED_DIAGONAL: _define_preset(
        description="Default adaptive expert config with combined dynamic diagonal.",
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.DIAGONAL_AXIS_MASK: _define_preset(
        description="Default adaptive expert config with diagonal-axis row masking.",
        row_mask_option=DiagonalAxisMaskConfig,
    ),
    ExperimentPreset.OUTER_PRODUCT_MASK: _define_preset(
        description="Default adaptive expert config with outer-product row masking.",
        row_mask_option=OuterProductMaskConfig,
    ),
    ExperimentPreset.PER_AXIS_SCORE_MASK: _define_preset(
        description="Default adaptive expert config with per-axis score row masking.",
        row_mask_option=PerAxisScoreMaskConfig,
    ),
    ExperimentPreset.TOP_SLICE_AXIS_MASK: _define_preset(
        description="Default adaptive expert config with top-slice axis row masking.",
        row_mask_option=TopSliceAxisMaskConfig,
    ),
    ExperimentPreset.WEIGHT_INFORMED_SCORE_MASK: _define_preset(
        description=(
            "Default adaptive expert config with weight-informed score row masking."
        ),
        row_mask_option=WeightInformedScoreAxisMaskConfig,
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
        _define_preset(
            description=(
                "Default adaptive expert config with single-model weights, "
                "additive bias, and combined diagonal."
            ),
            weight_option=SingleModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
        )
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
        _define_preset(
            description=(
                "Default adaptive expert config with dual-model weights, "
                "additive bias, and combined diagonal."
            ),
            weight_option=DualModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
        )
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
        _define_preset(
            description=(
                "Default adaptive expert config with layered weighted-bank "
                "weights, additive bias, and combined diagonal."
            ),
            weight_option=LayeredWeightedBankDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
        )
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
        _define_preset(
            description=(
                "Default adaptive expert config with low-rank weights, "
                "additive bias, and combined diagonal."
            ),
            weight_option=LowRankDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
        )
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
        _define_preset(
            description=(
                "Default adaptive expert config with single-model weights, "
                "additive bias, and standard diagonal."
            ),
            weight_option=SingleModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=StandardDynamicDiagonalConfig,
        )
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
        _define_preset(
            description=(
                "Default adaptive expert config with dual-model weights, "
                "additive bias, and standard diagonal."
            ),
            weight_option=DualModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=StandardDynamicDiagonalConfig,
        )
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
        _define_preset(
            description=(
                "Default adaptive expert config with layered weighted-bank "
                "weights, additive bias, and standard diagonal."
            ),
            weight_option=LayeredWeightedBankDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=StandardDynamicDiagonalConfig,
        )
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
        _define_preset(
            description=(
                "Default adaptive expert config with low-rank weights, "
                "additive bias, and standard diagonal."
            ),
            weight_option=LowRankDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=StandardDynamicDiagonalConfig,
        )
    ),
    ExperimentPreset.DECAY_EXPONENTIAL_WEIGHT: _define_preset(
        description=(
            "Default adaptive expert config with exponentially decayed "
            "dual-model dynamic weights."
        ),
        weight_option=DualModelDynamicWeightConfig,
        weight_decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
        weight_decay_rate=0.001,
        weight_decay_warmup_batches=500,
    ),
    ExperimentPreset.NORM_L2_WEIGHT: _define_preset(
        description=(
            "Default adaptive expert config with L2-scale normalized dynamic weights."
        ),
        weight_option=DualModelDynamicWeightConfig,
        weight_normalization_option=WeightNormalizationOptions.L2_SCALE,
    ),
    ExperimentPreset.DEEP_GENERATOR: _define_preset(
        description="Default adaptive expert config with a depth-8 weight generator.",
        weight_option=DualModelDynamicWeightConfig,
        generator_depth=8,
    ),
    ExperimentPreset.FULL_STACK: _define_preset(
        description=(
            "Default adaptive expert config with dynamic weights, bias, "
            "diagonal, and row masking."
        ),
        base=_FULL_STACK_VALUES,
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING: _define_preset(
        description=(
            "Default adaptive expert config with dual-model weights and "
            "per-layer gating."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_gate_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING: _define_preset(
        description=(
            "Default adaptive expert config with dual-model weights and stack halting."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_halting_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING_HALTING: _define_preset(
        description=(
            "Default adaptive expert config with dual-model weights, "
            "per-layer gating, and stack halting."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_MEMORY: _define_preset(
        description=(
            "Default adaptive expert config with dual-model weights and "
            "shared stack memory."
        ),
        weight_option=DualModelDynamicWeightConfig,
        memory_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING_MEMORY: _define_preset(
        description=(
            "Default adaptive expert config with dual-model weights, "
            "per-layer gating, and shared stack memory."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING_MEMORY: _define_preset(
        description=(
            "Default adaptive expert config with dual-model weights, "
            "stack halting, and shared stack memory."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.FULL_STACK_GATING: _define_preset(
        description="Default full adaptive expert config with per-layer gating.",
        base=_FULL_STACK_VALUES,
        stack_gate_flag=True,
    ),
    ExperimentPreset.FULL_STACK_HALTING: _define_preset(
        description="Default full adaptive expert config with stack halting.",
        base=_FULL_STACK_VALUES,
        stack_halting_flag=True,
    ),
    ExperimentPreset.FULL_STACK_MEMORY: _define_preset(
        description="Default full adaptive expert config with shared stack memory.",
        base=_FULL_STACK_VALUES,
        memory_flag=True,
    ),
    ExperimentPreset.FULL_STACK_GATING_HALTING: _define_preset(
        description=(
            "Default full adaptive expert config with per-layer gating and "
            "stack halting."
        ),
        base=_FULL_STACK_VALUES,
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.FULL_STACK_RECURRENT: _define_preset(
        description="Default full adaptive expert config wrapped in recurrence.",
        base=_FULL_STACK_VALUES,
        recurrent_flag=True,
    ),
    ExperimentPreset.BANK_WEIGHT_MASK: _define_preset(
        description=(
            "Default adaptive expert config with bank weights and "
            "weight-informed row masking."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
        row_mask_option=WeightInformedScoreAxisMaskConfig,
    ),
    ExperimentPreset.LOW_RANK_POST_NORM: _define_preset(
        description=(
            "Default adaptive expert config with low-rank weights and "
            "post-layer normalization."
        ),
        weight_option=LowRankDynamicWeightConfig,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RECURRENT: _define_preset(
        description=(
            "Default config wrapped in fixed-step recurrence, reusing the adaptive "
            "expert stack for each recurrent step."
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
            "Default recurrent config whose reused adaptive expert stack has "
            "shared memory enabled."
        ),
        recurrent_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: _define_preset(
        description=(
            "Default recurrent config with step-level gating and shared memory "
            "in the reused adaptive expert stack."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent config with recurrent halting and shared memory "
            "in the reused adaptive expert stack."
        ),
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent config with step-level gating, recurrent "
            "halting, and shared memory in the reused adaptive expert stack."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: _define_preset(
        description=(
            "Default recurrent adaptive expert config using a residual expert "
            "stack at each recurrent step."
        ),
        recurrent_flag=True,
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.RECURRENT_POST_NORM: _define_preset(
        description=(
            "Default recurrent adaptive expert config using a post-normalized "
            "expert stack at each recurrent step."
        ),
        recurrent_flag=True,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.ADAPTIVE_SHARED_ROUTER: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights and shared "
            "expert routing."
        ),
        weight_option=DualModelDynamicWeightConfig,
        routing_initialization_mode=RoutingInitializationMode.SHARED,
    ),
    ExperimentPreset.ADAPTIVE_AFTER_WEIGHT: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights and expert "
            "weighting after expert outputs."
        ),
        weight_option=DualModelDynamicWeightConfig,
        weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
    ),
    ExperimentPreset.ADAPTIVE_TOP1_SWITCH: _define_preset(
        description=(
            "Default adaptive config with dual-model dynamic weights, top-1 switch "
            "routing, and switch auxiliary loss enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        top_k=1,
        sampler_normalize_probabilities_flag=False,
        sampler_switch_loss_weight=0.1,
    ),
    ExperimentPreset.ADAPTIVE_FULL_SHARED: _define_preset(
        description=(
            "Default adaptive config with dynamic weights, bias, diagonal, row "
            "mask, and shared expert routing."
        ),
        base=_FULL_STACK_VALUES,
        routing_initialization_mode=RoutingInitializationMode.SHARED,
    ),
    ExperimentPreset.ADAPTIVE_FULL_CAPACITY: _define_preset(
        description=(
            "Default adaptive config with dynamic weights, bias, diagonal, row "
            "mask, top-1 capacity limiting, and dropped tokens zeroed."
        ),
        base=_FULL_STACK_VALUES,
        top_k=1,
        capacity_factor=1.0,
        dropped_token_behavior=DroppedTokenOptions.ZEROS,
        sampler_normalize_probabilities_flag=False,
    ),
    ExperimentPreset.ADAPTIVE_BANK_ROUTER: _define_preset(
        description=(
            "Default adaptive config with layered weighted-bank dynamic "
            "weights and shared expert routing."
        ),
        router_weight_option=LayeredWeightedBankDynamicWeightConfig,
        routing_initialization_mode=RoutingInitializationMode.SHARED,
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=NeuronExpertLinearAdaptiveConfigBuilder,
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
