# ruff: noqa: E501

from typing import Unpack

import models.vit.linear_adaptive.config as config
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
from emperor.embedding.absolute import (
    ImageSinusoidalPositionalEmbeddingConfig,
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
from models.vit.linear_adaptive.config_builder import VitLinearAdaptiveConfigBuilder
from models.vit.linear_adaptive.runtime_defaults import runtime_from_flat
from models.vit.linear_adaptive.runtime_options import PresetValues


def default_patch_size_for_dataset(dataset: type) -> int:
    image_height = dataset.default_height
    if (
        image_height >= config.IMAGE_HEIGHT
        and image_height % config.IMAGE_PATCH_SIZE == 0
    ):
        return config.IMAGE_PATCH_SIZE
    for patch_size in (4, 2, 1):
        if image_height % patch_size == 0:
            return patch_size
    return 1


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    POST_NORM = 2
    SINUSOIDAL = 3
    ATTENTION_BIAS = 4
    GATING = 5
    HALTING = 6
    MEMORY = 7
    GATING_HALTING = 8
    GATING_MEMORY = 9
    HALTING_MEMORY = 10
    GATING_HALTING_MEMORY = 11
    RESIDUAL = 12
    RESIDUAL_POST_NORM = 13
    RESIDUAL_GATING = 14
    RESIDUAL_HALTING = 15
    RESIDUAL_MEMORY = 16
    SINGLE_MODEL_WEIGHT = 17
    DUAL_MODEL_WEIGHT = 18
    LOW_RANK_WEIGHT = 19
    HYPERNETWORK_WEIGHT = 20
    LAYERED_WEIGHTED_BANK_WEIGHT = 21
    SOFT_WEIGHTED_BANK_WEIGHT = 22
    AFFINE_TRANSFORM_BIAS = 23
    ADDITIVE_BIAS = 24
    GENERATOR_BIAS = 25
    MULTIPLICATIVE_BIAS = 26
    SIGMOID_GATED_BIAS = 27
    TANH_GATED_BIAS = 28
    WEIGHTED_BANK_BIAS = 29
    STANDARD_DIAGONAL = 30
    ANTI_DIAGONAL = 31
    COMBINED_DIAGONAL = 32
    DIAGONAL_AXIS_MASK = 33
    OUTER_PRODUCT_MASK = 34
    PER_AXIS_SCORE_MASK = 35
    TOP_SLICE_AXIS_MASK = 36
    WEIGHT_INFORMED_SCORE_MASK = 37
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 38
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 39
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 40
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 41
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 42
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 43
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 44
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 45
    DECAY_EXPONENTIAL_WEIGHT = 46
    NORM_L2_WEIGHT = 47
    DEEP_GENERATOR = 48
    FULL_STACK = 49
    DUAL_WEIGHT_GATING = 50
    DUAL_WEIGHT_HALTING = 51
    DUAL_WEIGHT_GATING_HALTING = 52
    DUAL_WEIGHT_MEMORY = 53
    DUAL_WEIGHT_GATING_MEMORY = 54
    DUAL_WEIGHT_HALTING_MEMORY = 55
    FULL_STACK_GATING = 56
    FULL_STACK_HALTING = 57
    FULL_STACK_MEMORY = 58
    FULL_STACK_GATING_HALTING = 59
    FULL_STACK_RECURRENT = 60
    BANK_WEIGHT_MASK = 61
    LOW_RANK_POST_NORM = 62
    RECURRENT = 63
    RECURRENT_GATING = 64
    RECURRENT_HALTING = 65
    RECURRENT_MEMORY = 66
    RECURRENT_GATING_HALTING = 67
    RECURRENT_GATING_MEMORY = 68
    RECURRENT_HALTING_MEMORY = 69
    RECURRENT_GATING_HALTING_MEMORY = 70
    RECURRENT_RESIDUAL = 71
    RECURRENT_POST_NORM = 72


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
            "Default config: a Vision Transformer classifier with adaptive "
            "linear patch embeddings, a trainable class token, learned image positions, "
            "and a pre-norm bidirectional encoder."
        ),
    ),
    ExperimentPreset.POST_NORM: _define_preset(
        description=(
            "Default config with layer normalization after each encoder sub-block."
        ),
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.SINUSOIDAL: _define_preset(
        description="Default config with fixed sinusoidal image positional embeddings.",
        positional_embedding_option=ImageSinusoidalPositionalEmbeddingConfig,
    ),
    ExperimentPreset.ATTENTION_BIAS: _define_preset(
        description=(
            "Default config with attention projection bias and key/value bias enabled."
        ),
        attn_bias_flag=True,
        attn_add_key_value_bias_flag=True,
    ),
    ExperimentPreset.GATING: _define_preset(
        description="Default config with per-layer gating enabled on encoder blocks.",
        stack_gate_flag=True,
    ),
    ExperimentPreset.HALTING: _define_preset(
        description=(
            "Default config with encoder stack halting enabled, so examples can "
            "stop early as they move through the encoder."
        ),
        stack_halting_flag=True,
    ),
    ExperimentPreset.MEMORY: _define_preset(
        description="Default config with shared encoder stack memory enabled.",
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING: _define_preset(
        description=(
            "Default config with both per-layer gating and encoder stack "
            "halting enabled."
        ),
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.GATING_MEMORY: _define_preset(
        description=(
            "Default config with both per-layer gating and shared encoder stack "
            "memory enabled."
        ),
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.HALTING_MEMORY: _define_preset(
        description=(
            "Default config with both encoder stack halting and shared memory enabled."
        ),
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default config with per-layer gating, encoder stack halting, and "
            "shared memory enabled."
        ),
        stack_gate_flag=True,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RESIDUAL: _define_preset(
        description=(
            "Default config with residual skip connections enabled between "
            "same-width encoder stack layers."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
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
            "Default config with residual skip connections and encoder stack "
            "halting enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_halting_flag=True,
    ),
    ExperimentPreset.RESIDUAL_MEMORY: _define_preset(
        description=(
            "Default config with residual skip connections and shared encoder "
            "stack memory enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        memory_flag=True,
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT: _define_preset(
        description=(
            "Default config with VIT-compatible dynamic weights for the "
            "single-model preset family on eligible encoder attention and feed-forward "
            "linear stacks."
        ),
        weight_option=DualModelDynamicWeightConfig,
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT: _define_preset(
        description=(
            "Default config with dual-model dynamic weights on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        weight_option=DualModelDynamicWeightConfig,
    ),
    ExperimentPreset.LOW_RANK_WEIGHT: _define_preset(
        description=(
            "Default config with adaptive low-rank dynamic weights on eligible "
            "encoder attention and feed-forward linear stacks."
        ),
        weight_option=LowRankDynamicWeightConfig,
    ),
    ExperimentPreset.HYPERNETWORK_WEIGHT: _define_preset(
        description=(
            "Default config with hypernetwork dynamic weights on eligible "
            "encoder attention and feed-forward linear stacks."
        ),
        weight_option=HypernetworkDynamicWeightConfig,
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT: _define_preset(
        description=(
            "Default config with layered weighted-bank dynamic weights on "
            "eligible encoder attention and feed-forward linear stacks."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
    ),
    ExperimentPreset.SOFT_WEIGHTED_BANK_WEIGHT: _define_preset(
        description=(
            "Default config with soft weighted-bank dynamic weights on eligible "
            "encoder attention and feed-forward linear stacks."
        ),
        weight_option=SoftWeightedBankDynamicWeightConfig,
    ),
    ExperimentPreset.AFFINE_TRANSFORM_BIAS: _define_preset(
        description=(
            "Default config with affine-transform dynamic bias on eligible "
            "encoder attention and feed-forward linear stacks."
        ),
        bias_option=AffineTransformDynamicBiasConfig,
    ),
    ExperimentPreset.ADDITIVE_BIAS: _define_preset(
        description=(
            "Default config with additive dynamic bias on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        bias_option=AdditiveDynamicBiasConfig,
    ),
    ExperimentPreset.GENERATOR_BIAS: _define_preset(
        description=(
            "Default config with generator-based dynamic bias on eligible "
            "encoder attention and feed-forward linear stacks."
        ),
        bias_option=GeneratorDynamicBiasConfig,
    ),
    ExperimentPreset.MULTIPLICATIVE_BIAS: _define_preset(
        description=(
            "Default config with multiplicative dynamic bias on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        bias_option=MultiplicativeDynamicBiasConfig,
    ),
    ExperimentPreset.SIGMOID_GATED_BIAS: _define_preset(
        description=(
            "Default config with sigmoid-gated dynamic bias on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        bias_option=SigmoidGatedDynamicBiasConfig,
    ),
    ExperimentPreset.TANH_GATED_BIAS: _define_preset(
        description=(
            "Default config with tanh-gated dynamic bias on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        bias_option=TanhGatedDynamicBiasConfig,
    ),
    ExperimentPreset.WEIGHTED_BANK_BIAS: _define_preset(
        description=(
            "Default config with weighted-bank dynamic bias on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        bias_option=WeightedBankDynamicBiasConfig,
    ),
    ExperimentPreset.STANDARD_DIAGONAL: _define_preset(
        description=(
            "Default config with standard dynamic diagonal on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.ANTI_DIAGONAL: _define_preset(
        description=(
            "Default config with anti dynamic diagonal on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        diagonal_option=AntiDynamicDiagonalConfig,
    ),
    ExperimentPreset.COMBINED_DIAGONAL: _define_preset(
        description=(
            "Default config with combined dynamic diagonal on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.DIAGONAL_AXIS_MASK: _define_preset(
        description=(
            "Default config with diagonal-axis row masking on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        row_mask_option=DiagonalAxisMaskConfig,
    ),
    ExperimentPreset.OUTER_PRODUCT_MASK: _define_preset(
        description=(
            "Default config with outer-product row masking on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        row_mask_option=OuterProductMaskConfig,
    ),
    ExperimentPreset.PER_AXIS_SCORE_MASK: _define_preset(
        description=(
            "Default config with per-axis score row masking on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        row_mask_option=PerAxisScoreMaskConfig,
    ),
    ExperimentPreset.TOP_SLICE_AXIS_MASK: _define_preset(
        description=(
            "Default config with top-slice axis row masking on eligible encoder "
            "attention and feed-forward linear stacks."
        ),
        row_mask_option=TopSliceAxisMaskConfig,
    ),
    ExperimentPreset.WEIGHT_INFORMED_SCORE_MASK: _define_preset(
        description=(
            "Default config with weight-informed score axis row masking on "
            "eligible encoder attention and feed-forward linear stacks."
        ),
        row_mask_option=WeightInformedScoreAxisMaskConfig,
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _define_preset(
        description=(
            "Default config with VIT-compatible dynamic weights, additive bias, "
            "and combined dynamic diagonal for the single-model preset family on eligible "
            "encoder linear stacks."
        ),
        weight_option=DualModelDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _define_preset(
        description=(
            "Default config with dual-model dynamic weights, additive bias, and "
            "combined dynamic diagonal on eligible encoder linear stacks."
        ),
        weight_option=DualModelDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _define_preset(
        description=(
            "Default config with layered weighted-bank dynamic weights, additive "
            "bias, and combined dynamic diagonal on eligible encoder linear stacks."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _define_preset(
        description=(
            "Default config with low-rank dynamic weights, additive bias, and "
            "combined dynamic diagonal on eligible encoder linear stacks."
        ),
        weight_option=LowRankDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _define_preset(
        description=(
            "Default config with VIT-compatible dynamic weights, additive bias, "
            "and standard dynamic diagonal for the single-model preset family on eligible "
            "encoder linear stacks."
        ),
        weight_option=DualModelDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _define_preset(
        description=(
            "Default config with dual-model dynamic weights, additive bias, and "
            "standard dynamic diagonal on eligible encoder linear stacks."
        ),
        weight_option=DualModelDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _define_preset(
        description=(
            "Default config with layered weighted-bank dynamic weights, additive "
            "bias, and standard dynamic diagonal on eligible encoder linear stacks."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _define_preset(
        description=(
            "Default config with low-rank dynamic weights, additive bias, and "
            "standard dynamic diagonal on eligible encoder linear stacks."
        ),
        weight_option=LowRankDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.DECAY_EXPONENTIAL_WEIGHT: _define_preset(
        description=(
            "Default config with dual-model dynamic weights decaying "
            "exponentially toward static encoder linear layers."
        ),
        weight_option=DualModelDynamicWeightConfig,
        weight_decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
        weight_decay_rate=0.001,
        weight_decay_warmup_batches=500,
    ),
    ExperimentPreset.NORM_L2_WEIGHT: _define_preset(
        description=(
            "Default config with dual-model dynamic weights and L2-scale weight "
            "normalization on eligible encoder linear stacks."
        ),
        weight_option=DualModelDynamicWeightConfig,
        weight_normalization_option=WeightNormalizationOptions.L2_SCALE,
    ),
    ExperimentPreset.DEEP_GENERATOR: _define_preset(
        description=(
            "Default config with dual-model dynamic weights produced by a "
            "depth-8 generator network."
        ),
        weight_option=DualModelDynamicWeightConfig,
        generator_depth=8,
    ),
    ExperimentPreset.FULL_STACK: _define_preset(
        description=(
            "Default config with dual-model dynamic weights, additive bias, "
            "combined dynamic diagonal, and weight-informed row masking on eligible "
            "encoder linear stacks."
        ),
        base=_FULL_STACK_VALUES,
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING: _define_preset(
        description=(
            "Default config with dual-model dynamic weights and per-layer "
            "encoder gating enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_gate_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING: _define_preset(
        description=(
            "Default config with dual-model dynamic weights and encoder stack "
            "halting enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_halting_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING_HALTING: _define_preset(
        description=(
            "Default config with dual-model dynamic weights, per-layer encoder "
            "gating, and encoder stack halting enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_MEMORY: _define_preset(
        description=(
            "Default config with dual-model dynamic weights and shared encoder "
            "stack memory enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        memory_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING_MEMORY: _define_preset(
        description=(
            "Default config with dual-model dynamic weights, per-layer encoder "
            "gating, and shared encoder stack memory enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING_MEMORY: _define_preset(
        description=(
            "Default config with dual-model dynamic weights, encoder stack "
            "halting, and shared memory enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.FULL_STACK_GATING: _define_preset(
        description=(
            "Default config with full adaptive parameter controls and per-layer "
            "encoder gating enabled."
        ),
        base=_FULL_STACK_VALUES,
        stack_gate_flag=True,
    ),
    ExperimentPreset.FULL_STACK_HALTING: _define_preset(
        description=(
            "Default config with full adaptive parameter controls and encoder "
            "stack halting enabled."
        ),
        base=_FULL_STACK_VALUES,
        stack_halting_flag=True,
    ),
    ExperimentPreset.FULL_STACK_MEMORY: _define_preset(
        description=(
            "Default config with full adaptive parameter controls and shared "
            "encoder stack memory enabled."
        ),
        base=_FULL_STACK_VALUES,
        memory_flag=True,
    ),
    ExperimentPreset.FULL_STACK_GATING_HALTING: _define_preset(
        description=(
            "Default config with full adaptive parameter controls, per-layer "
            "encoder gating, and encoder stack halting enabled."
        ),
        base=_FULL_STACK_VALUES,
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.FULL_STACK_RECURRENT: _define_preset(
        description=(
            "Default config with full adaptive parameter controls wrapped in "
            "fixed-step encoder recurrence."
        ),
        base=_FULL_STACK_VALUES,
        recurrent_flag=True,
    ),
    ExperimentPreset.BANK_WEIGHT_MASK: _define_preset(
        description=(
            "Default config with layered weighted-bank dynamic weights and "
            "weight-informed row masking on eligible encoder linear stacks."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
        row_mask_option=WeightInformedScoreAxisMaskConfig,
    ),
    ExperimentPreset.LOW_RANK_POST_NORM: _define_preset(
        description=(
            "Default config with low-rank dynamic weights and post-layer "
            "normalization enabled."
        ),
        weight_option=LowRankDynamicWeightConfig,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RECURRENT: _define_preset(
        description=(
            "Default config wrapped in fixed-step recurrence, reusing the "
            "encoder stack for each recurrent step."
        ),
        recurrent_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING: _define_preset(
        description=(
            "Default recurrent config with step-level gating enabled after each "
            "recurrent encoder update."
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
            "Default recurrent config whose reused encoder stack has shared "
            "memory enabled."
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
            "Default recurrent config with step-level gating and shared memory "
            "in the reused encoder stack."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent config with recurrent halting and shared memory "
            "in the reused encoder stack."
        ),
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent config with step-level gating, recurrent halting, "
            "and shared memory in the reused encoder stack."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: _define_preset(
        description=(
            "Default recurrent config using a residual encoder stack at each "
            "recurrent step."
        ),
        recurrent_flag=True,
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.RECURRENT_POST_NORM: _define_preset(
        description=(
            "Default recurrent config using a post-normalized encoder stack at "
            "each recurrent step."
        ),
        recurrent_flag=True,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=VitLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            runtime_factory=runtime_from_flat,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            **super()._dataset_config(dataset),
            "image_patch_size": default_patch_size_for_dataset(dataset),
            "input_channels": dataset.num_channels,
            "image_height": dataset.default_height,
            "output_dim": dataset.num_classes,
        }


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
