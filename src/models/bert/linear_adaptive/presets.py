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
from emperor.datasets.text.bert_pretraining import PennTreebankBertPretraining
from emperor.embedding.absolute import (
    TextSinusoidalPositionalEmbeddingConfig,
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
from models.bert.linear_adaptive.config_builder import (
    BertLinearAdaptiveConfigBuilder,
)
from models.bert.linear_adaptive.runtime_defaults import runtime_from_flat
from models.bert.linear_adaptive.runtime_options import PresetValues


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    PRE_NORM = 2
    POST_NORM = 3
    SINUSOIDAL = 4
    CAUSAL = 5
    ATTENTION_BIAS = 6
    GATING = 7
    HALTING = 8
    GATING_HALTING = 9
    MEMORY = 10
    GATING_MEMORY = 11
    HALTING_MEMORY = 12
    GATING_HALTING_MEMORY = 13
    RECURRENT = 14
    RECURRENT_GATING = 15
    RECURRENT_HALTING = 16
    RECURRENT_MEMORY = 17
    RECURRENT_GATING_HALTING = 18
    RECURRENT_GATING_MEMORY = 19
    RECURRENT_HALTING_MEMORY = 20
    RECURRENT_GATING_HALTING_MEMORY = 21
    RESIDUAL = 22
    RESIDUAL_POST_NORM = 23
    RESIDUAL_GATING = 24
    RESIDUAL_HALTING = 25
    RESIDUAL_MEMORY = 26
    RECURRENT_RESIDUAL = 27
    RECURRENT_POST_NORM = 28
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
            "Default config: a BERT-style pretraining token encoder with "
            "adaptive-capable attention and feed-forward linear stacks, learned positional "
            "embeddings, and bidirectional attention."
        ),
    ),
    ExperimentPreset.PRE_NORM: _define_preset(
        description=(
            "Default config with layer normalization applied before each "
            "encoder sub-block."
        ),
        layer_norm_position=LayerNormPositionOptions.BEFORE,
    ),
    ExperimentPreset.POST_NORM: _define_preset(
        description=(
            "Default config with layer normalization applied after each "
            "encoder sub-block."
        ),
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.SINUSOIDAL: _define_preset(
        description="Default config with fixed sinusoidal token positional embeddings.",
        positional_embedding_option=TextSinusoidalPositionalEmbeddingConfig,
    ),
    ExperimentPreset.CAUSAL: _define_preset(
        description=(
            "Default config with causal attention masking enabled for "
            "autoregressive token modeling."
        ),
        causal_attention_mask_flag=True,
    ),
    ExperimentPreset.ATTENTION_BIAS: _define_preset(
        description=(
            "Default config with attention projection bias and key/value bias enabled."
        ),
        attn_bias_flag=True,
        attn_add_key_value_bias_flag=True,
    ),
    ExperimentPreset.GATING: _define_preset(
        description="Default config with per-encoder-block gating enabled.",
        stack_gate_flag=True,
    ),
    ExperimentPreset.HALTING: _define_preset(
        description="Default config with encoder-block stack halting enabled.",
        stack_halting_flag=True,
    ),
    ExperimentPreset.GATING_HALTING: _define_preset(
        description=(
            "Default config with both encoder-block gating and halting enabled."
        ),
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.MEMORY: _define_preset(
        description="Default config with shared encoder-stack memory enabled.",
        memory_flag=True,
    ),
    ExperimentPreset.GATING_MEMORY: _define_preset(
        description=(
            "Default config with encoder-block gating and shared memory enabled."
        ),
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.HALTING_MEMORY: _define_preset(
        description=(
            "Default config with encoder-block halting and shared memory enabled."
        ),
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default config with encoder-block gating, halting, and shared memory."
        ),
        stack_gate_flag=True,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT: _define_preset(
        description="Default token encoder stack wrapped in fixed-step recurrence.",
        recurrent_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING: _define_preset(
        description="Default recurrent token encoder with step-level gating enabled.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING: _define_preset(
        description="Default recurrent token encoder with recurrent halting enabled.",
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_MEMORY: _define_preset(
        description=(
            "Default recurrent token encoder whose reused stack has shared memory."
        ),
        recurrent_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: _define_preset(
        description=(
            "Default recurrent token encoder with step-level gating and halting."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: _define_preset(
        description=(
            "Default recurrent token encoder with step-level gating and shared memory."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent token encoder with recurrent halting and shared memory."
        ),
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent token encoder with step-level gating, recurrent "
            "halting, and shared memory."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RESIDUAL: _define_preset(
        description=(
            "Default config with residual skip connections between same-width "
            "encoder stack layers."
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
            "Default config with residual skip connections and per-layer "
            "encoder gating enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_gate_flag=True,
    ),
    ExperimentPreset.RESIDUAL_HALTING: _define_preset(
        description=(
            "Default config with residual skip connections and encoder-stack "
            "halting enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_halting_flag=True,
    ),
    ExperimentPreset.RESIDUAL_MEMORY: _define_preset(
        description=(
            "Default config with residual skip connections and shared encoder "
            "memory enabled."
        ),
        stack_residual_connection_option=AdditiveResidualConfig,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: _define_preset(
        description=(
            "Default recurrent config using a residual token encoder stack at "
            "each step."
        ),
        recurrent_flag=True,
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.RECURRENT_POST_NORM: _define_preset(
        description=(
            "Default recurrent config with post-layer normalization inside the "
            "reused token encoder stack."
        ),
        recurrent_flag=True,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT: _define_preset(
        description=(
            "BERT-compatible dynamic weights for the single-model preset "
            "family on eligible encoder attention and feed-forward linear stacks."
        ),
        weight_option=DualModelDynamicWeightConfig,
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT: _define_preset(
        description=(
            "Default config with dual-model dynamic weights on eligible BERT "
            "encoder linear stacks."
        ),
        weight_option=DualModelDynamicWeightConfig,
    ),
    ExperimentPreset.LOW_RANK_WEIGHT: _define_preset(
        description=(
            "Default config with low-rank dynamic weights on eligible BERT "
            "encoder linear stacks."
        ),
        weight_option=LowRankDynamicWeightConfig,
    ),
    ExperimentPreset.HYPERNETWORK_WEIGHT: _define_preset(
        description=(
            "Default config with hypernetwork dynamic weights on eligible BERT "
            "encoder linear stacks."
        ),
        weight_option=HypernetworkDynamicWeightConfig,
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT: _define_preset(
        description=(
            "Default config with layered weighted-bank dynamic weights on "
            "eligible BERT encoder linear stacks."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
    ),
    ExperimentPreset.SOFT_WEIGHTED_BANK_WEIGHT: _define_preset(
        description=(
            "Default config with soft weighted-bank dynamic weights on "
            "eligible BERT encoder linear stacks."
        ),
        weight_option=SoftWeightedBankDynamicWeightConfig,
    ),
    ExperimentPreset.AFFINE_TRANSFORM_BIAS: _define_preset(
        description=(
            "Default config with affine-transform dynamic bias on eligible "
            "BERT encoder linear stacks."
        ),
        bias_option=AffineTransformDynamicBiasConfig,
    ),
    ExperimentPreset.ADDITIVE_BIAS: _define_preset(
        description=(
            "Default config with additive dynamic bias on eligible BERT "
            "encoder linear stacks."
        ),
        bias_option=AdditiveDynamicBiasConfig,
    ),
    ExperimentPreset.GENERATOR_BIAS: _define_preset(
        description=(
            "Default config with generator-based dynamic bias on eligible BERT "
            "encoder linear stacks."
        ),
        bias_option=GeneratorDynamicBiasConfig,
    ),
    ExperimentPreset.MULTIPLICATIVE_BIAS: _define_preset(
        description=(
            "Default config with multiplicative dynamic bias on eligible BERT "
            "encoder linear stacks."
        ),
        bias_option=MultiplicativeDynamicBiasConfig,
    ),
    ExperimentPreset.SIGMOID_GATED_BIAS: _define_preset(
        description=(
            "Default config with sigmoid-gated dynamic bias on eligible BERT "
            "encoder linear stacks."
        ),
        bias_option=SigmoidGatedDynamicBiasConfig,
    ),
    ExperimentPreset.TANH_GATED_BIAS: _define_preset(
        description=(
            "Default config with tanh-gated dynamic bias on eligible BERT "
            "encoder linear stacks."
        ),
        bias_option=TanhGatedDynamicBiasConfig,
    ),
    ExperimentPreset.WEIGHTED_BANK_BIAS: _define_preset(
        description=(
            "Default config with weighted-bank dynamic bias on eligible BERT "
            "encoder linear stacks."
        ),
        bias_option=WeightedBankDynamicBiasConfig,
    ),
    ExperimentPreset.STANDARD_DIAGONAL: _define_preset(
        description=(
            "Default config with a standard dynamic diagonal on eligible BERT "
            "encoder linear stacks."
        ),
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.ANTI_DIAGONAL: _define_preset(
        description=(
            "Default config with an anti dynamic diagonal on eligible BERT "
            "encoder linear stacks."
        ),
        diagonal_option=AntiDynamicDiagonalConfig,
    ),
    ExperimentPreset.COMBINED_DIAGONAL: _define_preset(
        description=(
            "Default config with a combined dynamic diagonal on eligible BERT "
            "encoder linear stacks."
        ),
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.DIAGONAL_AXIS_MASK: _define_preset(
        description=(
            "Default config with diagonal-axis row masking on eligible BERT "
            "encoder linear stacks."
        ),
        row_mask_option=DiagonalAxisMaskConfig,
    ),
    ExperimentPreset.OUTER_PRODUCT_MASK: _define_preset(
        description=(
            "Default config with outer-product row masking on eligible BERT "
            "encoder linear stacks."
        ),
        row_mask_option=OuterProductMaskConfig,
    ),
    ExperimentPreset.PER_AXIS_SCORE_MASK: _define_preset(
        description=(
            "Default config with per-axis score row masking on eligible BERT "
            "encoder linear stacks."
        ),
        row_mask_option=PerAxisScoreMaskConfig,
    ),
    ExperimentPreset.TOP_SLICE_AXIS_MASK: _define_preset(
        description=(
            "Default config with top-slice axis row masking on eligible BERT "
            "encoder linear stacks."
        ),
        row_mask_option=TopSliceAxisMaskConfig,
    ),
    ExperimentPreset.WEIGHT_INFORMED_SCORE_MASK: _define_preset(
        description=(
            "Default config with weight-informed score row masking on eligible "
            "BERT encoder linear stacks."
        ),
        row_mask_option=WeightInformedScoreAxisMaskConfig,
    ),
    (
        ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL
    ): _define_preset(
        description=(
            "BERT-compatible dynamic weights, additive bias, and a combined "
            "dynamic diagonal for the single-model preset family."
        ),
        weight_option=DualModelDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    (
        ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL
    ): _define_preset(
        description=(
            "Default config with dual-model dynamic weights, additive bias, "
            "and a combined dynamic diagonal on eligible BERT encoder stacks."
        ),
        weight_option=DualModelDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    (
        ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL
    ): _define_preset(
        description=(
            "Default config with layered weighted-bank weights, additive bias, "
            "and a combined diagonal on eligible BERT encoder stacks."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _define_preset(
        description=(
            "Default config with low-rank weights, additive bias, and a "
            "combined diagonal on eligible BERT encoder stacks."
        ),
        weight_option=LowRankDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=CombinedDynamicDiagonalConfig,
    ),
    (
        ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL
    ): _define_preset(
        description=(
            "BERT-compatible dynamic weights, additive bias, and a standard "
            "dynamic diagonal for the single-model preset family."
        ),
        weight_option=DualModelDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    (
        ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL
    ): _define_preset(
        description=(
            "Default config with dual-model dynamic weights, additive bias, "
            "and a standard dynamic diagonal on eligible BERT encoder stacks."
        ),
        weight_option=DualModelDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    (
        ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL
    ): _define_preset(
        description=(
            "Default config with layered weighted-bank weights, additive bias, "
            "and a standard diagonal on eligible BERT encoder stacks."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _define_preset(
        description=(
            "Default config with low-rank weights, additive bias, and a "
            "standard diagonal on eligible BERT encoder stacks."
        ),
        weight_option=LowRankDynamicWeightConfig,
        bias_option=AdditiveDynamicBiasConfig,
        diagonal_option=StandardDynamicDiagonalConfig,
    ),
    ExperimentPreset.DECAY_EXPONENTIAL_WEIGHT: _define_preset(
        description=(
            "Default config with dual-model dynamic weights decaying "
            "exponentially toward static BERT encoder linear layers."
        ),
        weight_option=DualModelDynamicWeightConfig,
        weight_decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
        weight_decay_rate=0.001,
        weight_decay_warmup_batches=500,
    ),
    ExperimentPreset.NORM_L2_WEIGHT: _define_preset(
        description=(
            "Default config with dual-model dynamic weights and L2-scale "
            "normalization on eligible BERT encoder linear stacks."
        ),
        weight_option=DualModelDynamicWeightConfig,
        weight_normalization_option=WeightNormalizationOptions.L2_SCALE,
    ),
    ExperimentPreset.DEEP_GENERATOR: _define_preset(
        description=(
            "Default config with dual-model dynamic weights produced by a "
            "depth-8 generator network for BERT encoder layers."
        ),
        weight_option=DualModelDynamicWeightConfig,
        generator_depth=8,
    ),
    ExperimentPreset.FULL_STACK: _define_preset(
        description=(
            "Default BERT token encoder with dynamic weights, additive bias, "
            "combined diagonal, and weight-informed row masking on eligible stacks."
        ),
        base=_FULL_STACK_VALUES,
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING: _define_preset(
        description=(
            "Default BERT encoder with dual-model dynamic weights and "
            "per-layer gating enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_gate_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING: _define_preset(
        description=(
            "Default BERT encoder with dual-model dynamic weights and stack "
            "halting enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_halting_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING_HALTING: _define_preset(
        description=(
            "Default BERT encoder with dual-model dynamic weights, per-layer "
            "gating, and stack halting enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_MEMORY: _define_preset(
        description=(
            "Default BERT encoder with dual-model dynamic weights and shared "
            "stack memory enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        memory_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING_MEMORY: _define_preset(
        description=(
            "Default BERT encoder with dual-model weights, per-layer gating, "
            "and shared memory enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING_MEMORY: _define_preset(
        description=(
            "Default BERT encoder with dual-model weights, stack halting, and "
            "shared memory enabled."
        ),
        weight_option=DualModelDynamicWeightConfig,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.FULL_STACK_GATING: _define_preset(
        description=(
            "Full adaptive BERT encoder parameter controls with per-layer "
            "gating enabled."
        ),
        base=_FULL_STACK_VALUES,
        stack_gate_flag=True,
    ),
    ExperimentPreset.FULL_STACK_HALTING: _define_preset(
        description=(
            "Full adaptive BERT encoder parameter controls with stack halting enabled."
        ),
        base=_FULL_STACK_VALUES,
        stack_halting_flag=True,
    ),
    ExperimentPreset.FULL_STACK_MEMORY: _define_preset(
        description=(
            "Full adaptive BERT encoder parameter controls with shared stack "
            "memory enabled."
        ),
        base=_FULL_STACK_VALUES,
        memory_flag=True,
    ),
    ExperimentPreset.FULL_STACK_GATING_HALTING: _define_preset(
        description=(
            "Full adaptive BERT encoder controls with per-layer gating and "
            "stack halting enabled."
        ),
        base=_FULL_STACK_VALUES,
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.FULL_STACK_RECURRENT: _define_preset(
        description=(
            "Full adaptive BERT encoder parameter controls wrapped in "
            "fixed-step recurrence."
        ),
        base=_FULL_STACK_VALUES,
        recurrent_flag=True,
    ),
    ExperimentPreset.BANK_WEIGHT_MASK: _define_preset(
        description=(
            "Default BERT encoder with layered weighted-bank dynamic weights "
            "and weight-informed row masking."
        ),
        weight_option=LayeredWeightedBankDynamicWeightConfig,
        row_mask_option=WeightInformedScoreAxisMaskConfig,
    ),
    ExperimentPreset.LOW_RANK_POST_NORM: _define_preset(
        description=(
            "Default BERT encoder with low-rank dynamic weights and post-layer "
            "normalization enabled."
        ),
        weight_option=LowRankDynamicWeightConfig,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=BertLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            default_dataset=PennTreebankBertPretraining,
            runtime_factory=runtime_from_flat,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            **super()._dataset_config(dataset),
            "sequence_length": dataset.sequence_length,
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
