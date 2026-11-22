from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    AntiDynamicDiagonalConfig,
    AxisMaskConfig,
    CombinedDynamicDiagonalConfig,
    DiagonalAxisMaskConfig,
    DualModelDynamicWeightConfig,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
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
    WeightedBankDynamicBiasConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.config import BaseOptions
from emperor.datasets.text.translation import Multi30kDeEn
from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
from emperor.layers import (
    AdditiveResidualConfig,
    LayerNormPositionOptions,
)
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase

from .config_builder import TransformerExpertLinearAdaptiveConfigBuilder
from .runtime_defaults import runtime_from_flat
from .runtime_options import PresetValues


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    PRE_NORM = 2
    POST_NORM = 3
    LEARNED_POSITIONAL = 4
    ATTENTION_BIAS = 5
    GATING = 6
    HALTING = 7
    GATING_HALTING = 8
    MEMORY = 9
    GATING_MEMORY = 10
    HALTING_MEMORY = 11
    GATING_HALTING_MEMORY = 12
    RECURRENT = 13
    RECURRENT_GATING = 14
    RECURRENT_HALTING = 15
    RECURRENT_MEMORY = 16
    RECURRENT_GATING_HALTING = 17
    RECURRENT_GATING_MEMORY = 18
    RECURRENT_HALTING_MEMORY = 19
    RECURRENT_GATING_HALTING_MEMORY = 20
    RESIDUAL = 21
    RESIDUAL_POST_NORM = 22
    RESIDUAL_GATING = 23
    RESIDUAL_HALTING = 24
    RESIDUAL_MEMORY = 25
    RECURRENT_RESIDUAL = 26
    RECURRENT_POST_NORM = 27
    SINGLE_MODEL_WEIGHT = 28
    DUAL_MODEL_WEIGHT = 29
    LOW_RANK_WEIGHT = 30
    HYPERNETWORK_WEIGHT = 31
    LAYERED_WEIGHTED_BANK_WEIGHT = 32
    SOFT_WEIGHTED_BANK_WEIGHT = 33
    AFFINE_TRANSFORM_BIAS = 34
    ADDITIVE_BIAS = 35
    GENERATOR_BIAS = 36
    MULTIPLICATIVE_BIAS = 37
    SIGMOID_GATED_BIAS = 38
    TANH_GATED_BIAS = 39
    WEIGHTED_BANK_BIAS = 40
    STANDARD_DIAGONAL = 41
    ANTI_DIAGONAL = 42
    COMBINED_DIAGONAL = 43
    DIAGONAL_AXIS_MASK = 44
    OUTER_PRODUCT_MASK = 45
    PER_AXIS_SCORE_MASK = 46
    TOP_SLICE_AXIS_MASK = 47
    WEIGHT_INFORMED_SCORE_MASK = 48
    TOP1_SWITCH_AUX = 49
    LOW_RANK_EXPERT_WEIGHT = 50


_RECOMMENDED_HALTING_THRESHOLD = 0.999


def _bias_for_all_roles(bias: type[DynamicBiasConfig]) -> PresetValues:
    return PresetValues(
        attention_projection_adaptive_bias_option=bias,
        attention_expert_adaptive_bias_option=bias,
        router_adaptive_bias_option=bias,
        feed_forward_adaptive_bias_option=bias,
    )


def _diagonal_for_all_roles(diagonal: type[DynamicDiagonalConfig]) -> PresetValues:
    return PresetValues(
        attention_projection_adaptive_diagonal_option=diagonal,
        attention_expert_adaptive_diagonal_option=diagonal,
        router_adaptive_diagonal_option=diagonal,
        feed_forward_adaptive_diagonal_option=diagonal,
    )


def _row_mask_for_all_roles(row_mask: type[AxisMaskConfig]) -> PresetValues:
    return PresetValues(
        attention_projection_adaptive_row_mask_option=row_mask,
        attention_expert_adaptive_row_mask_option=row_mask,
        router_adaptive_row_mask_option=row_mask,
        feed_forward_adaptive_row_mask_option=row_mask,
    )


_PRESET_VALUES: dict[ExperimentPreset, PresetValues] = {
    ExperimentPreset.BASELINE: PresetValues(),
    ExperimentPreset.PRE_NORM: PresetValues(
        encoder_layer_norm_position=LayerNormPositionOptions.BEFORE,
        decoder_layer_norm_position=LayerNormPositionOptions.BEFORE,
    ),
    ExperimentPreset.POST_NORM: PresetValues(
        encoder_layer_norm_position=LayerNormPositionOptions.AFTER,
        decoder_layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.LEARNED_POSITIONAL: PresetValues(
        positional_embedding_option=TextLearnedPositionalEmbeddingConfig,
    ),
    ExperimentPreset.ATTENTION_BIAS: PresetValues(
        attn_bias_flag=True,
        attn_add_key_value_bias_flag=True,
    ),
    ExperimentPreset.GATING: PresetValues(
        stack_gate_flag=True,
    ),
    ExperimentPreset.HALTING: PresetValues(
        stack_halting_flag=True,
        halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
    ),
    ExperimentPreset.GATING_HALTING: PresetValues(
        stack_gate_flag=True,
        stack_halting_flag=True,
        halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
    ),
    ExperimentPreset.MEMORY: PresetValues(
        memory_flag=True,
    ),
    ExperimentPreset.GATING_MEMORY: PresetValues(
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.HALTING_MEMORY: PresetValues(
        stack_halting_flag=True,
        halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetValues(
        stack_gate_flag=True,
        stack_halting_flag=True,
        halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT: PresetValues(
        recurrent_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING: PresetValues(
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING: PresetValues(
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        recurrent_halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
    ),
    ExperimentPreset.RECURRENT_MEMORY: PresetValues(
        recurrent_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: PresetValues(
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
        recurrent_halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetValues(
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetValues(
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        recurrent_halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: PresetValues(
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
        recurrent_halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
        memory_flag=True,
    ),
    ExperimentPreset.RESIDUAL: PresetValues(
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: PresetValues(
        stack_residual_connection_option=AdditiveResidualConfig,
        encoder_layer_norm_position=LayerNormPositionOptions.AFTER,
        decoder_layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RESIDUAL_GATING: PresetValues(
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_gate_flag=True,
    ),
    ExperimentPreset.RESIDUAL_HALTING: PresetValues(
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_halting_flag=True,
        halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
    ),
    ExperimentPreset.RESIDUAL_MEMORY: PresetValues(
        stack_residual_connection_option=AdditiveResidualConfig,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: PresetValues(
        recurrent_flag=True,
        recurrent_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.RECURRENT_POST_NORM: PresetValues(
        recurrent_flag=True,
        encoder_layer_norm_position=LayerNormPositionOptions.AFTER,
        decoder_layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT: PresetValues(
        attention_projection_adaptive_weight_option=SingleModelDynamicWeightConfig,
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT: PresetValues(
        attention_projection_adaptive_weight_option=DualModelDynamicWeightConfig,
    ),
    ExperimentPreset.LOW_RANK_WEIGHT: PresetValues(
        attention_projection_adaptive_weight_option=LowRankDynamicWeightConfig,
    ),
    ExperimentPreset.HYPERNETWORK_WEIGHT: PresetValues(
        attention_projection_adaptive_weight_option=HypernetworkDynamicWeightConfig,
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT: PresetValues(
        attention_projection_adaptive_weight_option=LayeredWeightedBankDynamicWeightConfig,
    ),
    ExperimentPreset.SOFT_WEIGHTED_BANK_WEIGHT: PresetValues(
        attention_projection_adaptive_weight_option=SoftWeightedBankDynamicWeightConfig,
    ),
    ExperimentPreset.AFFINE_TRANSFORM_BIAS: _bias_for_all_roles(
        AffineTransformDynamicBiasConfig
    ),
    ExperimentPreset.ADDITIVE_BIAS: _bias_for_all_roles(AdditiveDynamicBiasConfig),
    ExperimentPreset.GENERATOR_BIAS: _bias_for_all_roles(GeneratorDynamicBiasConfig),
    ExperimentPreset.MULTIPLICATIVE_BIAS: _bias_for_all_roles(
        MultiplicativeDynamicBiasConfig
    ),
    ExperimentPreset.SIGMOID_GATED_BIAS: _bias_for_all_roles(
        SigmoidGatedDynamicBiasConfig
    ),
    ExperimentPreset.TANH_GATED_BIAS: _bias_for_all_roles(TanhGatedDynamicBiasConfig),
    ExperimentPreset.WEIGHTED_BANK_BIAS: _bias_for_all_roles(
        WeightedBankDynamicBiasConfig
    ),
    ExperimentPreset.STANDARD_DIAGONAL: _diagonal_for_all_roles(
        StandardDynamicDiagonalConfig
    ),
    ExperimentPreset.ANTI_DIAGONAL: _diagonal_for_all_roles(AntiDynamicDiagonalConfig),
    ExperimentPreset.COMBINED_DIAGONAL: _diagonal_for_all_roles(
        CombinedDynamicDiagonalConfig
    ),
    ExperimentPreset.DIAGONAL_AXIS_MASK: _row_mask_for_all_roles(
        DiagonalAxisMaskConfig
    ),
    ExperimentPreset.OUTER_PRODUCT_MASK: _row_mask_for_all_roles(
        OuterProductMaskConfig
    ),
    ExperimentPreset.PER_AXIS_SCORE_MASK: _row_mask_for_all_roles(
        PerAxisScoreMaskConfig
    ),
    ExperimentPreset.TOP_SLICE_AXIS_MASK: _row_mask_for_all_roles(
        TopSliceAxisMaskConfig
    ),
    ExperimentPreset.WEIGHT_INFORMED_SCORE_MASK: _row_mask_for_all_roles(
        WeightInformedScoreAxisMaskConfig
    ),
    ExperimentPreset.TOP1_SWITCH_AUX: PresetValues(
        top_k=1,
        normalize_probabilities_flag=False,
        switch_loss_weight=0.1,
    ),
    ExperimentPreset.LOW_RANK_EXPERT_WEIGHT: PresetValues(
        attention_expert_adaptive_weight_option=LowRankDynamicWeightConfig,
        feed_forward_adaptive_weight_option=LowRankDynamicWeightConfig,
    ),
}

_PRESET_DEFINITIONS = {
    preset: PresetDefinition(
        preset_values=values,
        description=f"Adaptive expert Transformer preset: {preset.name.replace('_', ' ').lower()}.",
    )
    for preset, values in _PRESET_VALUES.items()
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=TransformerExpertLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            default_dataset=Multi30kDeEn,
            runtime_factory=runtime_from_flat,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            "vocab_size": dataset.vocab_size,
            "source_sequence_length": dataset.source_sequence_length,
            "target_sequence_length": dataset.target_sequence_length,
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
