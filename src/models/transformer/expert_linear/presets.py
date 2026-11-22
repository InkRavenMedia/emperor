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

from .config_builder import TransformerExpertLinearConfigBuilder
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
    TOP1_SWITCH_AUX = 28


_RECOMMENDED_HALTING_THRESHOLD = 0.999


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
    ExperimentPreset.TOP1_SWITCH_AUX: PresetValues(
        top_k=1,
        normalize_probabilities_flag=False,
        switch_loss_weight=0.1,
    ),
}

_PRESET_DEFINITIONS = {
    preset: PresetDefinition(
        preset_values=values,
        description=f"Expert Transformer preset: {preset.name.replace('_', ' ').lower()}.",
    )
    for preset, values in _PRESET_VALUES.items()
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=TransformerExpertLinearConfigBuilder,
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
