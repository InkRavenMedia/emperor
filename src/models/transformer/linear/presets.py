from typing import Unpack

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

from .config_builder import TransformerLinearConfigBuilder
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


_RECOMMENDED_HALTING_THRESHOLD = 0.999


def _define_preset(
    *,
    description: str,
    **values: Unpack[PresetValues],
) -> PresetDefinition:
    return PresetDefinition(preset_values=values, description=description)


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: _define_preset(
        description="Canonical De-to-En Transformer baseline.",
    ),
    ExperimentPreset.PRE_NORM: _define_preset(
        description="Use pre-normalization in both encoder and decoder blocks.",
        encoder_layer_norm_position=LayerNormPositionOptions.BEFORE,
        decoder_layer_norm_position=LayerNormPositionOptions.BEFORE,
    ),
    ExperimentPreset.POST_NORM: _define_preset(
        description="Use post-normalization in both encoder and decoder blocks.",
        encoder_layer_norm_position=LayerNormPositionOptions.AFTER,
        decoder_layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.LEARNED_POSITIONAL: _define_preset(
        description="Replace the sinusoidal baseline with separate learned positions.",
        positional_embedding_option=TextLearnedPositionalEmbeddingConfig,
    ),
    ExperimentPreset.ATTENTION_BIAS: _define_preset(
        description="Enable projection and learned key/value bias in every attention path.",
        attn_bias_flag=True,
        attn_add_key_value_bias_flag=True,
    ),
    ExperimentPreset.GATING: _define_preset(
        description="Gate encoder and decoder block outputs.",
        stack_gate_flag=True,
    ),
    ExperimentPreset.HALTING: _define_preset(
        description="Enable block-level adaptive halting.",
        stack_halting_flag=True,
        halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
    ),
    ExperimentPreset.GATING_HALTING: _define_preset(
        description="Combine block gating and halting.",
        stack_gate_flag=True,
        stack_halting_flag=True,
        halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
    ),
    ExperimentPreset.MEMORY: _define_preset(
        description="Enable shared dynamic memory in both stacks.",
        memory_flag=True,
    ),
    ExperimentPreset.GATING_MEMORY: _define_preset(
        description="Combine block gating and dynamic memory.",
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.HALTING_MEMORY: _define_preset(
        description="Combine block halting and dynamic memory.",
        stack_halting_flag=True,
        halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: _define_preset(
        description="Combine block gating, halting, and memory.",
        stack_gate_flag=True,
        stack_halting_flag=True,
        halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT: _define_preset(
        description="Reuse both Transformer stacks recurrently.",
        recurrent_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING: _define_preset(
        description="Use recurrent stacks with step gates.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING: _define_preset(
        description="Use recurrent stacks with adaptive step halting.",
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        recurrent_halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
    ),
    ExperimentPreset.RECURRENT_MEMORY: _define_preset(
        description="Use recurrent stacks with shared block memory.",
        recurrent_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: _define_preset(
        description="Combine recurrence, step gating, and step halting.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
        recurrent_halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: _define_preset(
        description="Combine recurrence, step gating, and memory.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: _define_preset(
        description="Combine recurrence, step halting, and memory.",
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        recurrent_halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: _define_preset(
        description="Enable all recurrent stack controllers.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
        recurrent_halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
        memory_flag=True,
    ),
    ExperimentPreset.RESIDUAL: _define_preset(
        description="Add residual joins between full Transformer blocks.",
        stack_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: _define_preset(
        description="Combine block residuals with internal post-normalization.",
        stack_residual_connection_option=AdditiveResidualConfig,
        encoder_layer_norm_position=LayerNormPositionOptions.AFTER,
        decoder_layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.RESIDUAL_GATING: _define_preset(
        description="Combine block residuals and block gates.",
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_gate_flag=True,
    ),
    ExperimentPreset.RESIDUAL_HALTING: _define_preset(
        description="Combine block residuals and block halting.",
        stack_residual_connection_option=AdditiveResidualConfig,
        stack_halting_flag=True,
        halting_threshold=_RECOMMENDED_HALTING_THRESHOLD,
    ),
    ExperimentPreset.RESIDUAL_MEMORY: _define_preset(
        description="Combine block residuals and dynamic memory.",
        stack_residual_connection_option=AdditiveResidualConfig,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: _define_preset(
        description="Add residual joins between recurrent applications.",
        recurrent_flag=True,
        recurrent_residual_connection_option=AdditiveResidualConfig,
    ),
    ExperimentPreset.RECURRENT_POST_NORM: _define_preset(
        description="Use recurrent Transformer stacks with post-normalization.",
        recurrent_flag=True,
        encoder_layer_norm_position=LayerNormPositionOptions.AFTER,
        decoder_layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=TransformerLinearConfigBuilder,
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
