from typing import Unpack

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
from models.bert.linear.config_builder import (
    BertLinearConfigBuilder,
)
from models.bert.linear.runtime_defaults import runtime_from_flat
from models.bert.linear.runtime_options import PresetValues


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


def _define_preset(
    *,
    description: str,
    **values: Unpack[PresetValues],
) -> PresetDefinition:
    return PresetDefinition(preset_values=values, description=description)


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: _define_preset(
        description=(
            "Default config: a BERT-style pretraining encoder with linear "
            "attention and feed-forward sub-stacks, learned positional "
            "embeddings, and bidirectional attention."
        ),
    ),
    ExperimentPreset.PRE_NORM: _define_preset(
        description=(
            "Default config with layer normalization applied before each encoder "
            "sub-block."
        ),
        layer_norm_position=LayerNormPositionOptions.BEFORE,
    ),
    ExperimentPreset.POST_NORM: _define_preset(
        description=(
            "Default config with layer normalization applied after each encoder "
            "sub-block."
        ),
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.SINUSOIDAL: _define_preset(
        description="Default config with fixed sinusoidal positional embeddings.",
        positional_embedding_option=TextSinusoidalPositionalEmbeddingConfig,
    ),
    ExperimentPreset.CAUSAL: _define_preset(
        description=(
            "Default config with causal attention masking enabled for "
            "autoregressive modeling."
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
        description="Default config with both encoder-block gating and halting enabled.",
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.MEMORY: _define_preset(
        description="Default config with shared encoder-stack memory enabled.",
        memory_flag=True,
    ),
    ExperimentPreset.GATING_MEMORY: _define_preset(
        description="Default config with encoder-block gating and shared memory enabled.",
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.HALTING_MEMORY: _define_preset(
        description="Default config with encoder-block halting and shared memory enabled.",
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: _define_preset(
        description="Default config with encoder-block gating, halting, and shared memory.",
        stack_gate_flag=True,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT: _define_preset(
        description="Default encoder stack wrapped in fixed-step recurrence.",
        recurrent_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING: _define_preset(
        description="Default recurrent encoder with step-level gating enabled.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING: _define_preset(
        description="Default recurrent encoder with recurrent halting enabled.",
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_MEMORY: _define_preset(
        description="Default recurrent encoder whose reused stack has shared memory.",
        recurrent_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: _define_preset(
        description="Default recurrent encoder with step-level gating and halting.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: _define_preset(
        description="Default recurrent encoder with step-level gating and shared memory.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: _define_preset(
        description="Default recurrent encoder with recurrent halting and shared memory.",
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent encoder with step-level gating, recurrent halting, "
            "and shared memory."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
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
            "Default recurrent config with post-layer normalization enabled "
            "inside the reused encoder stack."
        ),
        recurrent_flag=True,
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=BertLinearConfigBuilder,
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
