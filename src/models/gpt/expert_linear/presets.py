from typing import Unpack

from emperor.config import BaseOptions
from emperor.datasets.text.language_modeling import WikiText2
from emperor.embedding.absolute import (
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.layers import LayerNormPositionOptions
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase
from model_runtime.task_behavior import validate_language_model_dataset
from models.gpt.expert_linear import config
from models.gpt.expert_linear.config_builder import GptExpertLinearConfigBuilder
from models.gpt.expert_linear.runtime_defaults import runtime_from_flat
from models.gpt.expert_linear.runtime_options import PresetValues


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    PRE_NORM = 2
    POST_NORM = 3
    SINUSOIDAL = 4
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
    TOP1_SWITCH_AUX = 21
    HIERARCHICAL = 22


def _define_preset(
    *,
    description: str,
    **values: Unpack[PresetValues],
) -> PresetDefinition:
    return PresetDefinition(preset_values=values, description=description)


_PRESET_DEFINITIONS = {
    ExperimentPreset.HIERARCHICAL: _define_preset(
        description="HAT-style byte encoder and decoder around the causal GPT backbone; no word vocabulary. Byte widths and depths remain configurable.",
        hierarchical_language_model_flag=True,
        lm_head_weight_tying_flag=False,
    ),
    ExperimentPreset.BASELINE: _define_preset(
        description=(
            "Default config: a GPT-style causal language-modeling decoder with "
            "expert attention and feed-forward sub-stacks, learned positional "
            "embeddings, and causal attention."
        ),
    ),
    ExperimentPreset.PRE_NORM: _define_preset(
        description=(
            "Default config with layer normalization applied before each decoder "
            "sub-block."
        ),
        layer_norm_position=LayerNormPositionOptions.BEFORE,
    ),
    ExperimentPreset.POST_NORM: _define_preset(
        description=(
            "Default config with layer normalization applied after each decoder "
            "sub-block."
        ),
        layer_norm_position=LayerNormPositionOptions.AFTER,
    ),
    ExperimentPreset.SINUSOIDAL: _define_preset(
        description="Default config with fixed sinusoidal positional embeddings.",
        positional_embedding_option=TextSinusoidalPositionalEmbeddingConfig,
    ),
    ExperimentPreset.ATTENTION_BIAS: _define_preset(
        description=(
            "Default config with attention projection bias and key/value bias enabled."
        ),
        attn_bias_flag=True,
        attn_add_key_value_bias_flag=True,
    ),
    ExperimentPreset.GATING: _define_preset(
        description="Default config with per-decoder-block gating enabled.",
        stack_gate_flag=True,
    ),
    ExperimentPreset.HALTING: _define_preset(
        description="Default config with decoder-block stack halting enabled.",
        stack_halting_flag=True,
    ),
    ExperimentPreset.GATING_HALTING: _define_preset(
        description="Default config with both decoder-block gating and halting enabled.",
        stack_gate_flag=True,
        stack_halting_flag=True,
    ),
    ExperimentPreset.MEMORY: _define_preset(
        description="Default config with shared decoder-stack memory enabled.",
        memory_flag=True,
    ),
    ExperimentPreset.GATING_MEMORY: _define_preset(
        description="Default config with decoder-block gating and shared memory enabled.",
        stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.HALTING_MEMORY: _define_preset(
        description="Default config with decoder-block halting and shared memory enabled.",
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: _define_preset(
        description="Default config with decoder-block gating, halting, and shared memory.",
        stack_gate_flag=True,
        stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT: _define_preset(
        description="Default decoder stack wrapped in fixed-step recurrence.",
        recurrent_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING: _define_preset(
        description="Default recurrent decoder with step-level gating enabled.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING: _define_preset(
        description="Default recurrent decoder with recurrent halting enabled.",
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_MEMORY: _define_preset(
        description="Default recurrent decoder whose reused stack has shared memory.",
        recurrent_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: _define_preset(
        description="Default recurrent decoder with step-level gating and halting.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: _define_preset(
        description="Default recurrent decoder with step-level gating and shared memory.",
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: _define_preset(
        description="Default recurrent decoder with recurrent halting and shared memory.",
        recurrent_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: _define_preset(
        description=(
            "Default recurrent decoder with step-level gating, recurrent halting, "
            "and shared memory."
        ),
        recurrent_flag=True,
        recurrent_stack_gate_flag=True,
        recurrent_stack_halting_flag=True,
        memory_flag=True,
    ),
}

_PRESET_DEFINITIONS[ExperimentPreset.TOP1_SWITCH_AUX] = _define_preset(
    description="Default config with top-1 expert routing and switch auxiliary loss.",
    top_k=1,
    sampler_normalize_probabilities_flag=False,
    sampler_switch_loss_weight=0.1,
)


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=GptExpertLinearConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            default_dataset=WikiText2,
            runtime_factory=runtime_from_flat,
        )
        self.dataset_validator = self._validate_dataset_selection

    def _validate_dataset_selection(self, preset, dataset, overrides):
        preset_values = self.overrides_for_preset(preset)
        hierarchical = overrides.get(
            "hierarchical_language_model_flag",
            preset_values.get(
                "hierarchical_language_model_flag",
                config.HIERARCHICAL_LANGUAGE_MODEL_FLAG,
            ),
        )
        validate_language_model_dataset(dataset, hierarchical=hierarchical)

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
