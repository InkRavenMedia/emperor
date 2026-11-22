from typing import Unpack

import models.gpt.expert_linear_adaptive.config as config
from emperor.augmentations.adaptive_parameters import (
    CombinedDynamicDiagonalConfig,
    GeneratorDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
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
from models.gpt.expert_linear_adaptive.config_builder import (
    GptExpertLinearAdaptiveConfigBuilder,
)
from models.gpt.expert_linear_adaptive.runtime_defaults import (
    runtime_from_flat,
)
from models.gpt.expert_linear_adaptive.runtime_options import PresetValues


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
    LOW_RANK_EXPERT_WEIGHT = 22
    SINGLE_LAYER_RECURRENT_ATTENTION_RESIDUAL = 23
    HIERARCHICAL = 24


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
            "an adaptive Mixture of Attention Heads, Mixture-of-Experts "
            "feed-forward sub-stacks, learned positional embeddings, and causal "
            "attention."
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
    ExperimentPreset.TOP1_SWITCH_AUX: _define_preset(
        description="Default config with top-1 expert routing and switch auxiliary loss.",
        top_k=1,
        sampler_normalize_probabilities_flag=False,
        sampler_switch_loss_weight=0.1,
    ),
    ExperimentPreset.LOW_RANK_EXPERT_WEIGHT: _define_preset(
        description=(
            "Default config with adaptive low-rank dynamic weights inside expert "
            "feed-forward internals."
        ),
        weight_option_flag=True,
        weight_option=config.LowRankDynamicWeightConfig,
    ),
    ExperimentPreset.SINGLE_LAYER_RECURRENT_ATTENTION_RESIDUAL: _define_preset(
        description=(
            "One dropout-free 32-dimensional decoder layer over 35-token sequences, "
            "with four attention heads and a 32-dimensional feed-forward stack. "
            "Top-two routing selects from twelve adaptive experts without a capacity "
            "limit, while key/value paths remain dense. The layer is reused for up to "
            "ten recurrent steps with an outer attention residual and raw-state "
            "stick-breaking halting at 0.99. "
            "Expert weights use a depth-five single-model generator, post-outer-product "
            "L2 normalization, and delayed exponential base-weight decay; generated "
            "biases and combined diagonals are enabled without weight masking."
        ),
        hidden_dim=32,
        sequence_length=35,
        stack_num_layers=1,
        stack_dropout_probability=0.0,
        attn_num_heads=4,
        ff_stack_hidden_dim=32,
        num_experts=12,
        top_k=2,
        capacity_factor=0.0,
        expert_attention_use_kv_expert_models_flag=False,
        sampler_switch_loss_weight=0.01,
        sampler_zero_centred_loss_weight=0.001,
        recurrent_flag=True,
        recurrent_max_steps=10,
        recurrent_residual_connection_option=config.AttentionResidualConfig,
        recurrent_stack_halting_flag=True,
        recurrent_halting_option=config.StickBreakingConfig,
        recurrent_halting_threshold=0.99,
        recurrent_halting_hidden_state_mode=config.HaltingHiddenStateModeOptions.RAW,
        expert_bias_flag=True,
        weight_option_flag=True,
        weight_option=SingleModelDynamicWeightConfig,
        generator_depth=5,
        weight_decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
        weight_decay_rate=0.0001,
        weight_decay_warmup_batches=5000,
        weight_normalization_option=WeightNormalizationOptions.L2_SCALE,
        weight_normalization_position_option=WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
        bias_option_flag=True,
        bias_option=GeneratorDynamicBiasConfig,
        diagonal_option_flag=True,
        diagonal_option=CombinedDynamicDiagonalConfig,
        mask_option_flag=False,
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=GptExpertLinearAdaptiveConfigBuilder,
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
