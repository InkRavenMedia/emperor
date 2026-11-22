# ruff: noqa: E501

from typing import Unpack

import models.vit.linear.config as config
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
from models.vit.linear.config_builder import VitLinearConfigBuilder
from models.vit.linear.runtime_defaults import runtime_from_flat
from models.vit.linear.runtime_options import PresetValues


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
    RECURRENT = 17
    RECURRENT_GATING = 18
    RECURRENT_HALTING = 19
    RECURRENT_MEMORY = 20
    RECURRENT_GATING_HALTING = 21
    RECURRENT_GATING_MEMORY = 22
    RECURRENT_HALTING_MEMORY = 23
    RECURRENT_GATING_HALTING_MEMORY = 24
    RECURRENT_RESIDUAL = 25
    RECURRENT_POST_NORM = 26


def _define_preset(
    *,
    description: str,
    **values: Unpack[PresetValues],
) -> PresetDefinition:
    return PresetDefinition(preset_values=values, description=description)


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: _define_preset(
        description=(
            "Default config: a Vision Transformer classifier with linear patch "
            "embeddings, a trainable class token, learned image positions, and a "
            "pre-norm bidirectional encoder."
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
            "Default recurrent config with step-level gating, recurrent "
            "halting, and shared memory in the reused encoder stack."
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
            builder_type=VitLinearConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            runtime_factory=runtime_from_flat,
        )

    def _dataset_config(self, dataset: type) -> dict:
        dataset_config = super()._dataset_config(dataset)
        dataset_patch_size = default_patch_size_for_dataset(dataset)

        return {
            **dataset_config,
            "image_patch_size": dataset_patch_size,
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
