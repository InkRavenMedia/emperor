from __future__ import annotations

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
    RecurrentCompositionConfig,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig
from models._recurrent_variants import RecurrentVariantOptions


def build_recurrent_composition(
    *,
    option: type[RecurrentCompositionConfig],
    input_dim: int,
    output_dim: int,
    block_config: ConfigBase,
    max_steps: int,
    recurrent_layer_norm_position: LayerNormPositionOptions,
    recurrent_normalization: NormalizationOptions = NormalizationOptions.LAYER_NORM,
    gate_config: GateConfig | None,
    residual_config: ResidualConfig | None,
    halting_config: HaltingConfig | None,
    memory_config: DynamicMemoryConfig | None,
    no_gradient_transition_count: int | None,
    reinject_original_hidden_flag: bool,
    latent_updates_per_answer_update: int,
    answer_update_count: int,
    high_cycles: int,
    low_cycles: int,
    initialization_standard_deviation: float,
    gradient_transition_count: int | None = None,
    initial_iterations: int,
    iteration_increment: int = 1,
    forward_calls_before_iteration_increment: int = 1,
    smooth_iteration_growth_flag: bool = False,
) -> RecurrentCompositionConfig:
    """Build the selected recurrent config leaf from package-local options."""

    stack = RecurrentLayerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        block_config=block_config,
        max_steps=max_steps,
        gradient_transition_count=gradient_transition_count,
        initial_iterations=initial_iterations,
        iteration_increment=iteration_increment,
        forward_calls_before_iteration_increment=forward_calls_before_iteration_increment,
        smooth_iteration_growth_flag=smooth_iteration_growth_flag,
        recurrent_layer_norm_position=recurrent_layer_norm_position,
        recurrent_normalization=recurrent_normalization,
        gate_config=gate_config,
        residual_config=residual_config,
        halting_config=halting_config,
        memory_config=memory_config,
        no_gradient_transition_count=no_gradient_transition_count,
        reinject_original_hidden_flag=reinject_original_hidden_flag,
    )
    variants = RecurrentVariantOptions(
        recurrent_composition_option=option,
        recurrent_latent_updates_per_answer_update=latent_updates_per_answer_update,
        recurrent_answer_update_count=answer_update_count,
        recurrent_high_cycles=high_cycles,
        recurrent_low_cycles=low_cycles,
        recurrent_initialization_standard_deviation=initialization_standard_deviation,
    )
    return variants.build_composition(stack)


__all__ = ["build_recurrent_composition"]
