from typing import TypeVar

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    HierarchicalReasoningModelRecurrentConfig,
    LayerStackConfig,
    RecurrentCompositionConfig,
    TinyRecursiveModelRecurrentConfig,
)

from .._submodule import TransformerSubmoduleFactory
from ._recurrent_composition import build_recurrent_composition
from ._residual import ResidualStackOptions, build_residual_config
from .runtime_options import (
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
    resolve_controller_stack_options,
)

_Model = TypeVar("_Model", bound=ConfigBase)


def configure_transformer_submodule(
    model_config: _Model,
    *,
    control_stack: LayerStackConfig,
    path_options: TransformerAttentionOptions | TransformerFeedForwardOptions,
    model_dim: int,
    residual_stack_options: ResidualStackOptions,
) -> _Model | RecurrentCompositionConfig:
    def wrap_recurrent(
        recurrent_gate_config: GateConfig | None,
        recurrent_halting_config: HaltingConfig | None,
    ) -> RecurrentCompositionConfig:
        recurrent = path_options.recurrent_controller_options
        initial_iterations = recurrent.recurrent_max_steps
        if recurrent.recurrent_composition_option is TinyRecursiveModelRecurrentConfig:
            initial_iterations = recurrent.recurrent_answer_update_count
        elif (
            recurrent.recurrent_composition_option
            is HierarchicalReasoningModelRecurrentConfig
        ):
            initial_iterations = recurrent.recurrent_high_cycles
        return build_recurrent_composition(
            option=recurrent.recurrent_composition_option,
            input_dim=model_dim,
            output_dim=model_dim,
            block_config=model_config,
            max_steps=recurrent.recurrent_max_steps,
            gradient_transition_count=None,
            initial_iterations=initial_iterations,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=recurrent.recurrent_layer_norm_position,
            recurrent_normalization=recurrent.recurrent_normalization,
            gate_config=recurrent_gate_config,
            residual_config=None,
            halting_config=recurrent_halting_config,
            memory_config=None,
            no_gradient_transition_count=(
                recurrent.recurrent_no_gradient_transition_count
            ),
            reinject_original_hidden_flag=(
                recurrent.recurrent_reinject_original_hidden_flag
            ),
            latent_updates_per_answer_update=(
                recurrent.recurrent_latent_updates_per_answer_update
            ),
            answer_update_count=recurrent.recurrent_answer_update_count,
            high_cycles=recurrent.recurrent_high_cycles,
            low_cycles=recurrent.recurrent_low_cycles,
            initialization_standard_deviation=(
                recurrent.recurrent_initialization_standard_deviation
            ),
        )

    factory = TransformerSubmoduleFactory(
        resolve_controller_stack_options, build_residual_config
    )
    return factory.configure(
        model_config,
        control_stack=control_stack,
        path_options=path_options,
        model_dim=model_dim,
        residual_stack_options=residual_stack_options,
        wrap_recurrent=wrap_recurrent,
    )
