from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from emperor.layers._composition.recurrent.config import (
    HierarchicalReasoningModelRecurrentConfig,
    RecurrentCompositionConfig,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
)
from emperor.layers._composition.residual.variants.attention import AttentionResidual
from emperor.layers._config import LayerConfig
from emperor.layers._options import ActivationOptions
from emperor.layers._pipeline.delegates import (
    LayerHaltingDelegate,
    LayerMemoryDelegate,
    LayerNormalizationDelegate,
    LayerPostprocessingDelegate,
    LayerResidualDelegate,
)
from emperor.layers._pipeline.delegates.residual_schedule import (
    DepthwiseRecurrentResidualSchedule,
    RecurrentResidualSchedule,
    SharedRecurrentResidualSchedule,
)
from emperor.nn import Module

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from emperor.layers._composition.recurrent.state import RecurrentState
    from emperor.layers._composition.residual.base import ResidualState
    from emperor.layers._state import LayerState


class LayerProcessingPipeline(Module):
    def __init__(self, cfg: LayerConfig | RecurrentCompositionConfig) -> None:
        super().__init__()
        self.config = cfg
        self.cfg: LayerConfig = self.__resolve_layer_config(self.config)
        self.postprocessing = LayerPostprocessingDelegate(self.cfg)
        self.residual = LayerResidualDelegate(self.cfg)
        self.halting = LayerHaltingDelegate(self.cfg)
        self.memory = LayerMemoryDelegate(self.cfg)
        self.normalization = LayerNormalizationDelegate(self.cfg)
        self.residual_schedule = self.__build_recurrent_residual_schedule()

    def __resolve_layer_config(
        self, cfg: LayerConfig | RecurrentCompositionConfig
    ) -> LayerConfig:
        if isinstance(cfg, LayerConfig):
            return cfg
        return LayerConfig(
            input_dim=cfg.output_dim,
            output_dim=cfg.output_dim,
            activation=ActivationOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=cfg.gate_config,
            halting_config=cfg.halting_config,
            memory_config=cfg.memory_config,
            residual_config=cfg.residual_config,
            layer_norm_position=cfg.recurrent_layer_norm_position,
            normalization=cfg.recurrent_normalization,
        )

    def __build_recurrent_residual_schedule(self) -> RecurrentResidualSchedule | None:
        connection = self.residual.connection
        if connection is None:
            return None
        if isinstance(self.config, RecurrentLayerConfig):
            step_count = self.config.max_steps
        elif isinstance(self.config, TinyRecursiveModelRecurrentConfig):
            step_count = self.config.answer_update_count
        elif isinstance(self.config, HierarchicalReasoningModelRecurrentConfig):
            step_count = self.config.high_cycles
        else:
            return None
        if isinstance(connection, AttentionResidual):
            return DepthwiseRecurrentResidualSchedule.from_connection(
                connection, step_count
            )
        return SharedRecurrentResidualSchedule(step_count)

    def process_normal_layer(
        self,
        state: LayerState,
        run_model: Callable[[LayerState], LayerState],
        *,
        halting_already_checked: bool = False,
    ) -> LayerState:
        if not halting_already_checked and self.halting.should_skip(state):
            return state
        input_hidden = state.hidden
        state = self.normalization.before_model(state)
        state = self.memory.before_model(state)
        state = run_model(state)
        state = self.memory.after_model(state)
        state = self.normalization.after_model(state)
        state = self.postprocessing.process(state)
        state = self.residual.apply_residual(state, input_hidden)
        state = self.normalization.after_residual(state)
        return self.halting.apply_halting(state)

    def process_recurrent_transition(
        self,
        state: LayerState,
        run_model: Callable[[LayerState], LayerState],
        *,
        recurrent_state: RecurrentState,
        previous_hidden: Tensor | None = None,
        target_state: RecurrentState | None = None,
        residual_step_index: int | None = None,
    ) -> LayerState:
        state = self.normalization.before_model(state)
        state = self.memory.before_model(state)
        output_state = run_model(state)
        state = replace(output_state)
        state = self.memory.after_model(state)
        state = self.normalization.after_model(state)
        state = self.postprocessing.process(state)
        residual_input = state.hidden
        if residual_step_index is None:
            residual_step_index = recurrent_state.transition_index
        if previous_hidden is None:
            previous_hidden = recurrent_state.output_hidden
        aligned_previous = output_state.align_recurrent_hidden(
            previous_hidden, source_state=recurrent_state.context_state
        )
        state = self.residual.apply_recurrent_residual(
            state,
            aligned_previous,
            recurrent_state.residual_state,
            transition_index=residual_step_index,
            schedule=self.residual_schedule,
        )
        state = self.normalization.after_residual(state)
        if target_state is not None:
            self.residual.advance_detached_state(
                residual_input,
                previous_hidden,
                target_state.residual_state,
                transition_index=residual_step_index,
                schedule=self.residual_schedule,
            )
        return state

    def new_recurrent_residual_state(
        self, initial_hidden: Tensor
    ) -> ResidualState | None:
        connection = self.residual.connection
        if self.residual_schedule is None or connection is None:
            return None
        return self.residual_schedule.new_state(connection, initial_hidden)

    def fork_recurrent_residual_state(
        self, residual_state: ResidualState | None
    ) -> ResidualState | None:
        if self.residual_schedule is None:
            return residual_state
        return self.residual_schedule.fork_state(residual_state)
