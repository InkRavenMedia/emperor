"""Construction and operations for optional recurrent controllers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.layers._composition.recurrent.config import RecurrentLayerConfig
from emperor.layers._composition.recurrent.runtime.residual_schedule import (
    RecurrentResidualSchedule,
    build_recurrent_residual_schedule,
)
from emperor.layers._composition.recurrent.validation import (
    RecurrentCompositionValidator,
)
from emperor.layers._config import LayerConfig
from emperor.layers._options import LayerNormPositionOptions, NormalizationOptions
from emperor.layers._layer.pipeline import LayerNormalizationDelegate
from emperor.layers._support import LayerModuleBase

if TYPE_CHECKING:
    from emperor.halting import HaltingInterface, HaltingStateBase
    from emperor.layers._composition.recurrent.config import RecurrentCompositionConfig
    from emperor.layers._composition.residual.base import (
        ResidualConnectionAbstract,
        ResidualState,
    )
    from emperor.layers._state import LayerState
    from emperor.memory import MemoryInterface, MemoryPositionOptions
    from emperor.nn import Module


class RecurrentUtilities(LayerModuleBase):
    VALIDATOR = RecurrentCompositionValidator

    def __init__(self, cfg: RecurrentCompositionConfig) -> None:
        super().__init__()
        self.output_dim = cfg.output_dim
        self.gate_config = cfg.gate_config
        self.residual_config = cfg.residual_config
        self.halting_config = cfg.halting_config
        self.memory_config = cfg.memory_config
        self.recurrent_layer_norm_position = (
            cfg.recurrent_layer_norm_position or LayerNormPositionOptions.DISABLED
        )
        self.recurrent_normalization = (
            cfg.recurrent_normalization
            if cfg.recurrent_normalization is not None
            else NormalizationOptions.LAYER_NORM
        )
        self.residual_transition_count = (
            cfg.max_steps if isinstance(cfg, RecurrentLayerConfig) else None
        )

        self.recurrent_gate = self.__build_recurrent_gate()
        self.residual_connection = self.__build_residual_connection()
        self.halting_model = self.__build_halting_model()
        self.memory_model = self.__build_memory_model()
        self.normalization = self.__build_normalization()
        self.residual_schedule = self.__build_residual_schedule()

    def __build_recurrent_gate(self) -> Module | None:
        return self._build_from_config(
            self.gate_config,
            gate_dim=self.output_dim,
        )

    def __build_residual_connection(self) -> ResidualConnectionAbstract | None:
        residual_connection = self._build_from_config(
            self.residual_config,
            residual_dim=self.output_dim,
        )
        return self.VALIDATOR.validate_built_residual_connection(residual_connection)

    def __build_halting_model(self) -> HaltingInterface[HaltingStateBase] | None:
        halting_model = self._build_from_config(
            self.halting_config,
            input_dim=self.output_dim,
        )
        return self.VALIDATOR.validate_built_halting_model(halting_model)

    def __build_memory_model(self) -> MemoryInterface | None:
        memory_model = self._build_from_config(
            self.memory_config,
            input_dim=self.output_dim,
            output_dim=self.output_dim,
        )
        return self.VALIDATOR.validate_built_memory_model(memory_model)

    def __build_normalization(self) -> LayerNormalizationDelegate:
        normalization_config = LayerConfig(
            input_dim=self.output_dim,
            output_dim=self.output_dim,
            layer_norm_position=self.recurrent_layer_norm_position,
            normalization=self.recurrent_normalization,
        )
        return LayerNormalizationDelegate(normalization_config)

    def __build_residual_schedule(self) -> RecurrentResidualSchedule | None:
        if self.residual_transition_count is None:
            return None
        return build_recurrent_residual_schedule(
            self.residual_connection,
            self.residual_transition_count,
        )

    def normalize_before_model(self, state: LayerState) -> LayerState:
        return self.normalization.before_model(state)

    def normalize_after_model(self, state: LayerState) -> LayerState:
        return self.normalization.after_model(state)

    def normalize_after_residual(self, state: LayerState) -> LayerState:
        return self.normalization.after_residual(state)

    def initialize_residual_state(
        self,
        initial_hidden: Tensor,
    ) -> ResidualState | None:
        """Create residual history for one forward pass using the owned schedule."""
        if self.residual_schedule is None or self.residual_connection is None:
            return None
        return self.residual_schedule.new_state(
            self.residual_connection, initial_hidden
        )

    def fork_residual_state(
        self,
        residual_state: ResidualState | None,
    ) -> ResidualState | None:
        """Fork residual history for an independently evolving recurrent branch."""
        if self.residual_schedule is None:
            return residual_state
        return self.residual_schedule.fork_state(residual_state)

    def apply_memory(
        self,
        hidden: Tensor,
        position: MemoryPositionOptions,
    ) -> Tensor:
        if self.memory_model is None:
            return hidden
        if self.memory_model.memory_position_option != position:
            return hidden
        return self.memory_model(hidden)

    def apply_gate(
        self,
        candidate_hidden: Tensor,
    ) -> Tensor:
        if self.recurrent_gate is None:
            return candidate_hidden
        return self.recurrent_gate(candidate_hidden)

    def apply_residual(
        self,
        candidate_hidden: Tensor,
        previous_hidden: Tensor,
        residual_state: ResidualState | None,
        transition_index: int,
    ) -> Tensor:
        if self.residual_connection is None:
            return candidate_hidden
        if self.residual_schedule is not None:
            return self.residual_schedule.apply(
                self.residual_connection,
                transition_index,
                candidate_hidden,
                previous_hidden,
                residual_state=residual_state,
            )
        return self.residual_connection(
            candidate_hidden,
            previous_hidden,
            residual_state=residual_state,
        )

    def advance_detached_target_residual_state(
        self,
        residual_input: Tensor,
        previous_evolving_hidden: Tensor,
        target_residual_state: ResidualState | None,
        transition_index: int,
    ) -> None:
        if target_residual_state is None:
            return
        residual_connection, residual_schedule = (
            self.VALIDATOR.validate_forward_local_residual_runtime(
                self.residual_connection,
                self.residual_schedule,
            )
        )
        with torch.no_grad():
            residual_schedule.advance_state(
                residual_connection,
                transition_index,
                residual_input.detach(),
                previous_evolving_hidden.detach(),
                residual_state=target_residual_state,
            )

    def apply_halting(
        self,
        previous_halting_state: HaltingStateBase | None,
        candidate_hidden: Tensor,
        update_enabled: bool,
    ) -> tuple[HaltingStateBase | None, Tensor]:
        if self.halting_model is None or not update_enabled:
            return previous_halting_state, candidate_hidden

        updated_halting_state, output_hidden = self.halting_model.update_halting_state(
            previous_halting_state,
            candidate_hidden,
        )
        self.VALIDATOR.validate_halting_output(output_hidden, candidate_hidden)
        return updated_halting_state, output_hidden

    def finalize_halting(
        self,
        hidden: Tensor,
        loss: Tensor | None,
        halting_state: HaltingStateBase | None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.halting_model is None or halting_state is None:
            return hidden, loss

        finalized_hidden, halting_loss = (
            self.halting_model.finalize_weighted_accumulation(halting_state, hidden)
        )
        reduced_halting_loss = self._reduce_auxiliary_loss(halting_loss)
        return finalized_hidden, self._accumulate_auxiliary_loss(
            loss, reduced_halting_loss
        )

    def all_items_halted(self, halting_state: HaltingStateBase | None) -> bool:
        if halting_state is None or halting_state.halt_mask is None:
            return False
        return bool(halting_state.halt_mask.all().item())
