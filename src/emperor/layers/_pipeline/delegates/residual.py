from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.layers._pipeline.delegates.validation import LayerResidualDelegateValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._composition.residual.base import (
        ResidualConnectionAbstract,
        ResidualState,
    )
    from emperor.layers._config import LayerConfig
    from emperor.layers._pipeline.delegates.residual_schedule import (
        RecurrentResidualSchedule,
    )
    from emperor.layers._state import LayerState


class LayerResidualDelegate(Module):
    """Apply the configured residual to a layer or a recurrent branch."""

    VALIDATOR = LayerResidualDelegateValidator

    def __init__(
        self,
        cfg: LayerConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.VALIDATOR.validate(self)
        self.config = self.cfg.residual_config
        self.output_dim: int = self.cfg.output_dim

        self.connection: ResidualConnectionAbstract | None = (
            self.__build_residual_connection()
        )

    def __build_residual_connection(self) -> ResidualConnectionAbstract | None:
        residual_connection = self._build_from_config(
            self.config,
            residual_dim=self.output_dim,
        )
        return self.VALIDATOR.validate_built_residual_connection_type(
            residual_connection
        )

    def apply_residual(
        self,
        state: LayerState,
        previous: Tensor,
    ) -> LayerState:
        connection = self.connection
        if connection is None:
            return state
        return connection.apply_to_layer_state(state, previous)

    def apply_recurrent_residual(
        self,
        state: LayerState,
        previous: Tensor,
        residual_state: ResidualState | None,
        *,
        transition_index: int,
        schedule: RecurrentResidualSchedule | None,
    ) -> LayerState:
        connection = self.connection
        if connection is None:
            return state
        if schedule is None:
            state.hidden = connection(
                state.hidden, previous, residual_state=residual_state
            )
        else:
            state.hidden = schedule.apply(
                connection,
                transition_index,
                state.hidden,
                previous,
                residual_state=residual_state,
            )
        return state

    def advance_detached_state(
        self,
        current: Tensor,
        previous: Tensor,
        residual_state: ResidualState | None,
        *,
        transition_index: int,
        schedule: RecurrentResidualSchedule | None,
    ) -> None:
        if residual_state is None:
            return
        connection, schedule = self.VALIDATOR.validate_forward_local_residual_runtime(
            self.connection, schedule
        )
        with torch.no_grad():
            schedule.advance_state(
                connection,
                transition_index,
                current.detach(),
                previous.detach(),
                residual_state=residual_state,
            )
