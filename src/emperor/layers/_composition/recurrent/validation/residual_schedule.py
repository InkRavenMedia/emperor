"""Validation of residual connection selection across recurrent transitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.layers._composition.recurrent.runtime.residual_schedule import (
        DepthwiseRecurrentResidualSchedule,
        RecurrentResidualSchedule,
    )


class RecurrentResidualScheduleValidator(ValidatorBase):
    @classmethod
    def validate(cls, schedule: RecurrentResidualSchedule) -> None:
        transition_count = schedule.transition_count
        if (
            isinstance(transition_count, bool)
            or not isinstance(transition_count, int)
            or transition_count <= 0
        ):
            raise ValueError("transition_count must be a positive integer.")

    @staticmethod
    def validate_transition_index(
        schedule: RecurrentResidualSchedule,
        transition_index: object,
    ) -> None:
        if (
            isinstance(transition_index, bool)
            or not isinstance(transition_index, int)
            or not 0 <= transition_index < schedule.transition_count
        ):
            raise IndexError(
                "transition_index must identify a configured recurrent transition."
            )

    @staticmethod
    def validate_subsequent_connections(
        schedule: DepthwiseRecurrentResidualSchedule,
    ) -> None:
        expected_connection_count = schedule.transition_count - 1
        if len(schedule.subsequent_connections) != expected_connection_count:
            raise ValueError(
                "subsequent_connections must contain one connection for every "
                "transition after transition zero."
            )
