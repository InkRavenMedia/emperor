from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from model_runtime.inspection._graph_accounting import GraphModule


class _RecurrentIterationSchedule(Protocol):
    def snapshot(self) -> object: ...


class _RecurrentScheduleSnapshot(Protocol):
    maximum_transition_count: Any
    active_transition_count: Any
    gradient_transition_count: Any
    iteration_unit: Any
    initial_iterations: Any
    maximum_iterations: Any
    active_iterations: Any
    iteration_increment: Any
    forward_calls_before_iteration_increment: Any
    forward_call_progress: Any
    complete: Any
    no_gradient_transition_count: Any
    smooth_iteration_growth: Any
    settled_iterations: Any
    transitioning: Any
    transition_source_iterations: Any
    transition_target_iterations: Any
    transition_forward_index: Any
    transition_forward_count: Any
    transition_weight: Any


@dataclass(frozen=True, slots=True)
class LayerControllerDetails:
    gate_active: bool = False
    gate_option: str | None = None
    halting_model: object | None = None
    layer_norm: str | None = None


class RecurrentDetails:
    @staticmethod
    def _schedule(
        module: GraphModule,
    ) -> tuple[_RecurrentScheduleSnapshot | None, Any | None]:
        iteration_schedule = getattr(module, "recurrent_iteration_schedule", None)
        schedule_snapshot: _RecurrentScheduleSnapshot | None = None
        if iteration_schedule is not None:
            schedule = cast(_RecurrentIterationSchedule, iteration_schedule)
            schedule_snapshot = cast(_RecurrentScheduleSnapshot, schedule.snapshot())
        max_steps = (
            schedule_snapshot.maximum_transition_count
            if schedule_snapshot is not None
            else getattr(module, "max_steps", None)
        )
        return schedule_snapshot, max_steps

    @staticmethod
    def _base_details(
        module: GraphModule,
        max_steps: Any,
        controllers: LayerControllerDetails,
    ) -> dict[str, Any]:
        return {
            "maxSteps": max_steps,
            "diagnostics": bool(
                getattr(module, "supports_recurrent_diagnostics", False)
            ),
            "gate": controllers.gate_active,
            "gateOption": controllers.gate_option,
            "halting": controllers.halting_model is not None,
        }

    @staticmethod
    def _append_schedule_details(
        recurrent: dict[str, Any],
        schedule_snapshot: _RecurrentScheduleSnapshot | None,
    ) -> None:
        if schedule_snapshot is None:
            return
        recurrent["activeSteps"] = schedule_snapshot.active_transition_count
        if schedule_snapshot.gradient_transition_count is not None:
            recurrent["gradientTransitionCount"] = (
                schedule_snapshot.gradient_transition_count
            )
        recurrent["iterationSchedule"] = {
            "unit": schedule_snapshot.iteration_unit,
            "initialIterations": schedule_snapshot.initial_iterations,
            "maximumIterations": schedule_snapshot.maximum_iterations,
            "activeIterations": schedule_snapshot.active_iterations,
            "iterationIncrement": schedule_snapshot.iteration_increment,
            "forwardCallsBeforeIterationIncrement": (
                schedule_snapshot.forward_calls_before_iteration_increment
            ),
            "forwardCallProgress": schedule_snapshot.forward_call_progress,
            "complete": schedule_snapshot.complete,
            "smoothIterationGrowth": (schedule_snapshot.smooth_iteration_growth),
            "settledIterations": schedule_snapshot.settled_iterations,
            "transitioning": schedule_snapshot.transitioning,
            "transitionSourceIterations": (
                schedule_snapshot.transition_source_iterations
            ),
            "transitionTargetIterations": (
                schedule_snapshot.transition_target_iterations
            ),
            "transitionForwardIndex": (schedule_snapshot.transition_forward_index),
            "transitionForwardCount": (schedule_snapshot.transition_forward_count),
            "transitionWeight": schedule_snapshot.transition_weight,
        }

    @staticmethod
    def _append_optional_details(
        module: GraphModule,
        recurrent: dict[str, Any],
        schedule_snapshot: _RecurrentScheduleSnapshot | None,
        controllers: LayerControllerDetails,
    ) -> None:
        min_steps = getattr(controllers.halting_model, "min_steps", None)
        if min_steps is not None:
            recurrent["minSteps"] = min_steps
        if schedule_snapshot is not None:
            recurrent["noGradientTransitionCount"] = (
                schedule_snapshot.no_gradient_transition_count
            )
        if controllers.layer_norm is not None:
            recurrent["layerNorm"] = controllers.layer_norm
        answer_update_count = getattr(module, "answer_update_count", None)
        if answer_update_count is not None:
            recurrent["answerUpdateCount"] = answer_update_count
        latent_updates_per_answer_update = getattr(
            module,
            "latent_updates_per_answer_update",
            None,
        )
        if latent_updates_per_answer_update is not None:
            recurrent["latentUpdatesPerAnswerUpdate"] = latent_updates_per_answer_update
        high_cycles = getattr(module, "high_cycles", None)
        if high_cycles is not None:
            recurrent["highCycles"] = high_cycles
        low_cycles = getattr(module, "low_cycles", None)
        if low_cycles is not None:
            recurrent["lowCycles"] = low_cycles

    def append(
        self,
        module: GraphModule,
        details: dict[str, Any],
        *,
        cluster_present: bool,
        controllers: LayerControllerDetails,
    ) -> None:
        schedule_snapshot, max_steps = self._schedule(module)
        if max_steps is None or cluster_present:
            return
        recurrent = self._base_details(
            module,
            max_steps,
            controllers,
        )
        self._append_schedule_details(recurrent, schedule_snapshot)
        self._append_optional_details(
            module,
            recurrent,
            schedule_snapshot,
            controllers,
        )
        details["recurrent"] = recurrent
