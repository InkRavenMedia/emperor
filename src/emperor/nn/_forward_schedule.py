"""Coordinate forward counters through bound access and shared execution scopes."""

from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum

import torch
from torch import Tensor, nn


class ForwardScheduleCounterOptions(Enum):
    """Registered counters coordinated during forward execution."""

    WARMUP_STEP = "warmup_step"
    DECAY_STEP = "decay_step"
    FORWARD_CALL_PROGRESS = "forward_call_progress"


def _validate_counter(counter: ForwardScheduleCounterOptions) -> None:
    if not isinstance(counter, ForwardScheduleCounterOptions):
        raise TypeError("counter must be a ForwardScheduleCounterOptions member")


_active_group: ContextVar = ContextVar("emperor_forward_schedule_group", default=None)
_active_run: ContextVar = ContextVar("emperor_forward_schedule_run", default=None)
_frozen: ContextVar[bool] = ContextVar("emperor_forward_schedule_frozen", default=False)


def _set_counter_value(
    module: nn.Module, counter: ForwardScheduleCounterOptions, value: Tensor | int
) -> None:
    _validate_counter(counter)
    if _frozen.get():
        return
    buffer = getattr(module, counter.value)
    value = torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device)
    group = _active_group.get()
    if group is None:
        buffer.copy_(value)
    else:
        group.values[(module, counter)] = value.detach().clone()


class _ScheduleGroups:
    def __init__(self):
        self.parent = _active_group.get()
        self.initial = {}
        self.completed = {}

    def read_initial(self, key):
        if key not in self.initial:
            module, counter = key
            value = (
                self.parent.read(key)
                if self.parent is not None
                else getattr(module, counter.value)
            )
            self.initial[key] = value.detach().clone()
        return self.initial[key]

    @contextmanager
    def group(self):
        group = _ScheduleGroup(self)
        token = _active_group.set(group)
        try:
            yield
            for key, value in group.values.items():
                self.completed[key] = torch.maximum(
                    self.completed.get(key, self.read_initial(key)), value
                )
        finally:
            _active_group.reset(token)

    def commit(self):
        for (module, counter), value in self.completed.items():
            _set_counter_value(module, counter, value)


class _ScheduleGroup:
    def __init__(self, owner):
        self.owner = owner
        self.values = {}

    def read(self, key):
        return self.values.get(key, self.owner.read_initial(key))


class ForwardSchedule:
    """Access one owner's registered counters within shared execution scopes.

    The owner retains its buffers and checkpoint names. This plain Python helper
    holds the owner, so counter reads follow buffer replacement and device moves
    without registering another neural-network child.
    """

    def __init__(self, module: nn.Module) -> None:
        self._module = module

    def value(self, counter: ForwardScheduleCounterOptions) -> Tensor:
        _validate_counter(counter)
        group = _active_group.get()
        value = (
            getattr(self._module, counter.value)
            if group is None
            else group.read((self._module, counter))
        )
        # Schedule buffers can advance before backward; autograd needs a snapshot.
        return value.detach().clone()

    def set_value(
        self, counter: ForwardScheduleCounterOptions, value: Tensor | int
    ) -> None:
        _set_counter_value(self._module, counter, value)

    def run_value(self, counter: ForwardScheduleCounterOptions) -> Tensor:
        """Read the initial value for an adaptive schedule in this recurrent run."""
        _validate_counter(counter)
        run = _active_run.get()
        if run is None:
            return self.value(counter)
        return run.read_initial((self._module, counter)).detach().clone()

    def set_run_value(
        self, counter: ForwardScheduleCounterOptions, value: Tensor | int
    ) -> None:
        """Defer an adaptive schedule update until the outer recurrent run succeeds."""
        _validate_counter(counter)
        if _frozen.get():
            return
        run = _active_run.get()
        if run is None:
            self.set_value(counter, value)
            return
        key = (self._module, counter)
        initial = run.read_initial(key)
        value = torch.as_tensor(value, dtype=initial.dtype, device=initial.device)
        run.completed[key] = (
            torch.maximum(run.completed.get(key, initial), value).detach().clone()
        )


@contextmanager
def grouped_forward_calls():
    """Replay each independent group's counters and commit their greatest progress.

    Unused schedules do not advance. A failed grouped execution discards pending
    updates; successful execution never sums progress across independent groups.
    """
    groups = _ScheduleGroups()
    yield groups.group
    groups.commit()


@contextmanager
def freeze_forward_schedules():
    """Use current schedule values without advancing them during generation."""
    token = _frozen.set(True)
    try:
        yield
    finally:
        _frozen.reset(token)


@contextmanager
def once_per_recurrent_run():
    """Share one adaptive decay step across nested recurrent invocations.

    Only run_value/set_run_value participate; depth-growth counters retain
    their separate lifecycle. Warmup and decay advance only for schedules
    actually executed, once after the outermost successful recurrent call.
    """
    if _active_run.get() is not None:
        yield
        return
    run = _ScheduleGroups()
    token = _active_run.set(run)
    try:
        yield
    finally:
        _active_run.reset(token)
    run.commit()


@contextmanager
def preserve_forward_schedule_updates():
    """Restore deferred schedule updates after a provisional recurrent branch."""
    run, group = _active_run.get(), _active_group.get()
    pending_run = None if run is None else dict(run.completed)
    pending_group = None if group is None else dict(group.values)
    try:
        yield
    finally:
        if run is not None:
            run.completed.clear()
            run.completed.update(pending_run)
        if group is not None:
            group.values.clear()
            group.values.update(pending_group)
