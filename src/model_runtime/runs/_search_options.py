"""Admission rules shared by search requests and materialized Run Plans."""

from typing import Literal

from model_runtime.runs.errors import InvalidRunRequest
from model_runtime.runs.records import SearchMode


class SearchOptionError(InvalidRunRequest):
    def __init__(self, field: Literal["mode", "random_samples"], reason: str) -> None:
        self.field = field
        super().__init__(f"{field} {reason}")


def resolve_search_options(
    mode: object,
    random_samples: object = None,
    *,
    planned: bool = False,
) -> tuple[SearchMode, int | None]:
    if not isinstance(mode, str) or mode not in ("grid", "random"):
        raise SearchOptionError("mode", "must be one of 'grid' or 'random'.")
    if mode == "grid":
        if random_samples is not None:
            raise SearchOptionError(
                "random_samples", "is only valid for random search."
            )
        return "grid", None
    if random_samples is None:
        if planned:
            raise SearchOptionError(
                "random_samples", "must contain the accepted sample count."
            )
        return "random", 10
    if isinstance(random_samples, bool) or not isinstance(random_samples, int):
        raise SearchOptionError("random_samples", "must be an integer.")
    if random_samples < 1:
        raise SearchOptionError("random_samples", "must be at least 1.")
    return "random", random_samples
