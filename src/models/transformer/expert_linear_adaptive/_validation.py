from .._expert_validation import validate_runtime as _validate_runtime
from .runtime_options import RuntimeOptions


def validate_runtime(runtime: RuntimeOptions) -> None:
    _validate_runtime(runtime)


__all__ = ["validate_runtime"]
