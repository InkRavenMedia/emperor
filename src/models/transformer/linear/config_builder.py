from __future__ import annotations

from typing import TYPE_CHECKING

from models.transformer._experiment_builder import build_translation_model_config

from ._building import build_experiment_config
from ._validation import validate_runtime
from .runtime_defaults import DEFAULT_RUNTIME
from .runtime_options import RuntimeOptions

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class TransformerLinearConfigBuilder:
    def __init__(
        self,
        *,
        runtime: RuntimeOptions = DEFAULT_RUNTIME,
    ) -> None:
        self.runtime = runtime

    @property
    def runtime(self) -> RuntimeOptions:
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: RuntimeOptions) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")
        validate_runtime(runtime)
        self._runtime = runtime

    def build(self) -> ModelConfig:
        return build_translation_model_config(self.runtime, build_experiment_config)


__all__ = ["TransformerLinearConfigBuilder"]
