from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from torch import Tensor, nn

from emperor.layers._composition.recurrent.state import new_recurrent_initial_buffer
from emperor.layers._state import LayerState
from emperor.layers._support import LayerModuleBase

if TYPE_CHECKING:
    from emperor.layers._composition.recurrent.config import RecurrentCompositionConfig


class RecurrentCompositionAbstract(LayerModuleBase, ABC):
    supports_recurrent_diagnostics = False
    _STEP_OWNED_NAMES: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        cfg: RecurrentCompositionConfig,
        overrides: RecurrentCompositionConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: RecurrentCompositionConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.register_load_state_dict_pre_hook(self.__load_step_checkpoint_keys)

    def __getattr__(self, name: str):
        if name in self._STEP_OWNED_NAMES:
            return getattr(self.step, name)
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._STEP_OWNED_NAMES:
            setattr(self.step, name, value)
            return
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._STEP_OWNED_NAMES:
            delattr(self.step, name)
            return
        super().__delattr__(name)

    def __load_step_checkpoint_keys(
        self, _module: nn.Module, state_dict: dict, prefix: str, *_args: object
    ) -> None:
        for key in tuple(state_dict):
            local_key = key.removeprefix(prefix)
            if local_key.partition(".")[0] in self._STEP_OWNED_NAMES:
                step_key = f"{prefix}step.{local_key}"
                if step_key not in state_dict:
                    state_dict[step_key] = state_dict.pop(key)

    def _new_recurrent_initial_buffer(self, standard_deviation: float) -> Tensor:
        return new_recurrent_initial_buffer(self.output_dim, standard_deviation)

    def forward_sequence(self, hidden: Tensor) -> LayerState:
        if self.step.compressor is None:
            return super().forward_sequence(hidden)
        from emperor.layers._layer import Layer

        Layer.VALIDATOR.validate_sequence_input(hidden)
        return self(LayerState(hidden))

    @abstractmethod
    def forward(self, state: LayerState) -> LayerState:
        pass
