from __future__ import annotations

import torch
from torch import nn

from emperor.layers._composition.recurrent.config import InnerThinkingRecurrentConfig
from emperor.layers._composition.recurrent.validation import (
    InnerThinkingRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.inner_thinking.step import (
    InnerThinkingRecurrentStep,
)
from emperor.layers._composition.recurrent.variants.standard import RecurrentLayer


class InnerThinkingRecurrent(RecurrentLayer):
    VALIDATOR = InnerThinkingRecurrentValidator

    def __init__(
        self,
        cfg: InnerThinkingRecurrentConfig,
        overrides: InnerThinkingRecurrentConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.cfg: InnerThinkingRecurrentConfig
        self.thinking_step_scale = self.step.thinking_step_scale
        self.sampler_config = self.cfg.sampler_config

        self.samplers = nn.ModuleList(
            self.sampler_config.build_with_router_input_dim(self.output_dim)
            for _ in range(self.max_steps - 1)
        )
        self.thinking_step_weights = nn.Parameter(
            torch.ones(self.max_steps, self.output_dim)
        )

    def _build_step(self) -> InnerThinkingRecurrentStep:
        return InnerThinkingRecurrentStep(
            cfg=self.cfg,
            get_module=self.get_submodule,
            get_parameter=self.get_parameter,
        )
