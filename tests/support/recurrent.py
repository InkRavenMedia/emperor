"""Shared recurrent test blocks and Inner Thinking configuration builders."""

from dataclasses import dataclass, replace

import torch

from emperor.config import ConfigBase, optional_field
from emperor.layers import (
    InnerThinkingRecurrentConfig,
    LayerNormPositionOptions,
    LayerState,
    NormalizationOptions,
)
from emperor.nn import Module
from emperor.sampler import RouterConfig, TokenSamplerConfig
from support.layers import _set_affine_parameters, linear_stack_config


@dataclass
class RecordingBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input dimension.")
    output_dim: int | None = optional_field("Output dimension.")

    def _registry_owner(self) -> type:
        return RecordingBlock


class RecordingBlock(Module):
    def __init__(self, cfg, overrides=None):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []
        self.grad_modes = []

    def forward(self, state):
        self.calls.append(replace(state, hidden=state.hidden.detach().clone()))
        self.grad_modes.append(torch.is_grad_enabled())
        state.hidden = state.hidden * self.scale
        state.loss = state.hidden.new_tensor(0.25) + (
            0 if state.loss is None else state.loss
        )
        return state


def inner_thinking_config(**changes):
    ratio = changes.pop("selection_ratio", 0.5)
    return replace(
        InnerThinkingRecurrentConfig(
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            recurrent_normalization=NormalizationOptions.LAYER_NORM,
            input_dim=2,
            output_dim=2,
            max_steps=3,
            initial_iterations=3,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            block_config=RecordingBlockConfig(),
            sampler_config=TokenSamplerConfig(
                selection_ratio=ratio,
                router_config=RouterConfig(
                    num_experts=1,
                    noisy_topk_flag=False,
                    model_config=linear_stack_config(2, output_dim=1, bias_flag=True),
                ),
            ),
        ),
        **changes,
    )


def set_router(model, step, direction):
    _set_affine_parameters(
        model.samplers[step].router.model, torch.tensor([[direction], [0.0]]), None
    )


@dataclass
class FailingStochasticStateBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    fail_on_transition_step: int | None = optional_field(
        "One-based mutable transition step that raises."
    )

    def _registry_owner(self) -> type:
        return FailingStochasticStateBlock


class FailingStochasticStateBlock(Module):
    def __init__(
        self,
        cfg: FailingStochasticStateBlockConfig,
        overrides: FailingStochasticStateBlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("transition_step", torch.zeros(()))

    def forward(self, state: LayerState) -> LayerState:
        self.transition_step.add_(1.0)
        state.hidden = state.hidden * self.scale + torch.rand_like(state.hidden)
        if int(self.transition_step.item()) == self.cfg.fail_on_transition_step:
            raise RuntimeError("target transition failed")
        return state
