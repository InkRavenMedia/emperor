from copy import deepcopy
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from emperor.config import ConfigBase, optional_field
from emperor.layers import (
    Layer,
    LayerStackConfig,
    MirroredLayerStackConfig,
    RecurrentCompositionConfig,
)
from emperor.nn import Module
from emperor.transformer._validation import (
    FeedForwardValidator,
)

if TYPE_CHECKING:
    from emperor.experts import MixtureOfExpertsModelConfig






@dataclass
class FeedForwardConfig(ConfigBase):
    input_dim: int | None = optional_field("Feed-forward input feature dimension.")
    output_dim: int | None = optional_field("Feed-forward output feature dimension.")
    stack_config: "LayerStackConfig | MixtureOfExpertsModelConfig | RecurrentCompositionConfig | None" = optional_field(  # noqa: E501
        "Either a LayerStackConfig (plain feed-forward stack), a "
        "MixtureOfExpertsModelConfig (mixture-of-experts model that manages its "
        "own routing state), or a concrete recurrent composition wrapping either "
        "form. "
        "Depth lives on the inner config."
    )

    def _registry_owner(self) -> type:
        return FeedForward


class FeedForward(Module):
    VALIDATOR = FeedForwardValidator

    def __init__(
        self,
        cfg: "FeedForwardConfig",
        overrides: "FeedForwardConfig | None" = None,
    ) -> None:
        super().__init__()
        self.cfg: FeedForwardConfig = self._override_config(cfg, overrides)

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.stack_config: ConfigBase = self.cfg.stack_config

        self.VALIDATOR.validate(self)
        self.uses_soft_moe = self.VALIDATOR.uses_soft_moe(self.stack_config)

        self.model = self.__build_stack_model()

    def __build_stack_model(self) -> Module:
        mirrored_config = self.__mirror_stack_config(
            self.stack_config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return mirrored_config.build()

    def __mirror_stack_config(
        self,
        stack_config: ConfigBase,
        *,
        input_dim: int,
        output_dim: int,
    ) -> ConfigBase:
        from emperor.experts import MixtureOfExpertsModelConfig

        if isinstance(stack_config, MirroredLayerStackConfig):
            mirrored = deepcopy(stack_config)
            mirrored.input_dim = input_dim
            mirrored.output_dim = output_dim
            return mirrored
        if isinstance(stack_config, LayerStackConfig):
            values = {
                config_field.name: deepcopy(getattr(stack_config, config_field.name))
                for config_field in fields(LayerStackConfig)
            }
            values.update(input_dim=input_dim, output_dim=output_dim)
            return MirroredLayerStackConfig(**values)
        if isinstance(stack_config, MixtureOfExpertsModelConfig):
            mirrored = deepcopy(stack_config)
            mirrored.input_dim = input_dim
            mirrored.output_dim = output_dim
            mirrored.stack_config = self.__mirror_stack_config(
                mirrored.stack_config,
                input_dim=input_dim,
                output_dim=output_dim,
            )
            return mirrored
        recurrent_config = cast(RecurrentCompositionConfig, stack_config)
        mirrored = deepcopy(recurrent_config)
        mirrored.input_dim = input_dim
        mirrored.output_dim = output_dim
        mirrored._map_transition_configs(
            lambda transition_config: self.__mirror_stack_config(
                transition_config,
                input_dim=output_dim,
                output_dim=output_dim,
            )
        )
        return mirrored

    def forward(
        self,
        input_batch: Tensor,
        *,
        padding_mask: Tensor | None = None,
        batch_first: bool = True,
    ) -> tuple[Tensor, Tensor]:
        self.VALIDATOR.validate_forward_inputs(self, input_batch)
        if self.uses_soft_moe:
            return self.__forward_soft_moe(input_batch, padding_mask, batch_first)
        original_shape = input_batch.shape
        flattened_input = input_batch.reshape(-1, self.input_dim)
        feed_forward_state = Layer.run_model_from_hidden(
            self.model,
            flattened_input,
        )
        output = feed_forward_state.hidden.view(*original_shape[:-1], self.output_dim)
        loss = (
            feed_forward_state.loss
            if feed_forward_state.loss is not None
            else input_batch.new_zeros(())
        )
        return output, loss

    def __forward_soft_moe(
        self, inputs: Tensor, padding_mask: Tensor | None, batch_first: bool
    ) -> tuple[Tensor, Tensor]:
        from emperor.experts import MixtureOfExpertsLayerState

        self.VALIDATOR.validate_soft_layout(inputs, padding_mask, batch_first)
        tokens = inputs if batch_first else inputs.transpose(0, 1)
        active = None
        if padding_mask is not None:
            excluded = (
                padding_mask
                if padding_mask.dtype == torch.bool
                else padding_mask.isneginf()
            )
            active = ~excluded.unsqueeze(-1)
            tokens = tokens.masked_fill(~active, 0)
        state = self.model(MixtureOfExpertsLayerState(hidden=tokens, skip_mask=active))
        output = (
            state.hidden if active is None else state.hidden.masked_fill(~active, 0)
        )
        if not batch_first:
            output = output.transpose(0, 1)
        loss = state.loss if state.loss is not None else inputs.new_zeros(())
        return output, loss
