from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.layers import LayerState
from emperor.neuron._neuron.validation import NeuronValidator
from emperor.nn import Module
from emperor.transformer import TransformerEncoderLayerState

if TYPE_CHECKING:
    from emperor.neuron._config import NeuronConfig


class Neuron(Module):
    """Compose signal processing, memory, and spatial routing for one neuron."""

    COORDINATE_EMBEDDING_FREQUENCY_BASE = 10000.0
    VALIDATOR = NeuronValidator

    def __init__(
        self,
        cfg: "NeuronConfig",
        overrides: "NeuronConfig | None" = None,
    ):
        super().__init__()
        neuron_config = getattr(cfg, "neuron_config", cfg)
        self.cfg: NeuronConfig = self._override_config(neuron_config, overrides)
        self.VALIDATOR.validate(self.cfg)
        self.coordinate_embedding_flag: bool = bool(self.cfg.coordinate_embedding_flag)
        self.__initialize_components()
        self.__initialize_counters()
        self.__register_coordinate_embedding()

    def __initialize_components(self) -> None:
        self.nucleus = self.cfg.nucleus_config.build()
        self.axons = self.cfg.axons_config.build()
        self.terminal = self.cfg.terminal_config.build()
        residual = self.cfg.sequence_residual_config
        self.sequence_residual = residual.build() if residual is not None else None

    def __initialize_counters(self) -> None:
        self.register_buffer(
            "batch_counter",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "atrophy_counter",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )

    def __register_coordinate_embedding(self) -> None:
        if self.coordinate_embedding_flag:
            self.register_buffer(
                "coordinate_embedding",
                self.__initialize_coordinate_embedding(),
                persistent=False,
            )
        else:
            self.coordinate_embedding = None

    def __initialize_coordinate_embedding(self) -> Tensor:
        axis_positions = (
            self.terminal.x_axis_position,
            self.terminal.y_axis_position,
            self.terminal.z_axis_position,
        )
        axis_channel_counts = self.__split_channels_across_axes(self.terminal.input_dim)
        axis_encodings = [
            self.__sinusoidal_axis_encoding(axis_position, axis_channel_count)
            for axis_position, axis_channel_count in zip(
                axis_positions,
                axis_channel_counts,
                strict=True,
            )
        ]
        return torch.cat(axis_encodings)

    def __split_channels_across_axes(self, input_dim: int) -> tuple[int, int, int]:
        base_channel_count = input_dim // 3
        remainder_channel_count = input_dim % 3
        return tuple(
            base_channel_count + (1 if axis_index < remainder_channel_count else 0)
            for axis_index in range(3)
        )

    def __sinusoidal_axis_encoding(
        self,
        axis_position: int,
        axis_channel_count: int,
    ) -> Tensor:
        channel_indices = torch.arange(axis_channel_count, dtype=torch.float32)
        frequency_exponents = (channel_indices - channel_indices % 2) / float(
            axis_channel_count
        )
        sinusoidal_angles = float(axis_position) / torch.pow(
            torch.tensor(self.COORDINATE_EMBEDDING_FREQUENCY_BASE),
            frequency_exponents,
        )
        return torch.where(
            channel_indices % 2 == 0,
            torch.sin(sinusoidal_angles),
            torch.cos(sinusoidal_angles),
        )

    def forward(
        self, input: Tensor | TransformerEncoderLayerState
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | TransformerEncoderLayerState:
        if isinstance(input, TransformerEncoderLayerState):
            return self.process_sequence(input)
        signal = self.process_signal_state(input)
        probabilities, selected_neurons, auxiliary_loss = self.route_signal(
            signal.hidden
        )
        if signal.loss is not None:
            auxiliary_loss = auxiliary_loss + signal.loss
        return signal.hidden, probabilities, selected_neurons, auxiliary_loss

    def process_signal(self, input: Tensor) -> Tensor:
        embedded_input = self.__signal_input(input)
        record_loss = LayerState.loss_recorder(self)
        if record_loss is None:
            return self.axons(self.nucleus(embedded_input))
        state = LayerState.run_tensor_module(self.nucleus, embedded_input)
        record_loss(state.loss)
        return self.axons(state.hidden)

    def process_signal_state(self, input: Tensor) -> LayerState:
        return LayerState.run_tensor_module(self, input, forward=self.process_signal)

    def __signal_input(self, input: Tensor) -> Tensor:
        self.VALIDATOR.validate_forward_input(input)
        self.VALIDATOR.validate_feature_dimension(self, input)
        if self.training:
            self.batch_counter += 1
        return self.__inject_coordinate_embedding(input)

    def process_sequence(
        self, state: TransformerEncoderLayerState
    ) -> TransformerEncoderLayerState:
        """Run one visit, with fresh inner halting, residual and loss state."""
        if self.training:
            self.batch_counter += 1
        local = replace(
            state,
            hidden=self.__inject_coordinate_embedding(state.hidden),
            loss=None,
            residual_state=None,
            halting_state=None,
        )
        result = self.nucleus(local)
        hidden = self.axons.process_sequence(result.hidden, result.key_padding_mask)
        loss = result.loss
        if state.loss is not None:
            loss = state.loss if loss is None else state.loss + loss
        return replace(
            result, hidden=hidden, loss=loss, residual_state=None, halting_state=None
        )

    def __inject_coordinate_embedding(self, input: Tensor) -> Tensor:
        if self.coordinate_embedding is None:
            return input
        return input + self.coordinate_embedding.to(
            device=input.device,
            dtype=input.dtype,
        )

    def route_signal(self, processed_signal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self.VALIDATOR.validate_forward_input(processed_signal)
        self.VALIDATOR.validate_feature_dimension(self, processed_signal)
        _, probabilities, selected_neurons, auxiliary_loss = self.terminal(
            self.__inject_coordinate_embedding(processed_signal)
        )
        return probabilities, selected_neurons, auxiliary_loss
