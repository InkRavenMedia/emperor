from typing import TYPE_CHECKING

from emperor.layers import Layer, LayerConfig, LayerState

if TYPE_CHECKING:
    from emperor.transformer.sequence_expansion._config import SequenceExpansionConfig
    from emperor.transformer.sequence_expansion._model import SequenceExpansion


class _SequenceExpansionBlockLayer(Layer):
    """Apply optional expansion before a transformer block captures its input."""

    def __init__(self, cfg: LayerConfig, overrides: LayerConfig | None = None) -> None:
        super().__init__(cfg, overrides)
        self.sequence_expansion_config: SequenceExpansionConfig | None = getattr(
            self.cfg, "sequence_expansion_config", None
        )
        self.sequence_expansion = self.__build_sequence_expansion()

    def __build_sequence_expansion(self) -> "SequenceExpansion | None":
        return self._build_from_config(
            self.sequence_expansion_config,
            embedding_dim=self.input_dim,
        )

    def _handle_layer_input(self, state: LayerState) -> LayerState:
        if self.sequence_expansion is None:
            return state
        self.VALIDATOR.validate_expansion_state(state)
        expansion_result = self.sequence_expansion(
            state.hidden,
            state.expansion_padding_mask,
            position_offset=state.expansion_position_offset,
        )
        state.apply_expansion(expansion_result)
        if expansion_result.loss is not None:
            state.loss = self._accumulate_auxiliary_loss(
                state.loss, self._reduce_auxiliary_loss(expansion_result.loss)
            )
        return state

    def _bind_shared_gate(self, config, model) -> None:
        if self.sequence_expansion is not None:
            raise ValueError("SequenceExpansion does not support shared gates.")
        super()._bind_shared_gate(config, model)

    def _bind_shared_halting(self, model) -> None:
        if self.sequence_expansion is not None:
            raise ValueError("SequenceExpansion does not support shared halting.")
        super()._bind_shared_halting(model)

    def _bind_shared_memory(self, model) -> None:
        if self.sequence_expansion is not None:
            raise ValueError("SequenceExpansion does not support shared layer memory.")
        super()._bind_shared_memory(model)
