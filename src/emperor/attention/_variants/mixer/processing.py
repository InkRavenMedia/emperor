"""Configure and execute token-axis models through their sequence interface."""

from typing import TYPE_CHECKING, TypeVar, cast

from torch import Tensor

from emperor.attention._variants.mixer.validation import MixerAttentionValidator
from emperor.config import ConfigBase
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.attention._variants.mixer.config import MixerAttentionConfig

_Config = TypeVar("_Config", bound=ConfigBase)


class MixerProcessor(Module):
    VALIDATOR = MixerAttentionValidator

    def __init__(
        self,
        cfg: "MixerAttentionConfig",
        validator: type[MixerAttentionValidator],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.VALIDATOR = validator
        self.sequence_length = cast(int, self.cfg.sequence_length)
        self.memory_sequence_length = self.__calculate_memory_sequence_length()
        self.mixing_sequence_length = self.sequence_length + self.memory_sequence_length
        self.mixing_model = self.__build_mixing_model()

    def __calculate_memory_sequence_length(self) -> int:
        source_memory_config = self.cfg.source_memory_config
        if source_memory_config is None:
            return 0
        return source_memory_config.source_memory_length(self.sequence_length)

    def __build_mixing_model(self) -> Module:
        model_config = cast(ConfigBase, self.cfg.mixing_model_config)
        mixing_config = self.__prepare_model_config(model_config)
        return mixing_config.build()

    def __prepare_model_config(self, config: _Config) -> _Config:
        from emperor.experts import MixtureOfExpertsModelConfig
        from emperor.layers import RecurrentCompositionConfig

        overrides = self._resolve_config_overrides(
            config,
            input_dim=self.mixing_sequence_length,
            output_dim=self.mixing_sequence_length,
        )
        mixing_config = cast(_Config, self._override_config(config, overrides))
        if isinstance(mixing_config, MixtureOfExpertsModelConfig):
            if mixing_config.stack_config is not None:
                mixing_config.stack_config = self.__prepare_model_config(
                    mixing_config.stack_config
                )
        elif isinstance(mixing_config, RecurrentCompositionConfig):
            mixing_config._map_transition_configs(self.__prepare_model_config)
        return mixing_config

    def compute_mixing(
        self,
        token_axis_vectors: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        mixing_state = self.mixing_model.forward_sequence(token_axis_vectors)
        self.VALIDATOR.validate_mixing_state(mixing_state, token_axis_vectors)
        return mixing_state.hidden, mixing_state.loss
