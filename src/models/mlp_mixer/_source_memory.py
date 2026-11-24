from collections.abc import Callable
from dataclasses import dataclass

from emperor.attention import AttentionSourceMemoryConfig
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions,
    GroupingConfig,
    SumGroupingConfig,
)
from emperor.config import ConfigBase
from emperor.linears import LinearLayerConfig
from emperor.memory import DynamicMemoryConfig
from models._projection_model import ProjectionModelOptions, resolve_projection_model


@dataclass(frozen=True, kw_only=True)
class TokenMixerSourceMemoryOptions:
    token_mixer_source_memory_chunk_size: int | None = None
    token_mixer_source_memory_grouping_method: type[GroupingConfig] = SumGroupingConfig
    token_mixer_source_memory_model_config: ConfigBase | None = None
    token_mixer_source_memory_model_option: ProjectionModelOptions | None = None
    token_mixer_source_memory_model_bias_flag: bool | None = None
    token_mixer_source_memory_model_stack_num_layers: int | None = None
    token_mixer_source_memory_model_stack_hidden_dim: int | None = None
    token_mixer_source_memory_model_recurrent_max_steps: int | None = None

    def build_token_mixer_source_memory(
        self,
        template_source: ConfigBase,
        per_token_memory: Callable[[], DynamicMemoryConfig | None],
        test_time_training: bool,
    ) -> DynamicMemoryConfig | AttentionSourceMemoryConfig | None:
        chunk_size = self.token_mixer_source_memory_chunk_size
        if chunk_size is None:
            if (
                self.token_mixer_source_memory_model_option
                is ProjectionModelOptions.ADAPTIVE_LINEAR
            ):
                raise ValueError(
                    "Adaptive token-mixer source memory requires "
                    "--token-mixer-source-memory-chunk-size; use 1 for one vector per token."
                )
            return per_token_memory()
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError(
                "token_mixer_source_memory_chunk_size must be a positive integer"
            )
        if test_time_training:
            raise ValueError(
                "Chunked source memory does not support per-token memory test-time training"
            )
        return AttentionSourceMemoryConfig(
            model_config=self._memory_projection(template_source),
            grouping_config=self.token_mixer_source_memory_grouping_method(
                scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                chunk_size=chunk_size,
            ),
        )

    def _memory_projection(self, template_source: ConfigBase) -> ConfigBase:
        return resolve_projection_model(
            self.token_mixer_source_memory_model_config,
            self.token_mixer_source_memory_model_option,
            self.token_mixer_source_memory_model_bias_flag,
            stack_num_layers=self.token_mixer_source_memory_model_stack_num_layers,
            stack_hidden_dim=self.token_mixer_source_memory_model_stack_hidden_dim,
            recurrent_max_steps=self.token_mixer_source_memory_model_recurrent_max_steps,
            template_source=template_source,
        ) or LinearLayerConfig(bias_flag=True)
