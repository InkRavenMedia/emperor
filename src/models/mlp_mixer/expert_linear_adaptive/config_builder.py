from __future__ import annotations

from emperor.config import ModelConfig
from models import _sequence_expansion
from models._model_config import sequence_model_config

from ._building import encoder_config, output_config, patch_config, sequence_length
from .experiment_config import ExperimentConfig
from .runtime_defaults import DEFAULT_RUNTIME
from .runtime_options import RuntimeOptions


class MlpMixerExpertLinearAdaptiveConfigBuilder:
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        self.runtime = runtime

    @property
    def runtime(self) -> RuntimeOptions:
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: RuntimeOptions) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")
        length = sequence_length(runtime)
        self.batch_size = runtime.batch_size
        self.learning_rate = runtime.learning_rate
        self.input_dim = runtime.input_dim
        self.hidden_dim = runtime.hidden_dim
        self.output_dim = runtime.output_dim
        self.sequence_length = length
        self._runtime = runtime

    def build(self) -> ModelConfig:
        runtime = self.runtime
        if runtime.hierarchical_masked_word_flag:
            from ._masked_word import build_experiment_config

            experiment_config = build_experiment_config(runtime)
        else:
            patch_embedding_config = patch_config(runtime)
            encoder_stack_config = encoder_config(runtime, self.sequence_length)
            expanded_encoder_config = (
                _sequence_expansion.with_expansion_attention_capacity(
                    self.runtime.configure_stack(encoder_stack_config),
                    self.sequence_length,
                )
            )
            output_layer_config = output_config(runtime)
            experiment_config = ExperimentConfig(
                patch_config=patch_embedding_config,
                encoder_config=expanded_encoder_config,
                output_config=output_layer_config,
                encoder_output_normalization=runtime.encoder_output_normalization,
            )
        return sequence_model_config(self, experiment_config)


__all__ = ["MlpMixerExpertLinearAdaptiveConfigBuilder"]
