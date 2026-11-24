from typing import TYPE_CHECKING

from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
)
from emperor.linears import LinearLayerConfig
from models import _attention_settings, _sequence_expansion
from models._model_config import sequence_model_config
from models._positional_embedding import build_position_config
from models.vit.linear._config_defaults import vit_positional_embedding_options
from models.vit.linear._core_config_factory import (
    CoreConfigDependencies,
    CoreConfigFactory,
)
from models.vit.linear._linear_layer_config_factory import (
    LinearLayerConfigDependencies,
    LinearLayerConfigFactory,
)
from models.vit.linear._patch_config_factory import (
    PatchConfigDependencies,
    PatchConfigFactory,
)
from models.vit.linear.experiment_config import ExperimentConfig
from models.vit.linear.runtime_defaults import DEFAULT_RUNTIME
from models.vit.linear.runtime_options import RuntimeOptions

from . import config

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class VitLinearConfigBuilder(_attention_settings.RuntimeBuilder[RuntimeOptions]):
    def __init__(
        self,
        *,
        runtime: RuntimeOptions = DEFAULT_RUNTIME,
    ) -> None:
        self._configure_runtime(runtime)

    def _configure_runtime(self, runtime: RuntimeOptions) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")
        self._runtime = runtime
        self.batch_size = self.runtime.batch_size
        self.learning_rate = self.runtime.learning_rate
        self.input_dim = self.runtime.input_dim
        self.output_dim = self.runtime.output_dim
        self.patch_options = self.runtime.patch_options
        self.encoder_options = self.runtime.encoder_options
        self.hidden_dim = self.__linear_layer_config_factory().hidden_dim
        self.positional_embedding_options = self.runtime.positional_embedding_options
        self.attention_options = self.runtime.attention_options
        self.feed_forward_options = self.runtime.feed_forward_options
        self.output_options = self.runtime.output_options
        self.attention_projection_stack_options = (
            self.runtime.attention_projection_stack_options
        )
        self.attention_projection_layer_controller_options = (
            self.runtime.attention_projection_layer_controller_options
        )
        self.attention_projection_dynamic_memory_options = (
            self.runtime.attention_projection_dynamic_memory_options
        )
        self.attention_projection_recurrent_controller_options = (
            self.runtime.attention_projection_recurrent_controller_options
        )
        self.feed_forward_stack_options = self.runtime.feed_forward_stack_options
        self.feed_forward_layer_controller_options = (
            self.runtime.feed_forward_layer_controller_options
        )
        self.feed_forward_dynamic_memory_options = (
            self.runtime.feed_forward_dynamic_memory_options
        )
        self.feed_forward_recurrent_controller_options = (
            self.runtime.feed_forward_recurrent_controller_options
        )
        self.encoder_stack_options = self.runtime.stack_options
        self.encoder_submodule_stack_options = self.runtime.submodule_stack_options
        self.encoder_layer_controller_options = self.runtime.layer_controller_options
        self.encoder_dynamic_memory_options = self.runtime.dynamic_memory_options
        self.encoder_recurrent_controller_options = (
            self.runtime.recurrent_controller_options
        )
        self.sequence_length = self.__sequence_length()

    def __sequence_length(self) -> int:
        patch_config_dependencies = self.__patch_config_dependencies()
        patch_config_factory = PatchConfigFactory(patch_config_dependencies)
        return patch_config_factory.sequence_length

    def build(self) -> "ModelConfig":
        patch_config = self.__patch_config()
        positional_embedding_config = self.__positional_embedding_config()
        encoder_config = self.__encoder_config()
        output_config = self.__output_config()
        encoder_output_normalization = config.ENCODER_OUTPUT_NORMALIZATION
        if self.encoder_options is not None:
            encoder_output_normalization = self.encoder_options.output_normalization
        experiment_config = ExperimentConfig(
            patch_config=patch_config,
            positional_embedding_config=positional_embedding_config,
            encoder_config=encoder_config,
            output_config=output_config,
            encoder_output_normalization=encoder_output_normalization,
        )
        return sequence_model_config(self, experiment_config)

    def __patch_config(self):
        patch_config_dependencies = self.__patch_config_dependencies()
        patch_config_factory = PatchConfigFactory(patch_config_dependencies)
        return patch_config_factory.build_patch_config()

    def __patch_config_dependencies(self) -> PatchConfigDependencies:
        linear_layer_config_factory = self.__linear_layer_config_factory()
        return PatchConfigDependencies(
            hidden_dim=self.hidden_dim,
            patch_options=self.patch_options,
            encoder_options=self.encoder_options,
            linear_layer_config_factory=linear_layer_config_factory,
        )

    def __positional_embedding_config(self) -> AbsolutePositionalEmbeddingConfig:
        options = self.positional_embedding_options
        if options is None:
            options = vit_positional_embedding_options(config)
        return build_position_config(
            options.option,
            num_embeddings=self.sequence_length - 1,
            embedding_dim=self.hidden_dim,
            padding_idx=options.padding_idx,
            auto_expand_flag=options.auto_expand_flag,
            class_token_flag=True,
        )

    def __encoder_config(self):
        core_config_dependencies = self.__core_config_dependencies()
        core_config_factory = CoreConfigFactory(core_config_dependencies)
        encoder_config = core_config_factory.build_encoder_config()
        return _sequence_expansion.with_expansion_attention_capacity(
            self.runtime.configure_stack(encoder_config), self.sequence_length
        )

    def __core_config_dependencies(self) -> CoreConfigDependencies:
        linear_layer_config_factory = self.__linear_layer_config_factory()
        return CoreConfigDependencies(
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            encoder_options=self.encoder_options,
            attention_options=self.attention_options,
            feed_forward_options=self.feed_forward_options,
            attention_projection_stack_options=self.attention_projection_stack_options,
            attention_projection_layer_controller_options=(
                self.attention_projection_layer_controller_options
            ),
            attention_projection_dynamic_memory_options=(
                self.attention_projection_dynamic_memory_options
            ),
            attention_projection_recurrent_controller_options=(
                self.attention_projection_recurrent_controller_options
            ),
            feed_forward_stack_options=self.feed_forward_stack_options,
            feed_forward_layer_controller_options=(
                self.feed_forward_layer_controller_options
            ),
            feed_forward_dynamic_memory_options=(
                self.feed_forward_dynamic_memory_options
            ),
            feed_forward_recurrent_controller_options=(
                self.feed_forward_recurrent_controller_options
            ),
            stack_options=self.encoder_stack_options,
            submodule_stack_options=self.encoder_submodule_stack_options,
            layer_controller_options=self.encoder_layer_controller_options,
            dynamic_memory_options=self.encoder_dynamic_memory_options,
            recurrent_controller_options=self.encoder_recurrent_controller_options,
            linear_layer_config_factory=linear_layer_config_factory,
        )

    def __output_config(self) -> LayerConfig:
        bias_flag = config.OUTPUT_BIAS_FLAG
        if self.output_options is not None:
            bias_flag = self.output_options.bias_flag
        layer_model_config = LinearLayerConfig(
            bias_flag=bias_flag,
        )
        return LayerConfig(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=layer_model_config,
        )

    def __linear_layer_config_factory(self) -> LinearLayerConfigFactory:
        linear_layer_config_dependencies = self.__linear_layer_config_dependencies()
        return LinearLayerConfigFactory(linear_layer_config_dependencies)

    def __linear_layer_config_dependencies(
        self,
    ) -> LinearLayerConfigDependencies:
        return LinearLayerConfigDependencies(
            encoder_options=self.encoder_options,
        )
