from typing import TYPE_CHECKING

from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from models import _attention_settings, _sequence_expansion
from models._model_config import sequence_model_config
from models._positional_embedding import build_position_config
from models.bert.linear._boundary_config_factory import (
    BoundaryConfigDependencies,
    BoundaryConfigFactory,
)
from models.bert.linear._config_defaults import bert_positional_embedding_options
from models.bert.linear._core_config_factory import (
    CoreConfigDependencies,
    CoreConfigFactory,
)
from models.bert.linear._linear_layer_config_factory import (
    LinearLayerConfigDependencies,
    LinearLayerConfigFactory,
)
from models.bert.linear.experiment_config import ExperimentConfig
from models.bert.linear.runtime_defaults import DEFAULT_RUNTIME
from models.bert.linear.runtime_options import RuntimeOptions

from . import config

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class BertLinearConfigBuilder(_attention_settings.RuntimeBuilder[RuntimeOptions]):
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
        self.batch_size = runtime.batch_size
        self.learning_rate = runtime.learning_rate
        self.input_dim = runtime.input_dim
        self.output_dim = runtime.output_dim
        self.sequence_length = runtime.sequence_length
        self.embedding_options = runtime.embedding_options
        self.encoder_options = runtime.encoder_options
        self.hidden_dim = self.__linear_layer_config_factory().hidden_dim
        self.positional_embedding_options = runtime.positional_embedding_options
        self.attention_options = runtime.attention_options
        self.feed_forward_options = runtime.feed_forward_options
        self.mlm_head_options = runtime.mlm_head_options
        self.nsp_head_options = runtime.nsp_head_options
        self.attention_projection_stack_options = (
            runtime.attention_projection_stack_options
        )
        self.attention_projection_layer_controller_options = (
            runtime.attention_projection_layer_controller_options
        )
        self.attention_projection_dynamic_memory_options = (
            runtime.attention_projection_dynamic_memory_options
        )
        self.attention_projection_recurrent_controller_options = (
            runtime.attention_projection_recurrent_controller_options
        )
        self.feed_forward_stack_options = runtime.feed_forward_stack_options
        self.feed_forward_layer_controller_options = (
            runtime.feed_forward_layer_controller_options
        )
        self.feed_forward_dynamic_memory_options = (
            runtime.feed_forward_dynamic_memory_options
        )
        self.feed_forward_recurrent_controller_options = (
            runtime.feed_forward_recurrent_controller_options
        )
        self.encoder_stack_options = runtime.stack_options
        self.encoder_submodule_stack_options = runtime.submodule_stack_options
        self.encoder_layer_controller_options = runtime.layer_controller_options
        self.encoder_dynamic_memory_options = runtime.dynamic_memory_options
        self.encoder_recurrent_controller_options = runtime.recurrent_controller_options

    def build(self) -> "ModelConfig":
        positional_embedding_config = self.__positional_embedding_config()
        boundary_config = self.__boundary_config()
        encoder_config = self.__encoder_config()
        encoder_output_normalization = config.ENCODER_OUTPUT_NORMALIZATION
        if self.encoder_options is not None:
            encoder_output_normalization = self.encoder_options.output_normalization
        experiment_config = ExperimentConfig(
            positional_embedding_config=positional_embedding_config,
            boundary_config=boundary_config,
            encoder_config=encoder_config,
            encoder_output_normalization=encoder_output_normalization,
        )
        return sequence_model_config(self, experiment_config)

    def __positional_embedding_config(self) -> AbsolutePositionalEmbeddingConfig:
        options = self.positional_embedding_options
        if not options:
            options = bert_positional_embedding_options(config)
        return build_position_config(
            options.option,
            num_embeddings=self.sequence_length,
            embedding_dim=self.hidden_dim,
            padding_idx=options.padding_idx,
            auto_expand_flag=options.auto_expand_flag,
        )

    def __boundary_config(self):
        boundary_config_dependencies = self.__boundary_config_dependencies()
        boundary_config_factory = BoundaryConfigFactory(boundary_config_dependencies)
        return boundary_config_factory.build_boundary_config()

    def __boundary_config_dependencies(self) -> BoundaryConfigDependencies:
        return BoundaryConfigDependencies(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            embedding_options=self.embedding_options,
            mlm_head_options=self.mlm_head_options,
            nsp_head_options=self.nsp_head_options,
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
            attention_projection_stack_options=(
                self.attention_projection_stack_options
            ),
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

    def __linear_layer_config_factory(self) -> LinearLayerConfigFactory:
        linear_layer_config_dependencies = self.__linear_layer_config_dependencies()
        return LinearLayerConfigFactory(linear_layer_config_dependencies)

    def __linear_layer_config_dependencies(
        self,
    ) -> LinearLayerConfigDependencies:
        return LinearLayerConfigDependencies(
            encoder_options=self.encoder_options,
        )
