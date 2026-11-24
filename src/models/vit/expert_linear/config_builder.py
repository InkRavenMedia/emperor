from typing import TYPE_CHECKING

import models.vit.expert_linear.config as config
from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.linears import LinearLayerConfig
from models import _attention_settings, _sequence_expansion
from models._model_config import sequence_model_config
from models._positional_embedding import build_position_config
from models.vit.expert_linear._config_defaults import vit_positional_embedding_options
from models.vit.expert_linear._core_config_factory import (
    CoreConfigDependencies,
    CoreConfigFactory,
)
from models.vit.expert_linear._expert_config_factory import (
    ExpertConfigDependencies,
    ExpertConfigFactory,
)
from models.vit.expert_linear._linear_layer_config_factory import (
    LinearLayerConfigDependencies,
    LinearLayerConfigFactory,
)
from models.vit.expert_linear._patch_config_factory import (
    PatchConfigDependencies,
    PatchConfigFactory,
)
from models.vit.expert_linear.experiment_config import ExperimentConfig
from models.vit.expert_linear.runtime_defaults import DEFAULT_RUNTIME
from models.vit.expert_linear.runtime_options import (
    DynamicMemoryOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsSubmoduleStackOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    RuntimeOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
    VitOutputOptions,
    VitPatchOptions,
)

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class _VitExpertLinearConfigBuilderImplementation:
    def __init__(
        self,
        *,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        output_dim: int = config.OUTPUT_DIM,
        patch_options: VitPatchOptions | None = None,
        encoder_options: TransformerEncoderOptions | None = None,
        positional_embedding_options: (
            TransformerPositionalEmbeddingOptions | None
        ) = None,
        attention_options: TransformerAttentionOptions | None = None,
        feed_forward_options: TransformerFeedForwardOptions | None = None,
        output_options: VitOutputOptions | None = None,
        attention_projection_stack_options: SubmoduleStackOptions | None = None,
        attention_projection_layer_controller_options: (
            LayerControllerOptions | None
        ) = None,
        attention_projection_dynamic_memory_options: (
            DynamicMemoryOptions | None
        ) = None,
        attention_projection_recurrent_controller_options: (
            RecurrentControllerOptions | None
        ) = None,
        feed_forward_stack_options: SubmoduleStackOptions | None = None,
        feed_forward_layer_controller_options: LayerControllerOptions | None = (None),
        feed_forward_dynamic_memory_options: DynamicMemoryOptions | None = (None),
        feed_forward_recurrent_controller_options: (
            RecurrentControllerOptions | None
        ) = None,
        stack_options: MainLayerStackOptions | None = None,
        submodule_stack_options: SubmoduleStackOptions | None = None,
        layer_controller_options: LayerControllerOptions | None = (None),
        dynamic_memory_options: DynamicMemoryOptions | None = (None),
        recurrent_controller_options: RecurrentControllerOptions | None = (None),
        mixture_options: ExpertsMixtureOptions | None = None,
        expert_stack_options: ExpertsSubmoduleStackOptions | None = None,
        sampler_options: ExpertsSamplerOptions | None = None,
        router_options: ExpertsRouterOptions | None = None,
        router_stack_options: ExpertsSubmoduleStackOptions | None = None,
        expert_layer_controller_options: ExpertsLayerControllerOptions | None = None,
        expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        expert_recurrent_controller_options: (
            ExpertsRecurrentControllerOptions | None
        ) = None,
        expert_attention_use_kv_expert_models_flag: bool = (
            config.EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG
        ),
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_options = patch_options
        self.encoder_options = encoder_options
        self.hidden_dim = self.__linear_layer_config_factory().hidden_dim
        self.positional_embedding_options = positional_embedding_options
        self.attention_options = attention_options
        self.feed_forward_options = feed_forward_options
        self.output_options = output_options
        self.attention_projection_stack_options = attention_projection_stack_options
        self.attention_projection_layer_controller_options = (
            attention_projection_layer_controller_options
        )
        self.attention_projection_dynamic_memory_options = (
            attention_projection_dynamic_memory_options
        )
        self.attention_projection_recurrent_controller_options = (
            attention_projection_recurrent_controller_options
        )
        self.feed_forward_stack_options = feed_forward_stack_options
        self.feed_forward_layer_controller_options = (
            feed_forward_layer_controller_options
        )
        self.feed_forward_dynamic_memory_options = feed_forward_dynamic_memory_options
        self.feed_forward_recurrent_controller_options = (
            feed_forward_recurrent_controller_options
        )
        self.encoder_stack_options = stack_options
        self.encoder_submodule_stack_options = submodule_stack_options
        self.encoder_layer_controller_options = layer_controller_options
        self.encoder_dynamic_memory_options = dynamic_memory_options
        self.encoder_recurrent_controller_options = recurrent_controller_options
        self.mixture_options = mixture_options
        self.expert_stack_options = expert_stack_options
        self.sampler_options = sampler_options
        self.router_options = router_options
        self.router_stack_options = router_stack_options
        self.expert_layer_controller_options = expert_layer_controller_options
        self.expert_dynamic_memory_options = expert_dynamic_memory_options
        self.expert_recurrent_controller_options = expert_recurrent_controller_options
        self.expert_attention_use_kv_expert_models_flag = (
            expert_attention_use_kv_expert_models_flag
        )
        self.sequence_length = self.__patch_config_factory().sequence_length

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
        return self.__patch_config_factory().build_patch_config()

    def __patch_config_factory(self) -> PatchConfigFactory:
        patch_config_dependencies = self.__patch_config_dependencies()
        return PatchConfigFactory(patch_config_dependencies)

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
        expert_config_factory = self.__expert_config_factory()
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
            expert_config_factory=expert_config_factory,
        )

    def __output_config(self) -> LinearLayerConfig:
        bias_flag = config.OUTPUT_BIAS_FLAG
        if self.output_options is not None:
            bias_flag = self.output_options.bias_flag
        return LinearLayerConfig(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
            bias_flag=bias_flag,
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

    def __expert_config_factory(self) -> ExpertConfigFactory:
        expert_config_dependencies = self.__expert_config_dependencies()
        return ExpertConfigFactory(expert_config_dependencies)

    def __expert_config_dependencies(self) -> ExpertConfigDependencies:
        return ExpertConfigDependencies(
            hidden_dim=self.hidden_dim,
            encoder_options=self.encoder_options,
            attention_options=self.attention_options,
            feed_forward_options=self.feed_forward_options,
            mixture_options=self.mixture_options,
            expert_stack_options=self.expert_stack_options,
            sampler_options=self.sampler_options,
            router_options=self.router_options,
            router_stack_options=self.router_stack_options,
            expert_layer_controller_options=self.expert_layer_controller_options,
            expert_dynamic_memory_options=self.expert_dynamic_memory_options,
            expert_recurrent_controller_options=(
                self.expert_recurrent_controller_options
            ),
            expert_attention_use_kv_expert_models_flag=(
                self.expert_attention_use_kv_expert_models_flag
            ),
        )


class VitExpertLinearConfigBuilder(
    _attention_settings.RuntimeBuilder[RuntimeOptions],
    _VitExpertLinearConfigBuilderImplementation,
):
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        self._configure_runtime(runtime)

    def _configure_runtime(self, runtime: RuntimeOptions) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.vit.expert_linear VitExpertLinearConfigBuilder runtime must be RuntimeOptions"
            )
        self._runtime = runtime
        super().__init__(**runtime._as_construction_kwargs())
