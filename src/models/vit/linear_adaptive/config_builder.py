from typing import TYPE_CHECKING

import models.vit.linear_adaptive.config as config
from emperor.augmentations.adaptive_parameters import GroupingConfig
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
from models.vit.linear_adaptive._config_defaults import vit_positional_embedding_options
from models.vit.linear_adaptive._core_config_factory import (
    CoreConfigDependencies,
    CoreConfigFactory,
)
from models.vit.linear_adaptive._linear_layer_config_factory import (
    AdaptiveAugmentationConfigFactory,
    AdaptiveAugmentationDependencies,
    LinearLayerConfigDependencies,
    LinearLayerConfigFactory,
)
from models.vit.linear_adaptive._patch_config_factory import (
    PatchConfigDependencies,
    PatchConfigFactory,
)
from models.vit.linear_adaptive.experiment_config import ExperimentConfig
from models.vit.linear_adaptive.runtime_defaults import DEFAULT_RUNTIME
from models.vit.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    DynamicMemoryOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
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


class _VitLinearAdaptiveConfigBuilderImplementation:
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
        attention_projection_dynamic_memory_options: DynamicMemoryOptions | None = None,
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
        adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None = None,
        grouping_config: GroupingConfig | None = None,
        hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None,
        hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None,
        hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None,
        hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None,
        attention_adaptive_generator_stack_options: (
            AdaptiveGeneratorStackOptions | None
        ) = None,
        attention_grouping_config: GroupingConfig | None = None,
        attention_hidden_adaptive_weight_options: (
            HiddenAdaptiveWeightOptions | None
        ) = None,
        attention_hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None,
        attention_hidden_adaptive_diagonal_options: (
            HiddenAdaptiveDiagonalOptions | None
        ) = None,
        attention_hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None,
        feed_forward_adaptive_generator_stack_options: (
            AdaptiveGeneratorStackOptions | None
        ) = None,
        feed_forward_grouping_config: GroupingConfig | None = None,
        feed_forward_hidden_adaptive_weight_options: (
            HiddenAdaptiveWeightOptions | None
        ) = None,
        feed_forward_hidden_adaptive_bias_options: (
            HiddenAdaptiveBiasOptions | None
        ) = None,
        feed_forward_hidden_adaptive_diagonal_options: (
            HiddenAdaptiveDiagonalOptions | None
        ) = None,
        feed_forward_hidden_adaptive_mask_options: (
            HiddenAdaptiveMaskOptions | None
        ) = None,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_options = patch_options
        self.encoder_options = encoder_options
        self.hidden_dim = self.__plain_linear_layer_config_factory().hidden_dim
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
        self.adaptive_generator_stack_options = adaptive_generator_stack_options
        self.hidden_adaptive_weight_options = hidden_adaptive_weight_options
        self.grouping_config = grouping_config
        self.hidden_adaptive_bias_options = hidden_adaptive_bias_options
        self.hidden_adaptive_diagonal_options = hidden_adaptive_diagonal_options
        self.hidden_adaptive_mask_options = hidden_adaptive_mask_options
        self.attention_adaptive_generator_stack_options = (
            attention_adaptive_generator_stack_options
        )
        self.attention_hidden_adaptive_weight_options = (
            attention_hidden_adaptive_weight_options
        )
        self.attention_grouping_config = attention_grouping_config
        self.attention_hidden_adaptive_bias_options = (
            attention_hidden_adaptive_bias_options
        )
        self.attention_hidden_adaptive_diagonal_options = (
            attention_hidden_adaptive_diagonal_options
        )
        self.attention_hidden_adaptive_mask_options = (
            attention_hidden_adaptive_mask_options
        )
        self.feed_forward_adaptive_generator_stack_options = (
            feed_forward_adaptive_generator_stack_options
        )
        self.feed_forward_hidden_adaptive_weight_options = (
            feed_forward_hidden_adaptive_weight_options
        )
        self.feed_forward_grouping_config = feed_forward_grouping_config
        self.feed_forward_hidden_adaptive_bias_options = (
            feed_forward_hidden_adaptive_bias_options
        )
        self.feed_forward_hidden_adaptive_diagonal_options = (
            feed_forward_hidden_adaptive_diagonal_options
        )
        self.feed_forward_hidden_adaptive_mask_options = (
            feed_forward_hidden_adaptive_mask_options
        )
        self.sequence_length = self.__sequence_length()
        self.adaptive_augmentation_config = self.__adaptive_augmentation_config()
        self.attention_adaptive_augmentation_config = (
            self.__attention_adaptive_augmentation_config()
        )
        self.feed_forward_adaptive_augmentation_config = (
            self.__feed_forward_adaptive_augmentation_config()
        )

    def __plain_linear_layer_config_factory(self) -> LinearLayerConfigFactory:
        linear_layer_config_dependencies = self.__linear_layer_config_dependencies()
        return LinearLayerConfigFactory(linear_layer_config_dependencies)

    def __sequence_length(self) -> int:
        patch_config_dependencies = self.__patch_config_dependencies()
        patch_config_factory = PatchConfigFactory(patch_config_dependencies)
        return patch_config_factory.sequence_length

    def __adaptive_augmentation_config(self):
        adaptive_dependencies = self.__adaptive_augmentation_dependencies()
        adaptive_factory = AdaptiveAugmentationConfigFactory(adaptive_dependencies)
        return adaptive_factory.build_adaptive_augmentation_config()

    def __adaptive_augmentation_dependencies(self) -> AdaptiveAugmentationDependencies:
        return AdaptiveAugmentationDependencies(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            adaptive_generator_stack_options=self.adaptive_generator_stack_options,
            grouping_config=self.grouping_config,
            hidden_adaptive_weight_options=self.hidden_adaptive_weight_options,
            hidden_adaptive_bias_options=self.hidden_adaptive_bias_options,
            hidden_adaptive_diagonal_options=self.hidden_adaptive_diagonal_options,
            hidden_adaptive_mask_options=self.hidden_adaptive_mask_options,
        )

    def __attention_adaptive_augmentation_config(self):
        adaptive_dependencies = self.__attention_adaptive_augmentation_dependencies()
        adaptive_factory = AdaptiveAugmentationConfigFactory(adaptive_dependencies)
        return adaptive_factory.build_adaptive_augmentation_config()

    def __attention_adaptive_augmentation_dependencies(
        self,
    ) -> AdaptiveAugmentationDependencies:
        adaptive_generator_stack_options = self.adaptive_generator_stack_options
        if self.attention_adaptive_generator_stack_options is not None:
            adaptive_generator_stack_options = (
                self.attention_adaptive_generator_stack_options
            )
        hidden_adaptive_weight_options = self.hidden_adaptive_weight_options
        if self.attention_hidden_adaptive_weight_options is not None:
            hidden_adaptive_weight_options = (
                self.attention_hidden_adaptive_weight_options
            )
        grouping_config = self.attention_grouping_config
        hidden_adaptive_bias_options = self.hidden_adaptive_bias_options
        if self.attention_hidden_adaptive_bias_options is not None:
            hidden_adaptive_bias_options = self.attention_hidden_adaptive_bias_options
        hidden_adaptive_diagonal_options = self.hidden_adaptive_diagonal_options
        if self.attention_hidden_adaptive_diagonal_options is not None:
            hidden_adaptive_diagonal_options = (
                self.attention_hidden_adaptive_diagonal_options
            )
        hidden_adaptive_mask_options = self.hidden_adaptive_mask_options
        if self.attention_hidden_adaptive_mask_options is not None:
            hidden_adaptive_mask_options = self.attention_hidden_adaptive_mask_options

        return AdaptiveAugmentationDependencies(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            adaptive_generator_stack_options=adaptive_generator_stack_options,
            grouping_config=grouping_config,
            hidden_adaptive_weight_options=hidden_adaptive_weight_options,
            hidden_adaptive_bias_options=hidden_adaptive_bias_options,
            hidden_adaptive_diagonal_options=hidden_adaptive_diagonal_options,
            hidden_adaptive_mask_options=hidden_adaptive_mask_options,
        )

    def __feed_forward_adaptive_augmentation_config(self):
        adaptive_dependencies = self.__feed_forward_adaptive_augmentation_dependencies()
        adaptive_factory = AdaptiveAugmentationConfigFactory(adaptive_dependencies)
        return adaptive_factory.build_adaptive_augmentation_config()

    def __feed_forward_adaptive_augmentation_dependencies(
        self,
    ) -> AdaptiveAugmentationDependencies:
        adaptive_generator_stack_options = self.adaptive_generator_stack_options
        if self.feed_forward_adaptive_generator_stack_options is not None:
            adaptive_generator_stack_options = (
                self.feed_forward_adaptive_generator_stack_options
            )
        hidden_adaptive_weight_options = self.hidden_adaptive_weight_options
        if self.feed_forward_hidden_adaptive_weight_options is not None:
            hidden_adaptive_weight_options = (
                self.feed_forward_hidden_adaptive_weight_options
            )
        grouping_config = self.feed_forward_grouping_config
        hidden_adaptive_bias_options = self.hidden_adaptive_bias_options
        if self.feed_forward_hidden_adaptive_bias_options is not None:
            hidden_adaptive_bias_options = (
                self.feed_forward_hidden_adaptive_bias_options
            )
        hidden_adaptive_diagonal_options = self.hidden_adaptive_diagonal_options
        if self.feed_forward_hidden_adaptive_diagonal_options is not None:
            hidden_adaptive_diagonal_options = (
                self.feed_forward_hidden_adaptive_diagonal_options
            )
        hidden_adaptive_mask_options = self.hidden_adaptive_mask_options
        if self.feed_forward_hidden_adaptive_mask_options is not None:
            hidden_adaptive_mask_options = (
                self.feed_forward_hidden_adaptive_mask_options
            )

        return AdaptiveAugmentationDependencies(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            adaptive_generator_stack_options=adaptive_generator_stack_options,
            grouping_config=grouping_config,
            hidden_adaptive_weight_options=hidden_adaptive_weight_options,
            hidden_adaptive_bias_options=hidden_adaptive_bias_options,
            hidden_adaptive_diagonal_options=hidden_adaptive_diagonal_options,
            hidden_adaptive_mask_options=hidden_adaptive_mask_options,
        )

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
        linear_layer_config_factory = self.__plain_linear_layer_config_factory()
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
        linear_layer_config_factory = self.__plain_linear_layer_config_factory()
        attention_projection_linear_layer_config_factory = (
            self.__adaptive_linear_layer_config_factory(
                self.attention_adaptive_augmentation_config
            )
        )
        feed_forward_linear_layer_config_factory = (
            self.__adaptive_linear_layer_config_factory(
                self.feed_forward_adaptive_augmentation_config
            )
        )
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
            attention_projection_linear_layer_config_factory=(
                attention_projection_linear_layer_config_factory
            ),
            feed_forward_linear_layer_config_factory=(
                feed_forward_linear_layer_config_factory
            ),
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

    def __adaptive_linear_layer_config_factory(
        self, adaptive_augmentation_config
    ) -> LinearLayerConfigFactory:
        linear_layer_config_dependencies = self.__linear_layer_config_dependencies(
            adaptive_augmentation_config=adaptive_augmentation_config
        )
        return LinearLayerConfigFactory(linear_layer_config_dependencies)

    def __linear_layer_config_dependencies(
        self,
        *,
        adaptive_augmentation_config=None,
    ) -> LinearLayerConfigDependencies:
        return LinearLayerConfigDependencies(
            encoder_options=self.encoder_options,
            adaptive_augmentation_config=adaptive_augmentation_config,
        )


class VitLinearAdaptiveConfigBuilder(
    _attention_settings.RuntimeBuilder[RuntimeOptions],
    _VitLinearAdaptiveConfigBuilderImplementation,
):
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        self._configure_runtime(runtime)

    def _configure_runtime(self, runtime: RuntimeOptions) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.vit.linear_adaptive VitLinearAdaptiveConfigBuilder runtime must be RuntimeOptions"
            )
        self._runtime = runtime
        super().__init__(**runtime._as_construction_kwargs())
