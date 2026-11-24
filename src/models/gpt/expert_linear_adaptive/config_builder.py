import copy
from collections.abc import Iterator
from dataclasses import fields
from typing import TYPE_CHECKING, TypeVar, cast

import torch

import models.gpt.expert_linear_adaptive.config as config
from emperor.attention import (
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    GroupingConfig,
)
from emperor.config import ConfigBase
from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
from emperor.embedding.contextual import (
    ByteContextualEmbeddingConfig,
    CausalPrefixKernelConfig,
)
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
from emperor.embedding.relative import DynamicPositionalBiasConfig
from emperor.experts import (
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    RecurrentCompositionConfig,
)
from emperor.transformer import (
    FeedForwardConfig,
    TransformerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from emperor.transformer._soft_moe_config import build_soft_moe_attention_experts
from models import _attention_settings
from models import _attention_source_memory as _source_memory
from models._attention_experts import (
    adaptive_expert_controller,
    attention_expert_stack,
    feed_forward_expert_stack,
    mixture_attention_heads,
)
from models.gpt._hierarchical_config import (
    build_hierarchical_config,
)
from models.gpt.expert_linear_adaptive._base_config_builder import (
    GptBackendConfigBuilder,
)
from models.gpt.expert_linear_adaptive._expert_control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.gpt.expert_linear_adaptive.experiment_config import ExperimentConfig
from models.gpt.expert_linear_adaptive.runtime_defaults import DEFAULT_RUNTIME
from models.gpt.expert_linear_adaptive.runtime_options import (
    ExpertsStackOptions,
    RuntimeOptions,
)

if TYPE_CHECKING:
    from emperor.config import ModelConfig

_StackConfig = TypeVar("_StackConfig", bound=ConfigBase)


def _nested_configs(config_node: object) -> Iterator[ConfigBase]:
    if isinstance(config_node, ConfigBase):
        yield config_node
        for config_field in fields(config_node):
            yield from _nested_configs(getattr(config_node, config_field.name))


def _isolate_token_rows(stack_config: _StackConfig) -> _StackConfig:
    """Copy a stack so each byte-encoded token stays independent of the batch."""
    isolated = copy.deepcopy(stack_config)
    for child in _nested_configs(isolated):
        # Grouping mixes rows; schedules are coordinated by the byte component.
        if isinstance(child, AdaptiveParameterAugmentationConfig):
            child.grouping_config = None
    return isolated


class GptExpertLinearAdaptiveConfigBuilder(
    _attention_settings.RuntimeBuilder[RuntimeOptions], GptBackendConfigBuilder
):
    def build(self) -> "ModelConfig":
        model_config = super().build()
        if self.embedding_options.contextual_flag:
            model_config.experiment_config.contextual_embedding_config = (
                self._build_contextual_embedding_config()
            )
        if self.embedding_options.hierarchical_flag:
            model_config.experiment_config.hierarchical_embedding_config = (
                self._build_hierarchical_embedding_config()
            )
        if self.embedding_options.hierarchical_language_model_flag:
            model_config.experiment_config.hierarchical_language_model_config = (
                build_hierarchical_config(
                    model_config,
                    self.embedding_options,
                )
            )
        return model_config

    def _build_hierarchical_embedding_config(self) -> HierarchicalByteEmbeddingConfig:
        options = self.embedding_options
        # Each token's bytes follow the leading [W] pooling symbol.
        byte_sequence_length = options.hierarchical_max_token_bytes + 1
        byte_position_config = TextLearnedPositionalEmbeddingConfig(
            num_embeddings=byte_sequence_length,
            embedding_dim=self.hidden_dim,
        )
        layer_model_config = self._build_hierarchical_encoder_layer_config(
            byte_sequence_length
        )
        layer_config = TransformerEncoderBlockLayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            layer_model_config=layer_model_config,
        )
        encoder_stack_config = LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=options.hierarchical_encoder_num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=True,
            # The wrapped encoder layer owns its norm, residual, and dropout.
            layer_config=layer_config,
        )
        encoder_config = TransformerConfig(encoder_stack_config=encoder_stack_config)
        linear_stack_config = self._build_linear_stack_config(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=1,
            bias_flag=self.feed_forward_options.bias_flag,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            apply_output_postprocessing_flag=False,
        )
        projection_config = _isolate_token_rows(linear_stack_config)
        return HierarchicalByteEmbeddingConfig(
            byte_embedding_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            max_token_bytes=options.hierarchical_max_token_bytes,
            byte_position_config=byte_position_config,
            encoder_config=encoder_config,
            projection_config=projection_config,
        )

    def _build_hierarchical_encoder_layer_config(
        self, byte_sequence_length: int
    ) -> TransformerEncoderLayerConfig:
        decoder_options = self.decoder_options
        attention_options = self.attention_options
        # The transformer's adaptive projection and feed-forward stacks, without
        # attention heads or feed-forward experts.
        attention_projection_base_stack_config = (
            self._build_attention_projection_base_stack_config()
        )
        projection_model_config = _isolate_token_rows(
            attention_projection_base_stack_config
        )
        feed_forward_base_stack_config = super()._build_feed_forward_base_stack_config()
        feed_forward_stack_config = _isolate_token_rows(
            cast(LayerStackConfig, feed_forward_base_stack_config)
        )
        residual_config = AdditiveResidualConfig()
        attention_config = SelfAttentionConfig(
            # Bounds the largest equal-length token group in one batch.
            batch_size=self.batch_size * self.sequence_length,
            num_heads=attention_options.num_heads,
            embedding_dim=self.hidden_dim,
            query_key_projection_dim=self.hidden_dim,
            value_projection_dim=self.hidden_dim,
            target_sequence_length=byte_sequence_length,
            source_sequence_length=byte_sequence_length,
            target_dtype=torch.float32,
            dropout_probability=decoder_options.dropout_probability,
            zero_attention_flag=False,
            causal_attention_mask_flag=False,
            add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
            average_attention_weights_flag=False,
            return_attention_weights_flag=False,
            batch_first_flag=True,
            projection_model_config=projection_model_config,
            projection_strategy=SelfAttentionProjectionStrategy.SEPARATE,
        )
        feed_forward_config = FeedForwardConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            stack_config=feed_forward_stack_config,
        )
        return TransformerEncoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=decoder_options.layer_norm_position,
            normalization=decoder_options.normalization,
            dropout_probability=decoder_options.dropout_probability,
            residual_config=residual_config,
            attention_config=attention_config,
            feed_forward_config=feed_forward_config,
        )

    def _build_contextual_embedding_config(self) -> ByteContextualEmbeddingConfig:
        options = self.embedding_options
        kernel_dim = options.kernel_dim
        byte_moe_config = self._build_embedding_moe_config(options.max_token_bytes * 9)
        positional_embedding_config = self._build_positional_embedding_config()
        relative_position_config = DynamicPositionalBiasConfig(
            num_heads=1,
            embedding_dim=kernel_dim,
            max_positions=max(1, self.sequence_length - 1),
        )
        prefix_kernel_config = CausalPrefixKernelConfig(
            hidden_dim=self.hidden_dim,
            kernel_dim=kernel_dim,
            relative_position_config=relative_position_config,
        )
        context_moe_config = self._build_embedding_moe_config(self.hidden_dim * 2)
        return ByteContextualEmbeddingConfig(
            max_token_bytes=options.max_token_bytes,
            hidden_dim=self.hidden_dim,
            byte_moe_config=byte_moe_config,
            positional_embedding_config=positional_embedding_config,
            prefix_kernel_config=prefix_kernel_config,
            context_moe_config=context_moe_config,
            residual_scale_initial_value=options.residual_scale_initial_value,
        )

    def _build_embedding_moe_config(self, input_dim: int) -> MixtureOfExpertsConfig:
        # Copy the transformer's expert and router architecture, not their weights.
        stack_options = self._attention_experts_stack_options()
        model_config = self._build_expert_model_config(
            stack_options, self.grouping_config
        )
        mixture = copy.deepcopy(
            model_config.stack_config.layer_config.layer_model_config
        )
        mixture.input_dim = input_dim
        mixture.output_dim = self.hidden_dim
        # These are the contextual component's causal/batch-isolation constraints.
        mixture.capacity_factor = 0.0
        mixture.routing_initialization_mode = RoutingInitializationMode.LAYER
        mixture.compute_expert_mixture_flag = True
        mixture.weighted_parameters_flag = True
        mixture.weighting_position_option = ExpertWeightingPositionOptions.AFTER_EXPERTS
        mixture.sampler_config.normalize_probabilities_flag = True
        mixture.sampler_config.router_config.input_dim = input_dim
        return mixture

    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        self._configure_runtime(runtime)

    def _configure_runtime(self, runtime: RuntimeOptions) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.gpt.expert_linear_adaptive GptExpertLinearAdaptiveConfigBuilder runtime must be RuntimeOptions"
            )
        self._runtime = runtime
        options = runtime.construction_options(config, group_defaults=DEFAULT_RUNTIME)
        self.adaptive_generator_stack_options = options.adaptive_generator_stack_options
        self.hidden_adaptive_weight_options = options.hidden_adaptive_weight_options
        self.grouping_config = options.grouping_config
        self.attention_grouping_config = options.attention_grouping_config
        self.feed_forward_grouping_config = options.feed_forward_grouping_config
        self.hidden_adaptive_bias_options = options.hidden_adaptive_bias_options
        self.hidden_adaptive_diagonal_options = options.hidden_adaptive_diagonal_options
        self.hidden_adaptive_mask_options = options.hidden_adaptive_mask_options
        self.router_layer_controller_options = options.router_layer_controller_options
        self.router_dynamic_memory_options = options.router_dynamic_memory_options
        self.router_recurrent_controller_options = (
            options.router_recurrent_controller_options
        )
        self.router_adaptive_weight_options = options.router_adaptive_weight_options
        self.router_grouping_config = options.router_grouping_config
        self.router_adaptive_bias_options = options.router_adaptive_bias_options
        self.router_adaptive_diagonal_options = options.router_adaptive_diagonal_options
        self.router_adaptive_mask_options = options.router_adaptive_mask_options
        self.mixture_submodule_stack_options = options.mixture_submodule_stack_options
        self.mixture_layer_controller_options = options.mixture_layer_controller_options
        self.mixture_dynamic_memory_options = options.mixture_dynamic_memory_options
        self.mixture_recurrent_controller_options = (
            options.mixture_recurrent_controller_options
        )
        self.mixture_options = options.mixture_options
        self.expert_stack_options = options.expert_stack_options
        self.sampler_options = options.sampler_options
        self.router_options = options.router_options
        self.router_stack_options = options.router_stack_options
        self.expert_layer_controller_options = options.expert_layer_controller_options
        self.expert_dynamic_memory_options = options.expert_dynamic_memory_options
        self.expert_recurrent_controller_options = (
            options.expert_recurrent_controller_options
        )
        self.expert_attention_use_kv_expert_models_flag = (
            options.expert_attention_use_kv_expert_models_flag
        )
        super().__init__(options, experiment_config_type=ExperimentConfig)

    def _build_feed_forward_base_stack_config(self) -> MixtureOfExpertsModelConfig:
        stack_options = self._feed_forward_experts_stack_options()
        feed_forward_base_stack_config = self._build_expert_model_config(
            stack_options, self.feed_forward_grouping_config
        )
        return feed_forward_base_stack_config

    def _build_attention_config(self):
        decoder_options = self.decoder_options
        attention_options = self.attention_options
        source_memory_flag = attention_options.source_memory_flag
        experts_config = self._build_attention_experts_config()
        attention_experts_config = build_soft_moe_attention_experts(
            experts_config, attention_options.attention_soft_moe_config
        )
        projection_model_config = self._build_attention_projection_stack_config()
        source_memory_config = _source_memory.build_source_memory_config(
            attention_options, attention_experts_config
        )
        return mixture_attention_heads(
            batch_size=self.batch_size,
            num_heads=attention_options.num_heads,
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length,
            dropout_probability=decoder_options.dropout_probability,
            causal_attention_mask_flag=True,
            add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
            projection_model_config=cast(
                LayerStackConfig | RecurrentCompositionConfig, projection_model_config
            ),
            experts_config=attention_experts_config,
            use_kv_expert_models_flag=self.expert_attention_use_kv_expert_models_flag,
            source_memory_flag=source_memory_flag,
            source_memory_config=source_memory_config,
        )

    def _build_attention_experts_config(self):
        stack_options = self._attention_experts_stack_options()
        model_config = self._build_expert_model_config(
            stack_options, self.grouping_config
        )
        return model_config.stack_config.layer_config.layer_model_config

    def _feed_forward_experts_stack_options(self) -> ExpertsStackOptions:
        feed_forward_stack_options = self._effective_feed_forward_stack_options()
        return feed_forward_expert_stack(
            ExpertsStackOptions, feed_forward_stack_options
        )

    def _attention_experts_stack_options(self) -> ExpertsStackOptions:
        return attention_expert_stack(
            ExpertsStackOptions,
            self.hidden_dim,
            self.decoder_options,
            self.feed_forward_options,
            self.expert_stack_options,
        )

    def _build_linear_layer_config(
        self,
        *,
        bias_flag: bool,
    ) -> AdaptiveLinearLayerConfig:
        adaptive_bias_enabled = self.hidden_adaptive_bias_options.option_flag
        stack_options = self._attention_experts_stack_options()
        control_factory = self._control_config_factory(
            stack_options, self.attention_grouping_config
        )
        linear_layer_config = control_factory.build_hidden_adaptive_linear_layer_config(
            bias_flag or adaptive_bias_enabled
        )
        return linear_layer_config

    def _build_expert_model_config(
        self,
        stack_options: ExpertsStackOptions,
        grouping_config: GroupingConfig | None,
    ) -> MixtureOfExpertsModelConfig:
        model_config = self._control_config_factory(
            stack_options, grouping_config
        ).build()
        if isinstance(model_config, MixtureOfExpertsModelConfig):
            return model_config
        return model_config.block_config

    def _control_config_factory(
        self,
        stack_options: ExpertsStackOptions,
        grouping_config: GroupingConfig | None,
    ) -> ControlConfigFactory:
        return adaptive_expert_controller(
            self,
            stack_options,
            submodule_stack_options=self.submodule_stack_options,
            layer_controller_options=self.layer_controller_options,
            dynamic_memory_options=self.dynamic_memory_options,
            recurrent_controller_options=self.recurrent_controller_options,
            grouping_config=grouping_config,
            inputs=ControlConfigDependencies,
            factory=ControlConfigFactory,
        )
