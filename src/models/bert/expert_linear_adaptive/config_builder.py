from typing import cast

import models.bert.expert_linear_adaptive.config as config
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    GroupingConfig,
)
from emperor.experts import MixtureOfExpertsModelConfig
from emperor.layers import LayerStackConfig, RecurrentCompositionConfig
from emperor.transformer._soft_moe_config import build_soft_moe_attention_experts
from models import _attention_settings
from models import _attention_source_memory as _source_memory
from models._attention_experts import (
    adaptive_expert_controller,
    attention_expert_stack,
    feed_forward_expert_stack,
    mixture_attention_heads,
)
from models.bert.expert_linear_adaptive._base_config_builder import (
    BertBackendConfigBuilder,
)
from models.bert.expert_linear_adaptive._expert_control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.bert.expert_linear_adaptive.experiment_config import ExperimentConfig
from models.bert.expert_linear_adaptive.runtime_defaults import DEFAULT_RUNTIME
from models.bert.expert_linear_adaptive.runtime_options import (
    ExpertsStackOptions,
    RuntimeOptions,
)


class BertExpertLinearAdaptiveConfigBuilder(
    _attention_settings.RuntimeBuilder[RuntimeOptions], BertBackendConfigBuilder
):
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        self._configure_runtime(runtime)

    def _configure_runtime(self, runtime: RuntimeOptions) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.bert.expert_linear_adaptive BertExpertLinearAdaptiveConfigBuilder runtime must be RuntimeOptions"
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
        encoder_options = self.encoder_options
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
            dropout_probability=encoder_options.dropout_probability,
            causal_attention_mask_flag=encoder_options.causal_attention_mask_flag,
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
            self.encoder_options,
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
