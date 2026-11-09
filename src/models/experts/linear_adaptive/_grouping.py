"""Package-owned scalar grouping resolution."""

from dataclasses import dataclass

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    GroupingConfig,
    SummaryNormalizationOptions,
)
from emperor.config import ConfigBase
from models import _learned_grouping
from models.experts.linear_adaptive import config


@dataclass(frozen=True, kw_only=True)
class GroupingDefaultValues:
    grouping_scope: AdaptiveParameterGroupingScopeOptions | None = config.GROUPING_SCOPE
    group_count: int | None = config.GROUP_COUNT
    chunk_size: int | None = config.CHUNK_SIZE
    grouping_sequence_length: int | None = config.GROUPING_SEQUENCE_LENGTH
    grouping_input_order: AdaptiveParameterInputOrderOptions | None = (
        config.GROUPING_INPUT_ORDER
    )
    grouping_method: type[GroupingConfig] | None = config.GROUPING_METHOD
    grouping_summary_normalization: SummaryNormalizationOptions | None = (
        config.GROUPING_SUMMARY_NORMALIZATION
    )
    grouping_model_config: ConfigBase | None = config.GROUPING_MODEL_CONFIG
    grouping_model_option: _learned_grouping.GroupingModelOptions | None = None
    grouping_model_bias_flag: bool | None = None
    grouping_attention_hidden_dim: int | None = config.GROUPING_ATTENTION_HIDDEN_DIM
    grouping_rms_norm_epsilon: float | None = config.GROUPING_RMS_NORM_EPSILON
    router_grouping_scope: AdaptiveParameterGroupingScopeOptions | None = (
        config.ROUTER_GROUPING_SCOPE
    )
    router_group_count: int | None = config.ROUTER_GROUP_COUNT
    router_chunk_size: int | None = config.ROUTER_CHUNK_SIZE
    router_grouping_sequence_length: int | None = config.ROUTER_GROUPING_SEQUENCE_LENGTH
    router_grouping_input_order: AdaptiveParameterInputOrderOptions | None = (
        config.ROUTER_GROUPING_INPUT_ORDER
    )
    router_grouping_method: type[GroupingConfig] | None = config.ROUTER_GROUPING_METHOD
    router_grouping_summary_normalization: SummaryNormalizationOptions | None = (
        config.ROUTER_GROUPING_SUMMARY_NORMALIZATION
    )
    router_grouping_model_config: ConfigBase | None = (
        config.ROUTER_GROUPING_MODEL_CONFIG
    )
    router_grouping_model_option: _learned_grouping.GroupingModelOptions | None = None
    router_grouping_model_bias_flag: bool | None = None
    router_grouping_attention_hidden_dim: int | None = (
        config.ROUTER_GROUPING_ATTENTION_HIDDEN_DIM
    )
    router_grouping_rms_norm_epsilon: float | None = (
        config.ROUTER_GROUPING_RMS_NORM_EPSILON
    )
    input_layer_grouping_scope: AdaptiveParameterGroupingScopeOptions | None = (
        config.INPUT_LAYER_GROUPING_SCOPE
    )
    input_layer_group_count: int | None = config.INPUT_LAYER_GROUP_COUNT
    input_layer_chunk_size: int | None = config.INPUT_LAYER_CHUNK_SIZE
    input_layer_grouping_sequence_length: int | None = (
        config.INPUT_LAYER_GROUPING_SEQUENCE_LENGTH
    )
    input_layer_grouping_input_order: AdaptiveParameterInputOrderOptions | None = (
        config.INPUT_LAYER_GROUPING_INPUT_ORDER
    )
    input_layer_grouping_method: type[GroupingConfig] | None = (
        config.INPUT_LAYER_GROUPING_METHOD
    )
    input_layer_grouping_summary_normalization: SummaryNormalizationOptions | None = (
        config.INPUT_LAYER_GROUPING_SUMMARY_NORMALIZATION
    )
    input_layer_grouping_model_config: ConfigBase | None = (
        config.INPUT_LAYER_GROUPING_MODEL_CONFIG
    )
    input_layer_grouping_model_option: _learned_grouping.GroupingModelOptions | None = (
        None
    )
    input_layer_grouping_model_bias_flag: bool | None = None
    input_layer_grouping_attention_hidden_dim: int | None = (
        config.INPUT_LAYER_GROUPING_ATTENTION_HIDDEN_DIM
    )
    input_layer_grouping_rms_norm_epsilon: float | None = (
        config.INPUT_LAYER_GROUPING_RMS_NORM_EPSILON
    )
    output_layer_grouping_scope: AdaptiveParameterGroupingScopeOptions | None = (
        config.OUTPUT_LAYER_GROUPING_SCOPE
    )
    output_layer_group_count: int | None = config.OUTPUT_LAYER_GROUP_COUNT
    output_layer_chunk_size: int | None = config.OUTPUT_LAYER_CHUNK_SIZE
    output_layer_grouping_sequence_length: int | None = (
        config.OUTPUT_LAYER_GROUPING_SEQUENCE_LENGTH
    )
    output_layer_grouping_input_order: AdaptiveParameterInputOrderOptions | None = (
        config.OUTPUT_LAYER_GROUPING_INPUT_ORDER
    )
    output_layer_grouping_method: type[GroupingConfig] | None = (
        config.OUTPUT_LAYER_GROUPING_METHOD
    )
    output_layer_grouping_summary_normalization: SummaryNormalizationOptions | None = (
        config.OUTPUT_LAYER_GROUPING_SUMMARY_NORMALIZATION
    )
    output_layer_grouping_model_config: ConfigBase | None = (
        config.OUTPUT_LAYER_GROUPING_MODEL_CONFIG
    )
    output_layer_grouping_model_option: (
        _learned_grouping.GroupingModelOptions | None
    ) = None
    output_layer_grouping_model_bias_flag: bool | None = None
    output_layer_grouping_attention_hidden_dim: int | None = (
        config.OUTPUT_LAYER_GROUPING_ATTENTION_HIDDEN_DIM
    )
    output_layer_grouping_rms_norm_epsilon: float | None = (
        config.OUTPUT_LAYER_GROUPING_RMS_NORM_EPSILON
    )
