from collections.abc import Callable, Mapping
from dataclasses import replace
from types import ModuleType
from typing import Any, Generic, TypeVar

from models._adaptive_generator_stack import GeneratorStackSource

_WeightOptions = TypeVar("_WeightOptions")
_BiasOptions = TypeVar("_BiasOptions")


def generation_keys(config_module: ModuleType) -> set[str]:
    prefixes = (
        "weight_input_factor_",
        "weight_output_factor_",
        "weight_coefficient_",
        "weight_mixture_",
        "bias_mixture_",
    )
    return {
        name.lower()
        for name in vars(config_module)
        if name.isupper() and any(part in name.lower() for part in prefixes)
    }


class GenerationSettingsReader(Generic[_WeightOptions, _BiasOptions]):
    stack_source_type: Callable[..., GeneratorStackSource]
    weight_options_type: Callable[..., _WeightOptions]
    bias_options_type: Callable[..., _BiasOptions]

    def __init__(
        self, values: Mapping[str, Any], config_module: ModuleType, prefix: str
    ) -> None:
        self.values = values
        self.config_module = config_module
        self.prefix = prefix

    def weight_options(self) -> _WeightOptions:
        return self.weight_options_type(
            input_factor_source=self.__value("weight_input_factor_source"),
            output_factor_source=self.__value("weight_output_factor_source"),
            mixture_num_experts=self.__value("weight_mixture_num_experts"),
            mixture_top_k=self.__value("weight_mixture_top_k"),
            mixture_normalize_probabilities_flag=self.__value(
                "weight_mixture_normalize_probabilities_flag"
            ),
            input_factor_generator_stack=self.__stack_source(
                "weight_input_factor_generator_stack"
            ),
            output_factor_generator_stack=self.__stack_source(
                "weight_output_factor_generator_stack"
            ),
            coefficient_generator_stack=self.__stack_source(
                "weight_coefficient_generator_stack"
            ),
            mixture_router_generator_stack=self.__stack_source(
                "weight_mixture_router_generator_stack"
            ),
        )

    def bias_options(self) -> _BiasOptions:
        return self.bias_options_type(
            mixture_num_experts=self.__value("bias_mixture_num_experts"),
            mixture_top_k=self.__value("bias_mixture_top_k"),
            mixture_normalize_probabilities_flag=self.__value(
                "bias_mixture_normalize_probabilities_flag"
            ),
            mixture_router_generator_stack=self.__stack_source(
                "bias_mixture_router_generator_stack"
            ),
        )

    def __stack_source(self, prefix: str) -> GeneratorStackSource:
        return self.stack_source_type(
            independent_flag=self.__value(f"{prefix}_independent_flag"),
            hidden_dim=self.__value(f"{prefix}_hidden_dim"),
            layer_norm_position=self.__value(f"{prefix}_layer_norm_position"),
            normalization=self.__value(f"{prefix}_normalization"),
            num_layers=self.__value(f"{prefix}_num_layers"),
            activation=self.__value(f"{prefix}_activation"),
            residual_connection_option=self.__value(
                f"{prefix}_residual_connection_option"
            ),
            residual_block_size=self.__value(f"{prefix}_residual_block_size"),
            residual_rms_norm_epsilon=self.__value(
                f"{prefix}_residual_rms_norm_epsilon"
            ),
            residual_model_flag=self.__value(f"{prefix}_residual_model_flag"),
            dropout_probability=self.__value(f"{prefix}_dropout_probability"),
            last_layer_bias_option=self.__value(f"{prefix}_last_layer_bias_option"),
            apply_output_postprocessing_flag=self.__value(
                f"{prefix}_apply_output_postprocessing_flag"
            ),
            bias_flag=self.__value(f"{prefix}_bias_flag"),
        )

    def __value(self, name: str) -> Any:
        key = self.prefix + name
        if key in self.values:
            return self.values[key]
        if self.prefix in ("attn_", "ff_") and name in self.values:
            return self.values[name]
        return getattr(self.config_module, key.upper(), None)

    @classmethod
    def apply(
        cls,
        options: dict[str, Any],
        values: Mapping[str, Any],
        config_module: ModuleType,
    ) -> dict[str, Any]:
        result = dict(options)
        for role, prefix in (
            ("hidden_adaptive", ""),
            ("attention_hidden_adaptive", "attn_"),
            ("feed_forward_hidden_adaptive", "ff_"),
            ("router_adaptive", "router_"),
        ):
            settings = cls(values, config_module, prefix)
            weight_key = role + "_weight_options"
            bias_key = role + "_bias_options"
            if result.get(weight_key) is not None:
                result[weight_key] = replace(
                    result[weight_key],
                    generation=settings.weight_options(),
                )
            if result.get(bias_key) is not None:
                result[bias_key] = replace(
                    result[bias_key],
                    generation=settings.bias_options(),
                )
        for role, prefix in (
            ("input_boundary_options", "input_layer_"),
            ("output_boundary_options", "output_layer_"),
        ):
            if result.get(role) is not None:
                settings = cls(values, config_module, prefix)
                result[role] = replace(
                    result[role],
                    weight_generation=settings.weight_options(),
                    bias_generation=settings.bias_options(),
                )
        return result
