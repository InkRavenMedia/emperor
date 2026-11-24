from dataclasses import dataclass, fields, replace
from types import ModuleType
from typing import Any, TypeVar, cast, get_type_hints

from emperor.config import ConfigBase
from emperor.layers import (
    HierarchicalReasoningModelRecurrentConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
    RecurrentCompositionConfig,
    RecurrentLayerConfig,
    ResidualConfig,
    TinyRecursiveModelRecurrentConfig,
)
from model_runtime.packages.runtime_values import validate_runtime_default_value_types
from models._residual_stack import build_residual_config

_Block = TypeVar("_Block", bound=ConfigBase)


@dataclass(frozen=True, kw_only=True)
class RecurrentVariantOptions:
    recurrent_composition_option: type[RecurrentCompositionConfig] = (
        RecurrentLayerConfig
    )
    recurrent_latent_updates_per_answer_update: int = 2
    recurrent_answer_update_count: int = 2
    recurrent_high_cycles: int = 2
    recurrent_low_cycles: int = 2
    recurrent_initialization_standard_deviation: float = 1.0

    def configure_stack(self, stack: _Block) -> _Block | RecurrentCompositionConfig:
        option = self.recurrent_composition_option
        if option is RecurrentLayerConfig:
            return stack
        if not isinstance(stack, RecurrentLayerConfig):
            raise ValueError(
                "Selecting a recurrent composition requires recurrent_flag=True."
            )
        return self.build_composition(stack)

    def build_composition(
        self, stack: RecurrentLayerConfig
    ) -> RecurrentCompositionConfig:
        option = cast(object, self.recurrent_composition_option)
        if option is RecurrentLayerConfig:
            return stack
        if not isinstance(option, type) or not issubclass(
            option, RecurrentCompositionConfig
        ):
            raise TypeError(
                "recurrent_composition_option must be a concrete RecurrentCompositionConfig subclass"
            )
        values = {
            field.name: getattr(stack, field.name)
            for field in fields(RecurrentLayerConfig)
        }
        values.update(
            high_block_config=stack.block_config,
            low_block_config=stack.block_config,
            initialization_standard_deviation=self.recurrent_initialization_standard_deviation,
            latent_updates_per_answer_update=self.recurrent_latent_updates_per_answer_update,
            answer_update_count=self.recurrent_answer_update_count,
            high_cycles=self.recurrent_high_cycles,
            low_cycles=self.recurrent_low_cycles,
        )
        composition = option(
            **{
                field.name: values[field.name]
                for field in fields(option)
                if field.name in values
            }
        )
        composition.registry_owner()
        return composition


@dataclass(frozen=True, kw_only=True)
class RecurrentResidualOptions(RecurrentVariantOptions):
    recurrent_residual_connection_option: type[ResidualConfig] | None = None
    recurrent_residual_block_size: int | None = None
    recurrent_residual_rms_norm_epsilon: float | None = None

    def configure_stack(self, stack: _Block) -> _Block | RecurrentCompositionConfig:
        if self.recurrent_residual_connection_option is not None:
            if not isinstance(stack, RecurrentLayerConfig):
                raise ValueError(
                    "Selecting a recurrent residual requires recurrent_flag=True."
                )
            configured = replace(
                stack,
                residual_config=build_residual_config(
                    self.recurrent_residual_connection_option,
                    False,
                    residual_block_size=self.recurrent_residual_block_size,
                    residual_rms_norm_epsilon=self.recurrent_residual_rms_norm_epsilon,
                ),
            )
            return super().configure_stack(configured)
        return super().configure_stack(stack)


@dataclass(frozen=True, kw_only=True)
class ParametricRecurrentOptions(RecurrentResidualOptions):
    recurrent_flag: bool = False
    recurrent_max_steps: int = 2
    recurrent_initial_iterations: int = 2
    recurrent_iteration_increment: int = 1
    recurrent_forward_calls_before_iteration_increment: int = 1
    recurrent_smooth_iteration_growth_flag: bool = False
    recurrent_no_gradient_transition_count: int | None = None
    recurrent_gradient_transition_count: int | None = None
    recurrent_layer_norm_position: LayerNormPositionOptions = (
        LayerNormPositionOptions.BEFORE
    )
    recurrent_normalization: NormalizationOptions = NormalizationOptions.RMS_NORM

    def configure_stack(self, stack: _Block) -> _Block | RecurrentCompositionConfig:
        if self.recurrent_flag:
            configured = RecurrentLayerConfig(
                block_config=stack,
                max_steps=self.recurrent_max_steps,
                initial_iterations=self.recurrent_initial_iterations,
                iteration_increment=self.recurrent_iteration_increment,
                forward_calls_before_iteration_increment=self.recurrent_forward_calls_before_iteration_increment,
                smooth_iteration_growth_flag=self.recurrent_smooth_iteration_growth_flag,
                gradient_transition_count=self.recurrent_gradient_transition_count,
                no_gradient_transition_count=self.recurrent_no_gradient_transition_count,
                recurrent_layer_norm_position=self.recurrent_layer_norm_position,
                recurrent_normalization=self.recurrent_normalization,
            )
            return super().configure_stack(configured)
        return super().configure_stack(stack)


def declare_variant_options(
    namespace: dict[str, Any],
    options_type: type[RecurrentVariantOptions] = RecurrentVariantOptions,
) -> None:
    namespace["_RECURRENT_VARIANT_OPTIONS_TYPE"] = options_type
    aliases = namespace.setdefault("_CONFIG_FIELD_METADATA_ALIASES", {})
    hints = get_type_hints(options_type)
    for field in fields(options_type):
        key = field.name.upper()
        if key not in namespace:
            namespace[key] = field.default
            namespace["__annotations__"][key] = hints[field.name]
            aliases[key] = "RECURRENT_FLAG"
    for option in (
        RecurrentLayerConfig,
        TinyRecursiveModelRecurrentConfig,
        HierarchicalReasoningModelRecurrentConfig,
    ):
        namespace[option.__name__] = option


def variant_updates(
    values: dict[str, Any], source: ModuleType, base: RecurrentVariantOptions | None
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    options_type: type[RecurrentVariantOptions] = source._RECURRENT_VARIANT_OPTIONS_TYPE
    for field in fields(options_type):
        key = field.name
        default = getattr(base, key, getattr(source, key.upper()))
        updates[key] = values.pop(key, default)
    validate_runtime_default_value_types(
        updates, package=source.__name__, config_module=source
    )
    return updates


DEFAULT_VARIANT_OPTIONS = RecurrentVariantOptions()
