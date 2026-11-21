# ruff: noqa: E501
from collections.abc import Mapping
from dataclasses import fields, replace
from types import ModuleType
from typing import Final

import models.neuron.expert_linear_adaptive.config as config
from model_runtime.packages._runtime_default_fields import apply_runtime_default_fields
from model_runtime.packages.runtime_values import runtime_default_mapping
from models.neuron.expert_linear_adaptive._generation import (
    apply_generation_options,
    generation_keys,
)
from models.neuron.expert_linear_adaptive._grouping import (
    apply_grouping_options,
    grouping_keys,
)
from models.neuron.expert_linear_adaptive._hidden._config_implementation import (
    ResolvedRuntimeDefaults,
    RuntimeDefaultValues,
    resolve_runtime_defaults,
)
from models.neuron.expert_linear_adaptive._hidden.runtime_options import RuntimeOptions

_DEFAULT_VALUES: Final = RuntimeDefaultValues()


def _runtime_options_from_defaults(
    defaults: ResolvedRuntimeDefaults,
) -> RuntimeOptions:
    core = defaults.core
    control = defaults.control
    adaptive = defaults.adaptive
    return RuntimeOptions(
        batch_size=core.batch_size,
        learning_rate=core.learning_rate,
        input_dim=core.input_dim,
        output_dim=core.output_dim,
        stack_options=core.stack_options,
        submodule_stack_options=core.submodule_stack_options,
        mixture_options=core.mixture_options,
        expert_stack_options=core.expert_stack_options,
        sampler_options=core.sampler_options,
        router_options=control.router_options,
        router_stack_options=control.router_stack_options,
        router_layer_controller_options=control.router_layer_controller_options,
        router_dynamic_memory_options=control.router_dynamic_memory_options,
        router_recurrent_controller_options=(
            control.router_recurrent_controller_options
        ),
        layer_controller_options=control.layer_controller_options,
        dynamic_memory_options=control.dynamic_memory_options,
        expert_layer_controller_options=control.expert_layer_controller_options,
        expert_dynamic_memory_options=control.expert_dynamic_memory_options,
        expert_recurrent_controller_options=(
            control.expert_recurrent_controller_options
        ),
        adaptive_generator_stack_options=adaptive.adaptive_generator_stack_options,
        hidden_adaptive_weight_options=adaptive.hidden_adaptive_weight_options,
        hidden_adaptive_bias_options=adaptive.hidden_adaptive_bias_options,
        hidden_adaptive_diagonal_options=adaptive.hidden_adaptive_diagonal_options,
        hidden_adaptive_mask_options=adaptive.hidden_adaptive_mask_options,
        input_boundary_options=adaptive.input_boundary_options,
        output_boundary_options=adaptive.output_boundary_options,
        router_adaptive_weight_options=adaptive.router_adaptive_weight_options,
        router_adaptive_bias_options=adaptive.router_adaptive_bias_options,
        router_adaptive_diagonal_options=adaptive.router_adaptive_diagonal_options,
        router_adaptive_mask_options=adaptive.router_adaptive_mask_options,
        recurrent_controller_options=control.recurrent_controller_options,
    )


def runtime_from_flat(
    flat_kwargs: Mapping[str, object] | None = None,
    config_module: ModuleType = config,
) -> RuntimeOptions:
    package = "models.neuron.expert_linear_adaptive._hidden"
    generation_values = runtime_default_mapping(flat_kwargs, package=package)
    values = apply_runtime_default_fields(
        _DEFAULT_VALUES,
        generation_values,
        package=package,
        config_module=config_module,
        delegated=generation_keys(config_module) | grouping_keys(config_module),
    )
    runtime = _runtime_options_from_defaults(resolve_runtime_defaults(values))
    runtime_values = {
        item.name: getattr(runtime, item.name) for item in fields(runtime)
    }
    runtime_values = apply_generation_options(
        runtime_values, generation_values, config_module
    )
    return replace(
        runtime,
        **apply_grouping_options(runtime_values, generation_values, config_module),
    )


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_flat()


__all__ = ["DEFAULT_RUNTIME", "runtime_from_flat"]
