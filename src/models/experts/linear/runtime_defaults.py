from dataclasses import fields
from types import ModuleType
from typing import Any, Final

import models.experts.linear.config as config
from model_runtime.packages.runtime_values import validate_runtime_default_value_types
from models._expert_flat_options import ExpertFlatOptions
from models._recurrent_variants import variant_updates
from models._runtime_binding import bind_runtime_options
from models.experts.linear import runtime_options as options
from models.experts.linear._control_defaults import (
    expert_control_defaults,
    main_control_defaults,
    router_stack_defaults,
)
from models.experts.linear._residual import (
    ResidualStackSource,
    resolve_residual_stack_options,
)


class _ExpertOptions(ExpertFlatOptions):
    runtime_fields = {field.name for field in fields(options.RuntimeOptions)}
    stack_type = options.ExpertsStackOptions
    submodule_type = options.ExpertsSubmoduleStackOptions
    mixture_type = options.ExpertsMixtureOptions
    sampler_type = options.ExpertsSamplerOptions
    router_type = options.ExpertsRouterOptions
    residual_source_type = ResidualStackSource
    resolve_residual = staticmethod(resolve_residual_stack_options)
    main_control_defaults = staticmethod(main_control_defaults)
    expert_control_defaults = staticmethod(expert_control_defaults)
    router_stack_defaults = staticmethod(router_stack_defaults)


def builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> dict[str, Any]:
    return _ExpertOptions().builder_kwargs_from_flat(flat_kwargs, config_module)


@bind_runtime_options(config, variant_updates)
def runtime_from_flat(
    flat_kwargs: dict[str, Any] | None = None,
    config_module: ModuleType = config,
) -> options.RuntimeOptions:
    flat = dict(flat_kwargs or {})
    validate_runtime_default_value_types(
        flat,
        package="models.experts.linear",
        config_module=config_module,
    )
    hidden_dim = flat.get("hidden_dim", config_module.HIDDEN_DIM)
    translated = builder_kwargs_from_flat(flat, config_module)
    values = {
        "batch_size": translated.pop("batch_size", config_module.BATCH_SIZE),
        "learning_rate": translated.pop(
            "learning_rate",
            config_module.LEARNING_RATE,
        ),
        "input_dim": translated.pop("input_dim", config_module.INPUT_DIM),
        "hidden_dim": hidden_dim,
        "output_dim": translated.pop("output_dim", config_module.OUTPUT_DIM),
        **translated,
    }
    unknown = set(values) - _ExpertOptions.runtime_fields
    if unknown:
        name = sorted(unknown)[0]
        raise TypeError(
            "LinearConfigBuilder.__init__() got an unexpected keyword "
            f"argument {name!r}"
        )
    return options.RuntimeOptions(**values)


DEFAULT_RUNTIME: Final[options.RuntimeOptions] = runtime_from_flat()


__all__ = [
    "DEFAULT_RUNTIME",
    "builder_kwargs_from_flat",
    "runtime_from_flat",
]
