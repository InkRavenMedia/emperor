from dataclasses import fields
from types import ModuleType
from typing import Any, Final

import models.neuron.expert_linear.config as config
from models._expert_flat_options import ExpertFlatOptions
from models.neuron.expert_linear._hidden import runtime_options as options
from models.neuron.expert_linear._hidden._control_defaults import (
    expert_control_defaults,
    main_control_defaults,
    router_stack_defaults,
)
from models.neuron.expert_linear._residual import (
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


def runtime_from_builder_options(
    builder_options: dict[str, Any],
    config_module: ModuleType = config,
) -> options.RuntimeOptions:
    values = {
        "batch_size": config_module.BATCH_SIZE,
        "learning_rate": config_module.LEARNING_RATE,
        "input_dim": config_module.INPUT_DIM,
        "hidden_dim": config_module.HIDDEN_DIM,
        "output_dim": config_module.OUTPUT_DIM,
        **builder_kwargs_from_flat({}, config_module),
        **builder_options,
    }
    unknown = set(values) - _ExpertOptions.runtime_fields
    if unknown:
        name = sorted(unknown)[0]
        raise TypeError(f"unexpected hidden runtime option {name!r}")
    return options.RuntimeOptions(**values)


def runtime_from_flat(
    flat_options: dict[str, Any] | None = None,
    config_module: ModuleType = config,
) -> options.RuntimeOptions:
    return runtime_from_builder_options(
        builder_kwargs_from_flat(flat_options or {}, config_module),
        config_module,
    )


DEFAULT_RUNTIME: Final[options.RuntimeOptions] = runtime_from_flat()
