from collections.abc import Mapping
from importlib import import_module
from typing import cast
from unittest.mock import patch

import pytest

from models.catalog import model_package

_RUNTIME_CASES = (
    (
        "models.linears.linear.runtime_defaults",
        "models.linears.linear.config",
    ),
    (
        "models.linears.linear_adaptive.runtime_defaults",
        "models.linears.linear_adaptive.config",
    ),
    (
        "models.neuron.linear._hidden.runtime_defaults",
        "models.neuron.linear.config",
    ),
    (
        "models.neuron.linear_adaptive._hidden.runtime_defaults",
        "models.neuron.linear_adaptive.config",
    ),
)

_NEURON_VARIANTS = (
    "linear",
    "linear_adaptive",
    "expert_linear",
    "expert_linear_adaptive",
)


@pytest.mark.parametrize(("runtime_module_name", "config_module_name"), _RUNTIME_CASES)
def test_runtime_defaults_remain_frozen_after_config_mutation(
    runtime_module_name: str,
    config_module_name: str,
) -> None:
    runtime_module = import_module(runtime_module_name)
    config_module = import_module(config_module_name)
    frozen_batch_size = runtime_module.DEFAULT_RUNTIME.batch_size

    with patch.object(config_module, "BATCH_SIZE", frozen_batch_size + 100):
        resolved = runtime_module.runtime_from_flat({})

    assert resolved.batch_size == frozen_batch_size


@pytest.mark.parametrize(
    ("runtime_module_name", "error_type", "message"),
    (
        (
            "models.linears.linear.runtime_defaults",
            ValueError,
            "models.linears.linear: 'batch_size' must be positive; got 0",
        ),
        (
            "models.linears.linear_adaptive.runtime_defaults",
            TypeError,
            "models.linears.linear_adaptive: runtime key 'stack_bias_flag' "
            "has type int; expected bool",
        ),
        (
            "models.neuron.linear._hidden.runtime_defaults",
            ValueError,
            "models.neuron.linear._hidden: 'batch_size' must be positive; got 0",
        ),
        (
            "models.neuron.linear_adaptive._hidden.runtime_defaults",
            TypeError,
            "models.neuron.linear_adaptive._hidden: runtime key "
            "'stack_bias_flag' has type int; expected bool",
        ),
    ),
)
def test_runtime_default_validation_keeps_first_error_precedence(
    runtime_module_name: str,
    error_type: type[Exception],
    message: str,
) -> None:
    runtime_from_flat = import_module(runtime_module_name).runtime_from_flat

    with pytest.raises(error_type) as raised:
        runtime_from_flat({"batch_size": 0, "stack_bias_flag": 1})

    assert str(raised.value) == message


@pytest.mark.parametrize("variant", _NEURON_VARIANTS)
def test_neuron_outer_defaults_remain_frozen_after_config_mutation(
    variant: str,
) -> None:
    config_module = import_module(f"models.neuron.{variant}.config")
    package = model_package(f"neuron/{variant}")
    assert package is not None
    frozen_configuration = package.build_configuration()
    frozen_max_steps = (
        frozen_configuration.experiment_config.neuron_cluster_config.max_steps
    )

    with patch.object(config_module, "CLUSTER_MAX_STEPS", frozen_max_steps + 100):
        resolved_configuration = package.build_configuration()

    assert (
        resolved_configuration.experiment_config.neuron_cluster_config.max_steps
        == frozen_max_steps
    )


@pytest.mark.parametrize("variant", _NEURON_VARIANTS)
def test_neuron_runtime_defaults_report_the_unknown_option(variant: str) -> None:
    runtime_from_flat = import_module(
        f"models.neuron.{variant}.runtime_defaults"
    ).runtime_from_flat

    with pytest.raises(
        (TypeError, ValueError), match="unknown_runtime_field"
    ) as raised:
        runtime_from_flat({"unknown_runtime_field": object()})

    assert "__init__" not in str(raised.value)


@pytest.mark.parametrize("variant", _NEURON_VARIANTS)
def test_neuron_outer_runtime_defaults_keep_non_string_key_errors(
    variant: str,
) -> None:
    runtime_from_flat = import_module(
        f"models.neuron.{variant}.runtime_defaults"
    ).runtime_from_flat
    invalid_values = cast(Mapping[str, object], {1: object()})

    with pytest.raises(TypeError) as raised:
        runtime_from_flat(invalid_values)

    assert (
        str(raised.value)
        == f"models.neuron.{variant}: Runtime Defaults keys must be strings"
    )


@pytest.mark.parametrize(
    "module_name",
    ("models.experts.linear_adaptive", "models.neuron.expert_linear_adaptive._hidden"),
)
def test_adaptive_expert_runtime_checks_structured_and_flat_fields(module_name):
    defaults = import_module(f"{module_name}.runtime_defaults")
    stack_options = defaults.DEFAULT_RUNTIME.stack_options
    provided = {"stack_options": stack_options, "batch_size": 3}
    runtime = defaults.runtime_from_flat(provided)
    assert runtime.stack_options == stack_options
    assert runtime.batch_size == 3
    assert provided == {"stack_options": stack_options, "batch_size": 3}

    with pytest.raises(
        TypeError, match="runtime key 'stack_options'.*ExpertsStackOptions"
    ):
        defaults.runtime_from_flat({"stack_options": object()})
    with pytest.raises(TypeError, match="runtime key 'batch_size'.*int"):
        defaults.runtime_from_flat({"stack_options": stack_options, "batch_size": "3"})
    with pytest.raises(TypeError, match="unknown runtime option.*'unrecognized_stack'"):
        defaults.runtime_from_flat({"unrecognized_stack": object()})
