"""Every package declaring Soft MoE can configure and run it through CLI flags."""

import copy
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import ModuleType

import pytest
import torch

from emperor.experts import SoftMoEConfig
from emperor.experts._routing.soft import SoftMoERouter
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from model_runtime.inspection import configuration_schema
from model_runtime.packages.runtime_values import validate_runtime_default_values
from model_runtime.packages.soft_moe import declare_soft_moe_runtime_defaults
from models.catalog import discover_model_packages, model_package
from models.cli_selection import resolve_cli_selection
from models.experiment_cli_parser import get_experiment_parser
from tests.model_packages.test_memory_expert_parity import (
    PACKAGES,
    configuration,
    inputs_for,
    run_model,
    soft_routing,
)


def cli_values(package, arguments):
    parsed = get_experiment_parser(package).parse_args(
        ["--preset", "baseline", "--config", *arguments]
    )
    selection = resolve_cli_selection(
        parsed, package, package.preset_type, build_monitor_callbacks=False
    )
    return selection.config_overrides


def soft_configs(value):
    if isinstance(value, SoftMoEConfig):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from soft_configs(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from soft_configs(item)


def test_every_declared_soft_moe_component_has_cli_and_workbench_controls():
    for package in discover_model_packages():
        spec = package.runtime_defaults_spec
        component_keys = [
            key for key in spec.supported_keys if key.endswith("SOFT_MOE_CONFIG")
        ]
        if not component_keys:
            continue
        visible_keys = {field.key for field in configuration_schema(package).fields}
        for key in component_keys:
            prefix = key.removesuffix("_CONFIG")
            expected = {
                f"{prefix}_{suffix}"
                for suffix in (
                    "FLAG",
                    "NUM_EXPERTS",
                    "SLOTS_PER_EXPERT",
                    "SCORING_HIDDEN_DIM",
                    "SCORING_NUM_LAYERS",
                    "SCORING_ACTIVATION",
                    "SCORING_DROPOUT_PROBABILITY",
                    "SCORING_LAYER_NORM_POSITION",
                    "SCORING_NORMALIZATION",
                    "SCORING_LAST_LAYER_BIAS_OPTION",
                    "SCORING_RESIDUAL_CONNECTION_OPTION",
                    "SCORING_RESIDUAL_MODEL_FLAG",
                )
            }
            assert expected <= set(spec.supported_keys), (
                package.catalog_key,
                expected - set(spec.supported_keys),
            )
            assert expected <= visible_keys, (
                package.catalog_key,
                expected - visible_keys,
            )
            runtime = package.bind_runtime_defaults(
                {
                    f"{prefix.lower()}_flag": True,
                    f"{prefix.lower()}_num_experts": 2,
                    f"{prefix.lower()}_slots_per_expert": 4,
                }
            )
            assert any(
                config.num_experts == 2 and config.slots_per_expert == 4
                for config in soft_configs(runtime)
            ), (package.catalog_key, prefix)


@pytest.mark.parametrize("name", PACKAGES)
def test_cli_soft_moe_and_memory_forward_backward_across_packages(name):
    package = model_package(name)
    arguments = [
        "--soft-moe-flag",
        "true",
        "--soft-moe-num-experts",
        "2",
        "--soft-moe-slots-per-expert",
        "2",
        "--soft-moe-scoring-hidden-dim",
        "8",
        "--soft-moe-scoring-num-layers",
        "2",
        "--soft-moe-scoring-activation",
        "GELU",
        "--soft-moe-scoring-layer-norm-position",
        "BEFORE",
        "--soft-moe-scoring-dropout-probability",
        "0.1",
    ]
    role = None
    if name.startswith("mlp_mixer/"):
        role = "token-mixer"
    elif "expert" in name:
        role = "attention"
    if role is not None:
        arguments.extend(
            [
                f"--{role}-soft-moe-flag",
                "true",
                f"--{role}-soft-moe-num-experts",
                "3",
                f"--{role}-soft-moe-slots-per-expert",
                "1",
                f"--{role}-soft-moe-scoring-hidden-dim",
                "6",
            ]
        )
    values = cli_values(package, arguments)
    model = package.build_model(configuration(name, **values)).eval()
    routers = [
        module for module in model.modules() if isinstance(module, SoftMoERouter)
    ]
    assert routers
    layouts = {
        (
            router.num_experts,
            router.slots_per_expert,
            router.cfg.router_config.model_config.hidden_dim,
        )
        for router in routers
    }
    assert (2, 2, 8) in layouts
    if role is not None:
        assert (3, 1, 6) in layouts
    for router in routers:
        layer = router.cfg.router_config.model_config.layer_config
        assert layer.activation is ActivationOptions.GELU
        assert layer.layer_norm_position is LayerNormPositionOptions.BEFORE
        assert layer.dropout_probability == 0.1
    outputs = run_model(model, name, inputs_for(name))
    assert all(torch.isfinite(output).all() for output in outputs)
    sum(output.square().mean() for output in outputs).backward()
    for router in routers:
        gradients = [
            parameter.grad
            for parameter in router.parameters()
            if parameter.grad is not None
        ]
        assert gradients and all(
            torch.isfinite(gradient).all() for gradient in gradients
        )
        assert any(gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize("name", [name for name in PACKAGES if "expert" in name])
def test_existing_router_stack_flags_also_configure_soft_moe(name):
    package = model_package(name)
    values = cli_values(
        package,
        [
            "--soft-moe-flag",
            "true",
            "--soft-moe-num-experts",
            "2",
            "--router-stack-hidden-dim",
            "7",
            "--router-stack-num-layers",
            "2",
            "--router-stack-activation",
            "GELU",
            "--router-stack-dropout-probability",
            "0.2",
        ],
    )
    model = package.build_model(configuration(name, **values))
    routers = [
        module for module in model.modules() if isinstance(module, SoftMoERouter)
    ]
    assert routers
    for router in routers:
        stack = router.cfg.router_config.model_config
        assert stack.hidden_dim == 7 and stack.num_layers == 2
        assert stack.layer_config.activation is ActivationOptions.GELU
        assert stack.layer_config.dropout_probability == 0.2


def test_router_residual_normalization_and_last_layer_bias_flags_build_and_run():
    package = model_package("bert/linear")
    values = cli_values(
        package,
        [
            "--soft-moe-flag",
            "true",
            "--soft-moe-num-experts",
            "2",
            "--soft-moe-scoring-hidden-dim",
            "8",
            "--soft-moe-scoring-num-layers",
            "3",
            "--soft-moe-scoring-layer-norm-position",
            "BEFORE",
            "--soft-moe-scoring-normalization",
            "LAYER_NORM",
            "--soft-moe-scoring-last-layer-bias-option",
            "DISABLED",
            "--soft-moe-scoring-residual-connection-option",
            "AdditiveResidualConfig",
        ],
    )
    model = package.build_model(configuration(package.catalog_key, **values)).eval()
    routers = [
        module for module in model.modules() if isinstance(module, SoftMoERouter)
    ]
    assert routers
    for router in routers:
        stack = router.cfg.router_config.model_config
        assert stack.last_layer_bias_option is LastLayerBiasOptions.DISABLED
        assert stack.layer_config.normalization is NormalizationOptions.LAYER_NORM
        residual = stack.layer_config.residual_config
        assert isinstance(residual, AdditiveResidualConfig)
    outputs = run_model(model, package.catalog_key, inputs_for(package.catalog_key))
    sum(output.square().mean() for output in outputs).backward()
    assert all(torch.isfinite(output).all() for output in outputs)


def test_scoring_attention_residual_reports_raw_logit_incompatibility():
    package = model_package("bert/linear")
    values = cli_values(
        package,
        [
            "--soft-moe-flag",
            "true",
            "--soft-moe-scoring-residual-connection-option",
            "AttentionResidualConfig",
        ],
    )
    with pytest.raises(ValueError, match="router requires raw logits"):
        configuration(package.catalog_key, **values)


@pytest.mark.parametrize("name", PACKAGES)
@pytest.mark.parametrize(
    "residual_option", [WeightedResidualConfig, WeightedBlendResidualConfig]
)
def test_scoring_coefficient_models_use_public_residual_settings(name, residual_option):
    torch.manual_seed(19)
    package = model_package(name)
    arguments = [
        "--soft-moe-flag",
        "true",
        "--soft-moe-scoring-hidden-dim",
        "8",
        "--soft-moe-scoring-num-layers",
        "3",
        "--soft-moe-scoring-residual-connection-option",
        residual_option.__name__,
        "--soft-moe-scoring-residual-model-flag",
        "true",
        "--residual-coefficient-activation",
        "SIGMOID",
        "--residual-stack-independent-flag",
        "true",
        "--residual-stack-hidden-dim",
        "5",
        "--residual-stack-num-layers",
        "2",
        "--residual-stack-activation",
        "GELU",
        "--residual-stack-dropout-probability",
        "0.0",
        "--residual-stack-apply-output-postprocessing-flag",
        "false",
        "--residual-stack-bias-flag",
        "true",
    ]
    if name.startswith("mlp_mixer/"):
        arguments.extend(["--token-mixer-soft-moe-flag", "true"])
    elif "expert" in name:
        arguments.extend(["--attention-soft-moe-flag", "true"])
    values = cli_values(package, arguments)
    cfg = configuration(name, **values)
    model = package.build_model(cfg).eval()
    routers = [
        module for module in model.modules() if isinstance(module, SoftMoERouter)
    ]
    assert routers
    residuals = []
    for router in routers:
        scoring_stack = router.cfg.router_config.model_config
        assert scoring_stack.apply_output_postprocessing_flag is False
        residual_config = scoring_stack.layer_config.residual_config
        assert isinstance(residual_config, residual_option)
        assert residual_config.coefficient_activation is ActivationOptions.SIGMOID
        coefficient_stack = residual_config.model_config
        assert isinstance(coefficient_stack, LayerStackConfig)
        assert coefficient_stack.hidden_dim == 5
        assert coefficient_stack.num_layers == 2
        assert coefficient_stack.layer_config.activation is ActivationOptions.GELU
        residuals.extend(
            module
            for module in router.modules()
            if isinstance(getattr(module, "cfg", None), residual_option)
        )
    assert residuals
    assert all(
        residual.raw_weight is None and residual.model is not None
        for residual in residuals
    )
    outputs = run_model(model, name, inputs_for(name))
    assert all(torch.isfinite(output).all() for output in outputs)
    sum(output.square().mean() for output in outputs).backward()
    for residual in residuals:
        gradients = [
            parameter.grad
            for parameter in residual.model.parameters()
            if parameter.grad is not None
        ]
        assert gradients and all(
            torch.isfinite(gradient).all() for gradient in gradients
        )
        assert any(gradient.abs().sum() > 0 for gradient in gradients)
    restored = package.build_model(cfg).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    inputs = inputs_for(name)
    for actual, expected in zip(
        run_model(restored, name, inputs), run_model(model, name, inputs), strict=True
    ):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "residual_option", [WeightedResidualConfig, WeightedBlendResidualConfig]
)
@pytest.mark.parametrize("model_flag", [False, True])
@pytest.mark.parametrize(
    "activation",
    [ActivationOptions.SIGMOID, ActivationOptions.TANH, ActivationOptions.DISABLED],
)
def test_scoring_residual_activation_reaches_scalar_and_model_paths(
    residual_option, model_flag, activation
):
    package = model_package("bert/linear")
    cfg = configuration(
        package.catalog_key,
        soft_moe_flag=True,
        soft_moe_scoring_hidden_dim=8,
        soft_moe_scoring_num_layers=3,
        soft_moe_scoring_residual_connection_option=residual_option,
        soft_moe_scoring_residual_model_flag=model_flag,
        residual_coefficient_activation=activation,
    )
    model = package.build_model(cfg)
    residuals = [
        module
        for router in model.modules()
        if isinstance(router, SoftMoERouter)
        for module in router.modules()
        if isinstance(getattr(module, "cfg", None), residual_option)
    ]
    assert residuals
    for residual in residuals:
        assert residual.coefficient_activation is activation
        assert (residual.model is not None) is model_flag
        current, previous = torch.randn(2, 8), torch.randn(2, 8)
        raw = (
            residual.model.forward_hidden(torch.cat((current, previous), dim=-1))
            if model_flag
            else residual.raw_weight
        )
        coefficient = (
            raw if activation is ActivationOptions.DISABLED else activation(raw)
        )
        expected = (
            previous + coefficient * current
            if residual_option is WeightedResidualConfig
            else (1 - coefficient) * previous + coefficient * current
        )
        torch.testing.assert_close(residual(current, previous), expected)


@pytest.mark.parametrize("residual_option", [None, AdditiveResidualConfig])
def test_scoring_model_flag_rejects_residuals_without_coefficients(residual_option):
    with pytest.raises(
        ValueError,
        match="SCORING_RESIDUAL_MODEL_FLAG.*SCORING_RESIDUAL_CONNECTION_OPTION",
    ):
        configuration(
            "bert/linear",
            soft_moe_flag=True,
            soft_moe_scoring_residual_model_flag=True,
            soft_moe_scoring_residual_connection_option=residual_option,
        )


def test_scoring_model_flag_role_override_and_inherited_stack_settings():
    package = model_package("bert/expert_linear")
    cfg = configuration(
        package.catalog_key,
        soft_moe_flag=True,
        attention_soft_moe_flag=True,
        soft_moe_scoring_residual_connection_option=WeightedResidualConfig,
        soft_moe_scoring_residual_model_flag=True,
        attention_soft_moe_scoring_residual_model_flag=False,
        submodule_stack_hidden_dim=7,
        submodule_stack_num_layers=3,
        submodule_stack_activation=ActivationOptions.GELU,
        residual_stack_independent_flag=False,
        residual_stack_hidden_dim=99,
        residual_coefficient_activation=ActivationOptions.TANH,
    )
    mixtures = list(soft_configs(cfg))
    assert len(mixtures) == 2
    residuals = [
        mixture.router_config.model_config.layer_config.residual_config
        for mixture in mixtures
    ]
    assert all(
        residual.coefficient_activation is ActivationOptions.TANH
        for residual in residuals
    )
    modeled = [
        residual.model_config
        for residual in residuals
        if residual.model_config is not None
    ]
    assert len(modeled) == 1
    assert modeled[0].hidden_dim == 7
    assert modeled[0].num_layers == 3
    assert modeled[0].layer_config.activation is ActivationOptions.GELU


def test_scoring_coefficient_stack_rejects_recursive_model_configuration():
    with pytest.raises(
        ValueError, match="RESIDUAL_STACK_RESIDUAL_MODEL_FLAG.*recursively"
    ):
        configuration(
            "bert/linear",
            soft_moe_flag=True,
            soft_moe_scoring_residual_connection_option=WeightedResidualConfig,
            soft_moe_scoring_residual_model_flag=True,
            residual_stack_residual_model_flag=True,
        )


@pytest.mark.parametrize("has_feed_forward", [False, True])
def test_new_component_uses_shared_flags_without_a_custom_builder(has_feed_forward):
    config = ModuleType("test_future_model.config")
    config.CUSTOM_SOFT_MOE_CONFIG = None
    config.__annotations__ = {"CUSTOM_SOFT_MOE_CONFIG": SoftMoEConfig | None}
    if has_feed_forward:
        config.SOFT_MOE_CONFIG = None
        config.__annotations__["SOFT_MOE_CONFIG"] = SoftMoEConfig | None
    declare_soft_moe_runtime_defaults(vars(config))
    assert hasattr(config, "SOFT_MOE_FLAG") is has_feed_forward
    values = validate_runtime_default_values(
        {
            "custom_soft_moe_flag": True,
            "soft_moe_num_experts": 3,
            "custom_soft_moe_slots_per_expert": 4,
        },
        package=config.__name__,
        config_module=config,
    )
    assert set(values) == {"custom_soft_moe_config"}
    assert values["custom_soft_moe_config"].slots_per_expert == 4
    assert values["custom_soft_moe_config"].num_experts == 3


@pytest.mark.parametrize("name", PACKAGES)
def test_explicit_soft_config_is_preserved_without_mutating_it(name):
    routing = soft_routing()
    expected = copy.deepcopy(routing)
    model = model_package(name).build_model(
        configuration(
            name,
            soft_moe_flag=True,
            soft_moe_slots_per_expert=7,
            soft_moe_config=routing,
        )
    )
    routers = [
        module for module in model.modules() if isinstance(module, SoftMoERouter)
    ]
    assert routers and all(router.slots_per_expert == 2 for router in routers)
    assert routing == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("soft-moe-num-experts", "0"),
        ("soft-moe-slots-per-expert", "0"),
        ("soft-moe-scoring-hidden-dim", "0"),
        ("soft-moe-scoring-num-layers", "0"),
        ("soft-moe-scoring-dropout-probability", "1.5"),
        ("soft-moe-scoring-dropout-probability", "nan"),
        ("attention-soft-moe-num-experts", "-1"),
    ],
)
def test_cli_rejects_invalid_soft_controls(field, value):
    from model_runtime.packages.runtime_defaults import RuntimeDefaultsError

    with pytest.raises((ValueError, RuntimeDefaultsError), match="positive|between"):
        model_package("bert/expert_linear").runtime_defaults_spec.parse_overrides(
            {field: value}
        )
