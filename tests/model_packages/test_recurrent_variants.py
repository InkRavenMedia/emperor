from importlib import import_module
from types import ModuleType

import pytest
import torch

from emperor.layers import (
    AttentionResidualConfig,
    HierarchicalReasoningModelRecurrentConfig,
    RecurrentCompositionConfig,
    TinyRecursiveModelRecurrentConfig,
)
from models.catalog import discover_model_packages, model_package
from tests.model_packages.test_attention_residual_options import walk
from tests.model_packages.test_attention_source_memory import (
    configuration as attention_configuration,
)
from tests.model_packages.test_memory_expert_parity import inputs_for, run_model
from tests.model_packages.test_mlp_mixer_source_memory import (
    configuration as mixer_configuration,
)
from tests.model_packages.test_recurrent_runtime_control_scope import (
    _config_builder_type,
)
from tests.model_packages.test_soft_moe_cli import cli_values

SEQUENCES = {"bert", "gpt", "mlp_mixer", "transformer", "vit"}


@pytest.mark.parametrize(
    "name",
    [
        "bert/linear",
        "gpt/linear",
        "vit/linear",
        "transformer/linear_adaptive",
        "transformer/expert_linear",
        "transformer/expert_linear_adaptive",
    ],
)
@pytest.mark.parametrize(
    "variant",
    [TinyRecursiveModelRecurrentConfig, HierarchicalReasoningModelRecurrentConfig],
)
def test_python_config_and_flat_options_build_same_recurrent_variant(
    name, variant, monkeypatch
):
    package = "models." + name.replace("/", ".")
    defaults = import_module(package + ".runtime_defaults")
    config = import_module(package + ".config")
    options = dict(
        recurrent_flag=True,
        recurrent_composition_option=variant,
        recurrent_latent_updates_per_answer_update=4,
        recurrent_answer_update_count=3,
        recurrent_low_cycles=4,
        recurrent_high_cycles=3,
        recurrent_initial_iterations=2,
        recurrent_initialization_standard_deviation=0.25,
        recurrent_compression_slots=3,
        recurrent_residual_connection_option=AttentionResidualConfig,
        recurrent_residual_block_size=1,
        recurrent_residual_rms_norm_epsilon=1e-6,
    )
    flat = defaults.runtime_from_flat(options)
    translation = name.startswith("transformer/")
    if translation:
        for key, value in options.items():
            monkeypatch.setattr(config, key.upper(), value)
        direct = defaults.runtime_from_config()
    else:
        source = ModuleType(config.__name__)
        vars(source).update(vars(config))
        vars(source).update({key.upper(): value for key, value in options.items()})
        direct = defaults.runtime_from_config(source)
    builder = _config_builder_type(name)
    for runtime in (direct, flat):
        assert runtime.recurrent_composition_option is variant
        experiment = builder(runtime=runtime).build().experiment_config
        owners = [
            cfg
            for cfg in walk(experiment)
            if isinstance(cfg, RecurrentCompositionConfig)
        ]
        assert len(owners) == (2 if translation else 1)
        for owner in owners:
            assert type(owner) is variant
            assert owner.initial_iterations == 2
            assert owner.initialization_standard_deviation == 0.25
            assert owner.compression_config.max_slots == 3
            assert isinstance(owner.residual_config, AttentionResidualConfig)
            assert owner.residual_config.block_size == 1
            assert owner.residual_config.rms_norm_epsilon == 1e-6
            if variant is TinyRecursiveModelRecurrentConfig:
                assert (
                    owner.latent_updates_per_answer_update,
                    owner.answer_update_count,
                ) == (4, 3)
            else:
                assert (owner.low_cycles, owner.high_cycles) == (4, 3)


def test_vit_linear_flat_flag_can_disable_recurrence_enabled_in_config(monkeypatch):
    from models.vit.linear import config, runtime_defaults
    from models.vit.linear.config_builder import VitLinearConfigBuilder

    monkeypatch.setattr(config, "RECURRENT_FLAG", True)
    runtime = runtime_defaults.runtime_from_flat({"recurrent_flag": False})
    experiment = VitLinearConfigBuilder(runtime=runtime).build().experiment_config
    assert not isinstance(experiment.encoder_config, RecurrentCompositionConfig)


def experiment_configuration(package, family, options):
    name = package.catalog_key
    if family == "mlp_mixer":
        return mixer_configuration(name, **options)
    if family in {"bert", "gpt", "vit"}:
        if name.endswith("_cluster"):
            options["cluster_x_axis_total_neurons"] = 1
        return attention_configuration(name, attn_source_memory_flag=False, **options)
    if family == "transformer":
        options.update(
            vocab_size=32,
            model_dim=8,
            source_sequence_length=4,
            target_sequence_length=4,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            ff_stack_hidden_dim=8,
        )
    return package.build_configuration(config_overrides=dict(batch_size=2, **options))


@pytest.mark.training
@pytest.mark.parametrize("name", [p.catalog_key for p in discover_model_packages()])
@pytest.mark.parametrize(
    "variant",
    [TinyRecursiveModelRecurrentConfig, HierarchicalReasoningModelRecurrentConfig],
)
def test_recurrent_variant_experiment_step(name, variant):
    torch.manual_seed(41)
    family = name.split("/")[0]
    package = model_package(name)
    args = [
        "--recurrent-flag",
        "true",
        "--recurrent-composition-option",
        variant.__name__,
        "--recurrent-initial-iterations",
        "2",
        "--recurrent-residual-connection-option",
        "AttentionResidualConfig",
        "--recurrent-residual-block-size",
        "1",
        "--recurrent-residual-rms-norm-epsilon",
        "1e-6",
    ]
    if variant is TinyRecursiveModelRecurrentConfig:
        args.extend(
            [
                "--recurrent-latent-updates-per-answer-update",
                "2",
                "--recurrent-answer-update-count",
                "2",
            ]
        )
    else:
        args.extend(["--recurrent-low-cycles", "2", "--recurrent-high-cycles", "2"])
    if family in SEQUENCES:
        args.extend(["--recurrent-compression-slots", "2"])
    config = experiment_configuration(
        package,
        family,
        cli_values(package, args, preset=package.preset_name(package.default_preset)),
    )
    owners = [
        cfg
        for cfg in walk(config.experiment_config)
        if isinstance(cfg, RecurrentCompositionConfig)
    ]
    assert len(owners) == (2 if family == "transformer" else 1)
    for owner in owners:
        assert type(owner) is variant
        assert isinstance(owner.residual_config, AttentionResidualConfig)
        if family in SEQUENCES:
            assert owner.compression_config.max_slots == 2
    model = package.build_model(config).train()
    if family == "transformer":
        tokens = inputs_for(name)
        result = model(tokens, tokens)
        logits = result[0] if isinstance(result, tuple) else result
    elif family in SEQUENCES:
        logits = run_model(model, name, inputs_for(name))[0]
    else:
        logits = run_model(model, name, torch.randn(2, config.input_dim))[0]
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    recurrent_modules = [
        m for m in model.modules() if type(m) is variant().registry_owner()
    ]
    assert recurrent_modules
    active_branches = []
    for recurrent in recurrent_modules:
        branch = (
            recurrent.step.block_model
            if variant is TinyRecursiveModelRecurrentConfig
            else recurrent.step.high_model
        )
        active_branches.append(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in branch.parameters()
            )
        )
        if family in SEQUENCES:
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in recurrent.step.compressor.parameters()
            )
    assert any(active_branches)
    torch.optim.SGD(model.parameters(), lr=0.001).step()


@pytest.mark.parametrize("family", ["vit", "mlp_mixer"])
@pytest.mark.parametrize(
    "variant",
    [TinyRecursiveModelRecurrentConfig, HierarchicalReasoningModelRecurrentConfig],
)
def test_compressed_vision_cycle_growth_and_checkpoint(family, variant):
    torch.manual_seed(42)
    package = model_package(f"{family}/expert_linear_adaptive")
    config = experiment_configuration(
        package,
        family,
        dict(
            recurrent_flag=True,
            recurrent_composition_option=variant,
            recurrent_latent_updates_per_answer_update=3,
            recurrent_low_cycles=3,
            recurrent_initial_iterations=1,
            recurrent_answer_update_count=2,
            recurrent_high_cycles=2,
            recurrent_iteration_increment=1,
            recurrent_forward_calls_before_iteration_increment=1,
            recurrent_compression_slots=2,
            recurrent_residual_connection_option=AttentionResidualConfig,
            recurrent_residual_block_size=1,
            recurrent_residual_rms_norm_epsilon=1e-6,
        ),
    )
    model = package.build_model(config).eval()
    recurrent = next(
        m for m in model.modules() if type(m) is variant().registry_owner()
    )
    inner_lengths, outer_lengths = [], []
    blocks = (
        (recurrent.step.block_model,)
        if variant is TinyRecursiveModelRecurrentConfig
        else (recurrent.step.low_model, recurrent.step.high_model)
    )
    for block in blocks:
        block.register_forward_pre_hook(
            lambda module, args: inner_lengths.append(args[0].hidden.shape[-2])
        )
    pipeline = recurrent.step.pipeline
    connections = [
        pipeline.residual.connection,
        *pipeline.residual_schedule.subsequent_connections,
    ]
    for connection in connections:
        connection.register_forward_pre_hook(
            lambda module, args: outer_lengths.append(args[0].shape[-2])
        )
    images = inputs_for(package.catalog_key)
    with torch.no_grad():
        run_model(model, package.catalog_key, images)
        full_length = config.sequence_length
        assert inner_lengths == [2] * 4 and outer_lengths == [full_length]
        restored = package.build_model(config).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        inner_lengths.clear()
        outer_lengths.clear()
        output = run_model(model, package.catalog_key, images)[0]
        assert inner_lengths == [2] * 8 and outer_lengths == [full_length] * 2
        torch.testing.assert_close(
            run_model(restored, package.catalog_key, images)[0], output
        )


@pytest.mark.parametrize("name", ["gpt/linear", "transformer/linear"])
@pytest.mark.parametrize(
    "variant",
    [TinyRecursiveModelRecurrentConfig, HierarchicalReasoningModelRecurrentConfig],
)
def test_causal_recurrent_compression_preserves_prefix(name, variant):
    package = model_package(name)
    config = experiment_configuration(
        package,
        name.split("/")[0],
        dict(
            recurrent_flag=True,
            recurrent_composition_option=variant,
            recurrent_compression_slots=2,
            recurrent_initial_iterations=2,
            recurrent_residual_connection_option=AttentionResidualConfig,
            recurrent_residual_block_size=1,
            recurrent_residual_rms_norm_epsilon=1e-6,
        ),
    )
    model = package.build_model(config).eval()
    source = inputs_for(name)
    changed = source.clone()
    changed[:, -1] = 6

    def logits(tokens):
        result = (
            model(source, tokens) if name.startswith("transformer") else model(tokens)
        )
        return result[0] if isinstance(result, tuple) else result

    with torch.no_grad():
        torch.testing.assert_close(logits(source)[:, :-1], logits(changed)[:, :-1])
