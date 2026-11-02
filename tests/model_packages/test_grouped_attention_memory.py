import json

import pytest
import torch

from emperor.attention import (
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
)
from emperor.augmentations.adaptive_parameters import MeanGroupingConfig
from models._attention_source_memory import SOURCE_MEMORY_FIELDS
from models.catalog import model_package
from models.cli_selection import resolve_cli_selection
from models.experiment_cli_parser import get_experiment_parser
from tests.model_packages.test_attention_source_memory import configuration, run_model

ATTENTION_TYPES = tuple(
    config().registry_owner()
    for config in (
        SelfAttentionConfig,
        IndependentAttentionConfig,
        MixtureOfAttentionHeadsConfig,
    )
)

PACKAGES = [
    f"{family}/{variant}"
    for family in ("bert", "gpt", "vit", "transformer")
    for variant in (
        "linear",
        "linear_adaptive",
        "expert_linear",
        "expert_linear_adaptive",
    )
]


@pytest.fixture(autouse=True)
def isolated_random_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(705)
        yield


def parse_overrides(package_name, *arguments):
    package = model_package(package_name)
    parsed = get_experiment_parser(package).parse_args(
        ["--preset", "baseline", "--config", *arguments]
    )
    return resolve_cli_selection(
        parsed, package, package.preset_type, build_monitor_callbacks=False
    ).config_overrides


def build_config(package_name, overrides):
    if not package_name.startswith("transformer/"):
        return configuration(package_name, **overrides)
    values = dict(
        batch_size=2,
        vocab_size=32,
        model_dim=8,
        source_sequence_length=5,
        target_sequence_length=5,
        encoder_num_layers=1,
        decoder_num_layers=1,
        attn_num_heads=2,
        ff_stack_hidden_dim=8,
        dropout_probability=0.0,
        decoder_cross_attn_source_memory_flag=False,
        attn_source_memory_flag=True,
    )
    if "expert" in package_name:
        values.update(
            num_experts=3,
            top_k=2,
            capacity_factor=0.0,
            expert_stack_hidden_dim=8,
            router_stack_hidden_dim=8,
        )
    values.update(overrides)
    return model_package(package_name).build_configuration(config_overrides=values)


def execute(model, package_name):
    if package_name.startswith("transformer/"):
        return model(
            torch.tensor([[2, 8, 3, 0], [2, 9, 10, 3]]),
            torch.tensor([[2, 11, 3], [2, 12, 3]]),
        )
    return run_model(model, package_name)


@pytest.mark.parametrize("package_name", PACKAGES)
def test_cli_options_reach_built_memory_and_forward(package_name):
    from model_runtime.inspection.schema import configuration_schema

    package = model_package(package_name)
    fields = {field.key for field in configuration_schema(package).fields}
    assert {f"ATTN_{name.upper()}" for name in SOURCE_MEMORY_FIELDS} <= fields
    assert not any(field.endswith("SOURCE_MEMORY_NUM_EMBEDDINGS") for field in fields)
    overrides = parse_overrides(
        package_name,
        "--attn-source-memory-flag",
        "true",
        "--attn-source-memory-chunk-size",
        "2",
        "--attn-source-memory-grouping-method",
        "MeanGroupingConfig",
        "--attn-source-memory-model-config",
        '{"type":"LinearLayerConfig","bias_flag":false}',
        "--attn-source-memory-summary-normalization",
        "RMS_NORM",
        "--attn-source-memory-rms-norm-epsilon",
        "0.0003",
    )
    model = package.build_model(build_config(package_name, overrides)).eval()
    attentions = [
        m
        for m in model.modules()
        if isinstance(m, ATTENTION_TYPES) and m.projector.source_memory is not None
    ]
    assert attentions
    for attention in attentions:
        memory = attention.projector.source_memory
        assert isinstance(memory.context.grouper.cfg, MeanGroupingConfig)
        assert memory.context.grouper.cfg.chunk_size == 2
        assert memory.context.grouper.cfg.rms_norm_epsilon == 0.0003
        assert memory.model.bias_params is None
    generated_memories = []

    def check_memory_shape(module, inputs, outputs):
        sequence_length, batch_size, embedding_dim = inputs[0].shape
        num_chunks = (sequence_length + 1) // 2
        assert outputs[0].shape == (num_chunks * batch_size, embedding_dim)
        assert outputs[1].shape == (num_chunks * batch_size, 1)
        generated_memories.append(module)

    hooks = [
        attention.projector.source_memory.register_forward_hook(check_memory_shape)
        for attention in attentions
    ]
    outputs = execute(model, package_name)
    for hook in hooks:
        hook.remove()
    assert len(generated_memories) == len(attentions)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    loss = sum(
        output.square().mean() for output in outputs if isinstance(output, torch.Tensor)
    )
    assert torch.isfinite(loss)
    loss.backward()
    for attention in attentions:
        assert attention.projector.source_memory.model.weight_params.grad is not None


@pytest.mark.parametrize("package_name", PACKAGES)
def test_memory_flag_requires_chunk_size_in_every_package(package_name):
    overrides = parse_overrides(package_name, "--attn-source-memory-flag", "true")
    overrides["attn_source_memory_chunk_size"] = None
    with pytest.raises(ValueError, match="chunk_size"):
        build_config(package_name, overrides)
    disabled = model_package(package_name).build_model(
        build_config(package_name, overrides | {"attn_source_memory_flag": False})
    )
    assert all(
        module.projector.source_memory is None
        for module in disabled.modules()
        if isinstance(module, ATTENTION_TYPES)
    )


@pytest.mark.parametrize("package_name", PACKAGES)
def test_cli_rejects_removed_memory_count_option(package_name):
    prefixes = ["attn"]
    if package_name.startswith("transformer/"):
        prefixes += ["encoder-attn", "decoder-self-attn", "decoder-cross-attn"]
    for prefix in prefixes:
        with pytest.raises(SystemExit) as error:
            parse_overrides(
                package_name, f"--{prefix}-source-memory-num-embeddings", "6"
            )
        assert error.value.code == 2


@pytest.mark.parametrize("method", ["AttentionGroupingConfig", "MeanStdGroupingConfig"])
def test_cli_learned_grouping_model_and_generator_round_trip(method):
    from model_runtime.packages.configuration import serialize_config_value

    grouping_model = {
        "type": "LayerConfig",
        "activation": "ActivationOptions.DISABLED",
        "dropout_probability": 0.0,
        "layer_norm_position": "LayerNormPositionOptions.DISABLED",
        "layer_model_config": {"type": "LinearLayerConfig", "bias_flag": True},
    }
    package = model_package("bert/linear")
    overrides = parse_overrides(
        package.catalog_key,
        "--attn-source-memory-flag",
        "true",
        "--attn-source-memory-chunk-size",
        "2",
        "--attn-source-memory-grouping-method",
        method,
        "--attn-source-memory-grouping-model-config",
        json.dumps(grouping_model),
    )
    config = overrides["attn_source_memory_grouping_model_config"]
    assert (
        package.runtime_defaults_spec.parse_value(
            "ATTN_SOURCE_MEMORY_GROUPING_MODEL_CONFIG", serialize_config_value(config)
        )
        == config
    )
    model = package.build_model(build_config(package.catalog_key, overrides))
    outputs = execute(model, package.catalog_key)
    outputs[0].sum().backward()
    for module in model.modules():
        if isinstance(module, ATTENTION_TYPES):
            assert any(
                p.grad is not None
                for p in module.projector.source_memory.context.grouper.parameters()
            )


def test_structured_config_round_trip_and_scalar_overrides_do_not_mutate_original():
    from emperor.attention import AttentionSourceMemoryConfig
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterGroupingScopeOptions,
        SumGroupingConfig,
    )
    from emperor.linears import LinearLayerConfig
    from model_runtime.packages.configuration import serialize_config_value

    explicit = AttentionSourceMemoryConfig(
        model_config=LinearLayerConfig(bias_flag=False),
        grouping_config=SumGroupingConfig(
            chunk_size=2, scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE
        ),
    )
    package = model_package("bert/linear")
    overrides = parse_overrides(
        package.catalog_key,
        "--attn-source-memory-flag",
        "true",
        "--attn-source-memory-config",
        serialize_config_value(explicit),
        "--attn-source-memory-chunk-size",
        "3",
    )
    model = package.build_model(build_config(package.catalog_key, overrides))
    memory = next(
        m.projector.source_memory
        for m in model.modules()
        if isinstance(m, ATTENTION_TYPES)
    )
    assert memory.context.chunk_size == 3
    assert memory.model.bias_params is None
    assert explicit.grouping_config.chunk_size == 2
    assert explicit.grouping_config.sequence_length is None


@pytest.mark.parametrize(
    "value",
    [
        '{"type":"NotARegisteredConfig"}',
        '{"type":"LinearLayerConfig","unexpected":1}',
        '{"type":"LinearLayerConfig",}',
        '{"type":"ActivationOptions"}',
    ],
)
def test_cli_rejects_invalid_config_objects(value):
    with pytest.raises(ValueError):
        parse_overrides("bert/linear", "--attn-source-memory-model-config", value)


def test_structured_config_rejects_removed_memory_count():
    with pytest.raises(ValueError, match="num_memory_embeddings"):
        parse_overrides(
            "bert/linear",
            "--attn-source-memory-config",
            '{"type":"AttentionSourceMemoryConfig","num_memory_embeddings":20}',
        )


def test_standard_experiment_dispatch_and_readme_soft_moe_forward(
    monkeypatch, tmp_path
):
    from models.project_cli.main import run_experiment
    from tests.model_packages.test_bert_expert_adaptive_cli import (
        configuration_from_cli,
        readme_arguments,
    )

    memory_flags = [
        "--attn-source-memory-chunk-size",
        "10",
        "--attn-source-memory-grouping-method",
        "SumGroupingConfig",
        "--attn-source-memory-summary-normalization",
        "DISABLED",
        "--attn-source-memory-model-config",
        '{"type":"LinearLayerConfig","bias_flag":true}',
    ]
    invocation = [
        "--model-type",
        "bert",
        "--model",
        "expert_linear_adaptive",
        "--preset",
        "baseline",
        "--experiment-task",
        "bert-pretraining",
        "--datasets",
        "wiki-text2-bert-pretraining",
        "--config",
        "--sequence-length",
        "100",
        "--attn-source-memory-flag",
        "true",
        *memory_flags,
    ]
    observed = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "models.package_cli.execute_runs",
        lambda package, plan, **kwargs: observed.append((package, plan)),
    )
    assert run_experiment(invocation) == 0
    assert observed[0][0].catalog_key == "bert/expert_linear_adaptive"
    assert len(observed[0][1].runs) == 1
    config = configuration_from_cli(
        readme_arguments()
        + ["--input-dim", "16", "--output-dim", "16", "--batch-size", "2"]
    )
    model = model_package("bert/expert_linear_adaptive").build_model(config).eval()
    model.set_token_vocabulary([f"token{index}" for index in range(16)])
    ids = torch.randint(0, 16, (2, 100))
    outputs = model(ids, torch.ones_like(ids), torch.zeros_like(ids))
    assert outputs[0].shape == (2, 100, 16)
    assert torch.isfinite(outputs[0]).all()
    assert all(
        m.slots_per_expert == 10
        for m in model.modules()
        if type(m).__name__ == "SoftMoERouter"
    )
    memory = next(
        m.projector.source_memory
        for m in model.modules()
        if isinstance(m, ATTENTION_TYPES) and m.projector.source_memory is not None
    )
    memory_rows, _, _ = memory(torch.randn(100, 2, 32))
    assert memory.context.chunk_size == 10
    assert memory_rows.shape == (10 * 2, 32)


@pytest.mark.parametrize(
    "variant", ["linear", "linear_adaptive", "expert_linear", "expert_linear_adaptive"]
)
def test_transformer_path_overrides_and_cross_attention_rejection(variant):
    name = f"transformer/{variant}"
    overrides = parse_overrides(
        name,
        "--attn-source-memory-flag",
        "true",
        "--attn-source-memory-chunk-size",
        "2",
        "--decoder-self-attn-source-memory-chunk-size",
        "1",
        "--decoder-self-attn-source-memory-grouping-method",
        "MeanGroupingConfig",
        "--decoder-cross-attn-source-memory-flag",
        "false",
    )
    model = model_package(name).build_model(build_config(name, overrides))
    memories = [
        m.cfg.source_memory_config
        for m in model.modules()
        if isinstance(m, ATTENTION_TYPES) and m.projector.source_memory is not None
    ]
    assert {m.grouping_config.chunk_size for m in memories} == {1, 2}
    assert any(isinstance(m.grouping_config, MeanGroupingConfig) for m in memories)
    assert torch.isfinite(execute(model, name)[0]).all()
    with pytest.raises(ValueError, match="self-attention|cross-attention"):
        build_config(name, overrides | {"decoder_cross_attn_source_memory_flag": True})
