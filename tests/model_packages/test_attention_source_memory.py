"""Public BERT/GPT/ViT configuration and execution of appended memory sources."""

import pytest
import torch

from emperor.attention._base import MultiHeadAttentionAbstract
from emperor.layers import LayerStackConfig, RecurrentLayerConfig
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig
from models.catalog import model_package

PACKAGES = tuple(
    f"{family}/{variant}"
    for family in ("bert", "gpt", "vit")
    for variant in (
        "linear",
        "linear_adaptive",
        "expert_linear",
        "expert_linear_adaptive",
    )
)


def configuration(package_name, *, default_source_stack=False, **overrides):
    values = dict(
        input_dim=16,
        output_dim=16,
        batch_size=2,
        sequence_length=6,
        hidden_dim=8,
        stack_num_layers=1,
        stack_dropout_probability=0.0,
        embedding_dropout_probability=0.0,
        attn_num_heads=2,
        submodule_stack_hidden_dim=8,
        submodule_stack_num_layers=1,
        attn_source_memory_flag=True,
        attn_source_memory_chunk_size=2,
    )
    if "expert" in package_name:
        values.update(
            num_experts=3,
            top_k=2,
            capacity_factor=0.0,
            expert_stack_hidden_dim=8,
            router_stack_hidden_dim=8,
            ff_stack_hidden_dim=8,
        )
    if "adaptive" in package_name:
        values["adaptive_generator_stack_hidden_dim"] = 8
    if default_source_stack:
        values.pop("submodule_stack_hidden_dim")
        values.pop("submodule_stack_num_layers")
    if package_name.startswith("vit"):
        values.pop("sequence_length")
        values.pop("embedding_dropout_probability")
        values.update(
            input_dim=64,
            image_patch_size=4,
            input_channels=1,
            image_height=8,
            patch_dropout_probability=0.0,
        )
    values.update(overrides)
    return model_package(package_name).build_configuration(config_overrides=values)


def run_model(model, package_name, inputs=None):
    if inputs is None:
        inputs = (
            torch.randn(2, 1, 8, 8)
            if package_name.startswith("vit")
            else torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        )
    if package_name.startswith("bert"):
        return model(inputs, torch.ones_like(inputs), torch.zeros_like(inputs))
    result = model(inputs)
    return result if isinstance(result, tuple) else (result,)


@pytest.mark.parametrize(
    ("option", "field", "config_type"),
    [
        ("LINEAR_STACK", "STACK_NUM_LAYERS", LayerStackConfig),
        ("RECURRENT_LINEAR", "RECURRENT_MAX_STEPS", RecurrentLayerConfig),
    ],
)
def test_source_memory_composed_generators_have_flat_runtime_options(
    option, field, config_type
):
    package_name = "gpt/linear"
    package = model_package(package_name)
    prefix = "ATTN_SOURCE_MEMORY_MODEL_"
    controls = {
        prefix + "OPTION": option,
        prefix + "BIAS_FLAG": "false",
        prefix + field: "3",
    }
    overrides = {
        key.lower(): package.runtime_defaults_spec.parse_value(key, value)
        for key, value in controls.items()
    }
    model = package.build_model(configuration(package_name, **overrides)).eval()
    memory = next(
        attention.projector.source_memory
        for attention in model.modules()
        if isinstance(attention, MultiHeadAttentionAbstract)
    )
    assert isinstance(memory.generator_model_config, config_type)
    if field == "STACK_NUM_LAYERS":
        assert memory.model.num_layers == 3
    else:
        assert memory.model.max_steps == 3
    assert torch.isfinite(run_model(model, package_name)[0]).all()


@pytest.mark.parametrize("package_name", PACKAGES)
def test_source_memory_does_not_require_a_separate_controller_stack(package_name):
    package = model_package(package_name)
    model = package.build_model(
        configuration(package_name, default_source_stack=True)
    ).eval()
    attentions = [
        m for m in model.modules() if isinstance(m, MultiHeadAttentionAbstract)
    ]
    assert attentions and all(m.projector.source_memory is not None for m in attentions)
    assert torch.isfinite(run_model(model, package_name)[0]).all()


@pytest.mark.parametrize("package_name", PACKAGES)
def test_public_options_build_and_execute_memory_in_every_package(package_name):
    torch.manual_seed(81)
    package = model_package(package_name)
    model = package.build_model(configuration(package_name)).eval()
    attentions = [
        m for m in model.modules() if isinstance(m, MultiHeadAttentionAbstract)
    ]
    assert attentions
    calls = []
    handles = []
    for attention in attentions:
        memory = attention.projector.source_memory.model
        assert isinstance(memory.cfg, LinearLayerConfig)
        assert attention.projector.source_memory.context.chunk_size == 2
        key_parameters = dict(attention.projector.key_model.named_parameters())
        memory_parameters = dict(memory.named_parameters())
        assert {p.data_ptr() for p in memory_parameters.values()}.isdisjoint(
            p.data_ptr() for p in key_parameters.values()
        )
        handles.append(
            memory.register_forward_hook(lambda _m, _a, out: calls.append(out))
        )
    outputs = run_model(model, package_name)
    expected_shape = (2, 16) if package_name.startswith("vit") else (2, 4, 16)
    assert outputs[0].shape == expected_shape
    assert len(calls) == len(attentions)
    sum(output.square().mean() for output in outputs).backward()
    for attention in attentions:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in attention.projector.source_memory.model.parameters()
        )
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("package_name", PACKAGES)
def test_source_memory_options_are_visible_parseable_and_independent(package_name):
    from model_runtime.inspection.schema import configuration_schema

    package = model_package(package_name)
    spec = package.runtime_defaults_spec
    raw = {
        "ATTN_SOURCE_MEMORY_FLAG": "true",
        "ATTN_SOURCE_MEMORY_CHUNK_SIZE": "2",
    }
    fields = {field.key for field in configuration_schema(package).fields}
    assert fields.issuperset(raw)
    from models._attention_source_memory import SOURCE_MEMORY_FIELDS

    assert {field for field in fields if field.startswith("ATTN_SOURCE_MEMORY_")} == {
        "ATTN_SOURCE_MEMORY_FLAG",
        *(f"ATTN_{name.upper()}" for name in SOURCE_MEMORY_FIELDS),
    }
    values = {key.lower(): spec.parse_value(key, value) for key, value in raw.items()}
    config = configuration(package_name, attn_memory_flag=True, **values)
    model = package.build_model(config)
    attentions = [
        m for m in model.modules() if isinstance(m, MultiHeadAttentionAbstract)
    ]
    for attention in attentions:
        assert isinstance(
            attention.projector.source_memory.model.cfg, LinearLayerConfig
        )
        assert attention.cfg.projection_model_config.shared_memory_config is not None
    model.eval()
    outputs = run_model(model, package_name)
    assert torch.isfinite(outputs[0]).all()
    disabled = package.build_model(
        configuration(package_name, attn_source_memory_flag=False)
    )
    assert all(
        m.projector.source_memory is None
        for m in disabled.modules()
        if isinstance(m, MultiHeadAttentionAbstract)
    )


@pytest.mark.parametrize(
    "package_name", [name for name in PACKAGES if "expert" in name]
)
def test_soft_expert_packages_preserve_causality_samples_and_checkpoints(
    package_name,
):
    from emperor.experts import SoftMoEConfig
    from tests.unit.test_expert_soft_moe import scoring_config

    config = configuration(
        package_name,
        attention_soft_moe_config=SoftMoEConfig(
            num_experts=2,
            slots_per_expert=2,
            router_config=RouterConfig(
                noisy_topk_flag=False, model_config=scoring_config()
            ),
        ),
    )
    package = model_package(package_name)
    model = package.build_model(config).eval()
    ids = (
        torch.randn(2, 1, 8, 8)
        if package_name.startswith("vit")
        else torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    )

    def run(module, tokens):
        return run_model(module, package_name, tokens)[0]

    expected = run(model, ids)
    changed = ids.clone()
    changed[1] = 5
    torch.testing.assert_close(run(model, changed)[0], expected[0])
    if package_name.startswith("gpt"):
        changed = ids.clone()
        changed[:, 2:] = 6
        torch.testing.assert_close(run(model, changed)[:, :2], expected[:, :2])
    restored = package.build_model(config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(run(restored, ids), expected)
    # Exercise gradients through the same source branch in training mode (no optimizer step).
    model.train()
    run(model, ids).square().mean().backward()
    for module in model.modules():
        if isinstance(module, MultiHeadAttentionAbstract):
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in module.projector.source_memory.model.parameters()
            )
