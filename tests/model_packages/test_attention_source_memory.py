"""Public BERT/GPT/ViT configuration and execution of appended memory sources."""

import pytest
import torch

from emperor.attention._base import MultiHeadAttentionAbstract
from emperor.memory import AttentionDynamicMemoryConfig
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
        attn_source_memory_option=AttentionDynamicMemoryConfig,
        attn_source_memory_num_memory_slots=3,
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


@pytest.mark.parametrize("package_name", PACKAGES)
def test_source_memory_can_use_default_controller_stack(package_name):
    package = model_package(package_name)
    model = package.build_model(
        configuration(package_name, default_source_stack=True)
    ).eval()
    attentions = [
        m for m in model.modules() if isinstance(m, MultiHeadAttentionAbstract)
    ]
    assert attentions and all(m.source_memory is not None for m in attentions)
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
        memory = attention.source_memory.model
        assert isinstance(memory.cfg, AttentionDynamicMemoryConfig)
        assert memory.num_memory_slots == 3
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
            for p in attention.source_memory.parameters()
        )
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("package_name", PACKAGES)
def test_source_memory_options_are_visible_parseable_and_independent(package_name):
    from emperor.memory import (
        ElementWiseWeightedDynamicMemoryConfig,
        GatedResidualDynamicMemoryConfig,
        WeightedDynamicMemoryConfig,
    )
    from model_runtime.inspection.schema import configuration_schema

    package = model_package(package_name)
    spec = package.runtime_defaults_spec
    raw = {
        "ATTN_SOURCE_MEMORY_FLAG": "true",
        "ATTN_SOURCE_MEMORY_NUM_MEMORY_SLOTS": "3",
        "ATTN_SOURCE_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE": "0.02",
        "ATTN_SOURCE_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS": "2",
    }
    fields = {field.key for field in configuration_schema(package).fields}
    assert fields.issuperset(raw)
    assert "ATTN_SOURCE_MEMORY_OPTION" in fields
    for variant in (
        AttentionDynamicMemoryConfig,
        GatedResidualDynamicMemoryConfig,
        WeightedDynamicMemoryConfig,
        ElementWiseWeightedDynamicMemoryConfig,
    ):
        values = {
            key.lower(): spec.parse_value(key, value) for key, value in raw.items()
        }
        values["attn_source_memory_option"] = spec.parse_value(
            "ATTN_SOURCE_MEMORY_OPTION", variant.__name__
        )
        config = configuration(package_name, attn_memory_flag=True, **values)
        model = package.build_model(config)
        attentions = [
            m for m in model.modules() if isinstance(m, MultiHeadAttentionAbstract)
        ]
        for attention in attentions:
            memory = attention.source_memory.model
            assert type(memory.cfg) is variant
            assert memory.test_time_training_learning_rate == 0.02
            assert memory.test_time_training_num_inner_steps == 2
            assert (
                attention.cfg.projection_model_config.shared_memory_config is not None
            )
            if variant is AttentionDynamicMemoryConfig:
                assert memory.num_memory_slots == 3
    model.eval()
    outputs = run_model(model, package_name)
    assert torch.isfinite(outputs[0]).all()
    disabled = package.build_model(
        configuration(package_name, attn_source_memory_flag=False)
    )
    assert all(
        m.source_memory is None
        for m in disabled.modules()
        if isinstance(m, MultiHeadAttentionAbstract)
    )


@pytest.mark.parametrize(
    "package_name", [name for name in PACKAGES if "expert" in name]
)
@pytest.mark.parametrize("adaptation_steps", [None, 1])
def test_soft_expert_packages_preserve_causality_samples_and_checkpoints(
    package_name, adaptation_steps
):
    from emperor.experts import SoftMoEConfig
    from tests.unit.test_expert_soft_moe import scoring_config

    config = configuration(
        package_name,
        attention_soft_moe_config=SoftMoEConfig(
            num_experts=2, slots_per_expert=2, model_config=scoring_config()
        ),
        attn_source_memory_test_time_training_learning_rate=0.01
        if adaptation_steps
        else None,
        attn_source_memory_test_time_training_num_inner_steps=adaptation_steps,
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
                for p in module.source_memory.parameters()
            )
