"""Executable memory and expert choices shared by the four model families."""

import pytest
import torch

from emperor.experts import SoftMoEConfig
from emperor.experts._routing.soft import SoftMoERouter
from emperor.memory import AttentionDynamicMemoryConfig
from emperor.memory._base import DynamicMemoryAbstract
from emperor.transformer import FeedForward
from models.catalog import model_package
from tests.model_packages.test_attention_source_memory import (
    configuration as attention_configuration,
)
from tests.model_packages.test_attention_source_memory import (
    run_model,
)
from tests.model_packages.test_mlp_mixer_source_memory import (
    configuration as mixer_configuration,
)
from tests.unit.test_expert_soft_moe import scoring_config

VARIANTS = ("linear", "linear_adaptive", "expert_linear", "expert_linear_adaptive")
PACKAGES = tuple(
    f"{family}/{variant}"
    for family in ("bert", "gpt", "vit", "mlp_mixer")
    for variant in VARIANTS
)


def memory_roles(name):
    roles = (
        ["memory", "token_mixer_memory", "channel_mixer_memory"]
        if name.startswith("mlp_mixer")
        else ["memory", "attn_memory", "ff_memory"]
    )
    if "expert" in name:
        roles.append("expert_memory")
    if name.endswith("expert_linear_adaptive") and not name.startswith("mlp_mixer"):
        roles.append("router_memory")
    return roles


def soft_routing():
    return SoftMoEConfig(
        num_experts=2, slots_per_expert=2, model_config=scoring_config()
    )


def configuration(name, **updates):
    if name.startswith("mlp_mixer"):
        return mixer_configuration(name, token_mixer_source_memory_flag=True, **updates)
    return attention_configuration(name, **updates)


def inputs_for(name):
    if name.startswith(("vit", "mlp_mixer")):
        return torch.randn(2, 1, 8, 8)
    return torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])


@pytest.mark.parametrize("name", PACKAGES)
def test_soft_feed_forward_and_memory_work_together_in_every_family(name):
    torch.manual_seed(29)
    package = model_package(name)
    config = configuration(name, soft_moe_config=soft_routing())
    model = package.build_model(config).eval()
    feed_forwards = [m for m in model.modules() if isinstance(m, FeedForward)]
    assert feed_forwards and all(m.uses_soft_moe for m in feed_forwards)
    inputs = inputs_for(name)
    logits = run_model(model, name, inputs)[0]
    assert torch.isfinite(logits).all()
    changed_sample = inputs.clone()
    changed_sample[1] = 0
    torch.testing.assert_close(run_model(model, name, changed_sample)[0][0], logits[0])
    if name.startswith("gpt"):
        changed_future = inputs.clone()
        changed_future[:, 2:] = 6
        torch.testing.assert_close(
            run_model(model, name, changed_future)[0][:, :2], logits[:, :2]
        )
        torch.testing.assert_close(
            run_model(model, name, inputs[:, :2])[0], logits[:, :2]
        )
    logits.square().mean().backward()
    for feed_forward in feed_forwards:
        routers = [m for m in feed_forward.modules() if isinstance(m, SoftMoERouter)]
        assert routers
        for router in routers:
            assert any(
                p.grad is not None
                and torch.isfinite(p.grad).all()
                and p.grad.abs().sum() > 0
                for p in router.parameters()
            )
    memory_parameters = [
        p for key, p in model.named_parameters() if "source_memory" in key
    ]
    assert memory_parameters and any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in memory_parameters
    )
    restored = package.build_model(config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(run_model(restored, name, inputs)[0], logits)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("channel_soft", [False, True])
def test_mixer_soft_token_experts_keep_samples_separate_with_memory(
    variant, channel_soft
):
    name = f"mlp_mixer/{variant}"
    package = model_package(name)
    config = configuration(
        name,
        token_mixer_soft_moe_config=soft_routing(),
        soft_moe_config=soft_routing() if channel_soft else None,
    )
    model = package.build_model(config).eval()
    inputs = inputs_for(name)
    logits = run_model(model, name, inputs)[0]
    assert logits.shape == (2, 3) and torch.isfinite(logits).all()
    changed = inputs.clone()
    changed[1] += 20
    torch.testing.assert_close(run_model(model, name, changed)[0][0], logits[0])
    logits.square().sum().backward()
    routers = [m for m in model.modules() if isinstance(m, SoftMoERouter)]
    assert routers
    for router in routers:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in router.parameters()
        )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for key, p in model.named_parameters()
        if "source_memory" in key
    )
    restored = package.build_model(config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(run_model(restored, name, inputs)[0], logits)


@pytest.mark.parametrize(
    "name,role", [(name, role) for name in PACKAGES for role in memory_roles(name)]
)
@pytest.mark.parametrize("slots", [None, 3])
def test_attention_memory_slots_reach_every_supported_layer_role(name, role, slots):
    updates = {
        f"{role}_flag": True,
        f"{role}_option": AttentionDynamicMemoryConfig,
    }
    if slots is not None:
        updates[f"{role}_num_memory_slots"] = slots
    if name.startswith("mlp_mixer"):
        updates.update(token_mixer_stack_hidden_dim=8, channel_mixer_stack_hidden_dim=8)
    else:
        updates.update(attn_stack_hidden_dim=8, ff_stack_hidden_dim=8)
    package = model_package(name)
    model = package.build_model(configuration(name, **updates)).eval()
    memories = [
        module
        for path, module in model.named_modules()
        if isinstance(module, DynamicMemoryAbstract) and "source_memory" not in path
    ]
    assert memories
    assert all(
        isinstance(module.cfg, AttentionDynamicMemoryConfig)
        and module.num_memory_slots == (4 if slots is None else slots)
        for module in memories
    )
    logits = run_model(model, name, inputs_for(name))[0]
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    # Sparse routing can leave an expert unused in this particular batch.
    gradients = [
        p.grad for memory in memories for p in memory.parameters() if p.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize("variant", VARIANTS)
def test_mixer_recurrent_memory_uses_selected_type_and_inherits_slot_count(variant):
    name = f"mlp_mixer/{variant}"
    model = (
        model_package(name)
        .build_model(
            configuration(
                name,
                recurrent_flag=True,
                recurrent_initial_iterations=2,
                recurrent_max_steps=2,
                recurrent_memory_flag=True,
                memory_option=AttentionDynamicMemoryConfig,
                memory_num_memory_slots=3,
            )
        )
        .eval()
    )
    memories = [
        memory
        for memory in model.modules()
        if isinstance(memory, DynamicMemoryAbstract)
        and isinstance(memory.cfg, AttentionDynamicMemoryConfig)
    ]
    assert memories and all(memory.num_memory_slots == 3 for memory in memories)
    logits = run_model(model, name, inputs_for(name))[0]
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for memory in memories
        for parameter in memory.parameters()
    )






@pytest.mark.parametrize("name", PACKAGES)
def test_memory_and_expert_options_are_exposed_in_public_configuration(name):
    from model_runtime.inspection.schema import configuration_schema

    package = model_package(name)
    fields = {field.key for field in configuration_schema(package).fields}
    assert "SOFT_MOE_CONFIG" in fields
    prefix = (
        "TOKEN_MIXER_SOURCE_MEMORY_"
        if name.startswith("mlp_mixer")
        else "ATTN_SOURCE_MEMORY_"
    )
    assert fields.issuperset(
        prefix + suffix
        for suffix in (
            "FLAG",
            "OPTION",
            "NUM_MEMORY_SLOTS",
            "TEST_TIME_TRAINING_LEARNING_RATE",
            "TEST_TIME_TRAINING_NUM_INNER_STEPS",
        )
    )
    for role in memory_roles(name):
        key = f"{role}_num_memory_slots".upper()
        assert key in fields
        assert package.runtime_defaults_spec.parse_value(key, "3") == 3
    if name.startswith("mlp_mixer"):
        assert "TOKEN_MIXER_SOFT_MOE_CONFIG" in fields
    elif "expert" in name:
        assert "ATTENTION_SOFT_MOE_CONFIG" in fields




@pytest.mark.parametrize("variant", VARIANTS)
def test_gpt_soft_feed_forward_with_memory_recurrence_padding_and_generation(variant):
    name = f"gpt/{variant}"
    updates = dict(
        soft_moe_config=soft_routing(),
        recurrent_flag=True,
        recurrent_initial_iterations=2,
        recurrent_max_steps=2,
    )
    if "expert" in variant:
        updates["attention_soft_moe_config"] = soft_routing()
    package = model_package(name)
    model = package.build_model(configuration(name, **updates)).eval()
    # Non-padding IDs keep positional indices fixed when excluded content changes.
    ids = torch.tensor([[1, 2, 3, 4], [9, 5, 6, 9]])
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 0]])
    logits, loss = model(ids, attention_mask=mask)
    changed = ids.clone()
    changed[mask == 0] = 7
    changed[:, 3] = 8
    torch.testing.assert_close(
        model(changed, attention_mask=mask)[0][:, :3][mask[:, :3].bool()],
        logits[:, :3][mask[:, :3].bool()],
    )
    assert torch.isfinite(logits).all() and torch.isfinite(loss).all()
    (logits.square().mean() + loss.sum()).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for key, p in model.named_parameters()
        if "soft_router" in key
    )
    prompt = ids[:, :2]
    expected = prompt
    with torch.no_grad():
        for _ in range(2):
            prediction = model(expected)[0][:, -1].argmax(-1, keepdim=True)
            expected = torch.cat((expected, prediction), dim=1)
    torch.testing.assert_close(model.generate(prompt, max_new_tokens=2), expected)
