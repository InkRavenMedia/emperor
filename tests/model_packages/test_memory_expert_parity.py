"""Executable memory and expert choices shared by the four model families."""

import pytest
import torch

from emperor.memory import AttentionDynamicMemoryConfig
from emperor.memory._base import DynamicMemoryAbstract
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




def configuration(name, **updates):
    if name.startswith("mlp_mixer"):
        return mixer_configuration(name, token_mixer_source_memory_flag=True, **updates)
    return attention_configuration(name, **updates)


def inputs_for(name):
    if name.startswith(("vit", "mlp_mixer")):
        return torch.randn(2, 1, 8, 8)
    return torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])






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
