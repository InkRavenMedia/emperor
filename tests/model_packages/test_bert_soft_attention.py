"""Independent Soft MoE selection for BERT attention and feed-forward layers."""

import copy

import pytest
import torch

from emperor.attention._variants.mixture.layer import MixtureOfAttentionHeads
from emperor.attention._variants.mixture.soft import (
    SoftMixtureOfAttentionHeadsProjector,
)
from emperor.experts import SoftMoEConfig
from emperor.experts._routing.soft import SoftMoERouter
from models.catalog import model_package
from tests.model_packages.test_bert_hierarchical_soft_moe import (
    TOKEN_TEXTS,
    configuration,
    inputs,
)
from tests.unit.test_expert_soft_moe import scoring_config

EXPERT_BERT_PACKAGES = ("bert/expert_linear", "bert/expert_linear_adaptive")


def attention_routing_config():
    scoring_stack = scoring_config(layers=3)
    return SoftMoEConfig(num_experts=3, slots_per_expert=2, model_config=scoring_stack)


@pytest.mark.parametrize("package_name", EXPERT_BERT_PACKAGES)
@pytest.mark.parametrize(
    "soft_attention,soft_feed_forward",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_attention_and_feed_forward_routing_are_independent(
    package_name, soft_attention, soft_feed_forward
):
    attention_config = None
    if soft_attention:
        attention_config = attention_routing_config()
    model_config = configuration(
        package_name,
        hierarchical=False,
        soft=soft_feed_forward,
        attention_soft_moe_config=attention_config,
    )
    package = model_package(package_name)
    model = package.build_model(model_config).eval()
    attention = next(
        module
        for module in model.modules()
        if isinstance(module, MixtureOfAttentionHeads)
    )
    assert (
        isinstance(attention.projector, SoftMixtureOfAttentionHeadsProjector)
        == soft_attention
    )
    feed_forward_routers = [
        module
        for name, module in model.named_modules()
        if "feed_forward" in name and isinstance(module, SoftMoERouter)
    ]
    assert bool(feed_forward_routers) == soft_feed_forward
    if soft_attention:
        attention_router = attention.projector.soft_router
        assert attention_router.num_experts == 3
        assert attention_router.slot_logit_model.cfg.generator_depth == 3
        attention_parameters = {
            id(parameter) for parameter in attention_router.parameters()
        }
        for router in feed_forward_routers:
            assert router.num_experts == 2
            assert attention_parameters.isdisjoint(
                id(parameter) for parameter in router.parameters()
            )
        assert attention_config.model_config.generator_depth is None
        assert attention_config.model_config.model_config.input_dim is None
    logits, sentence_logits, loss = model(*inputs())
    assert logits.shape == (2, 4, len(TOKEN_TEXTS))
    assert sentence_logits.shape == (2, 2)
    total_loss = logits.square().mean() + sentence_logits.square().mean() + loss.sum()
    total_loss.backward()
    assert torch.isfinite(total_loss)
    for module in model.modules():
        if isinstance(module, SoftMoERouter):
            assert all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in module.parameters()
            )
    if soft_attention:
        for projections in (
            attention.projector.query_model,
            attention.projector.output_model,
            attention.projector.key_model,
            attention.projector.value_model,
        ):
            assert any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in projections.parameters()
            )


@pytest.mark.parametrize("package_name", EXPERT_BERT_PACKAGES)
def test_soft_attention_composes_with_byte_embeddings_and_reasoning_growth(
    package_name,
):
    routing_config = attention_routing_config()
    model_config = configuration(
        package_name,
        attention_soft_moe_config=routing_config,
        recurrent_flag=True,
        recurrent_initial_iterations=2,
        recurrent_max_steps=3,
        recurrent_no_gradient_transition_count=0,
        recurrent_iteration_increment=1,
        recurrent_forward_calls_before_iteration_increment=2,
        recurrent_smooth_iteration_growth_flag=True,
    )
    package = model_package(package_name)
    model = package.build_model(model_config).eval()
    model.set_token_vocabulary(TOKEN_TEXTS)
    schedule = model.transformer.recurrent_iteration_schedule
    assert schedule.active_iterations == 2
    for _ in range(3):
        outputs = model(*inputs())
    assert schedule.active_iterations == 3
    total_loss = (
        outputs[0].square().mean() + outputs[1].square().mean() + outputs[2].sum()
    )
    total_loss.backward()
    assert torch.isfinite(total_loss)
    restored_config = copy.deepcopy(model_config)
    restored = package.build_model(restored_config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    for original_output, restored_output in zip(
        model(*inputs()), restored(*inputs()), strict=True
    ):
        torch.testing.assert_close(original_output, restored_output)
    assert (
        restored.transformer.recurrent_iteration_schedule.snapshot()
        == schedule.snapshot()
    )


@pytest.mark.parametrize("package_name", EXPERT_BERT_PACKAGES)
def test_runtime_defaults_accept_attention_soft_moe_configuration(package_name):
    package = model_package(package_name)
    routing_config = attention_routing_config()
    assert (
        package.bind_runtime_defaults({"attention_soft_moe_config": routing_config})
        is not None
    )


@pytest.mark.parametrize("package_name", EXPERT_BERT_PACKAGES)
def test_bert_soft_attention_rejects_expert_key_value_configuration(package_name):
    routing_config = attention_routing_config()
    model_config = configuration(
        package_name,
        hierarchical=False,
        attention_soft_moe_config=routing_config,
        expert_attention_use_kv_expert_models_flag=True,
    )
    with pytest.raises(ValueError, match="expert key/value projections"):
        model_package(package_name).build_model(model_config)
