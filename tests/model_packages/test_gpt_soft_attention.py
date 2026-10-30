"""Causal Soft MoE attention in both expert GPT model packages."""

import copy

import pytest
import torch

from emperor.attention._variants.mixture.layer import MixtureOfAttentionHeads
from emperor.attention._variants.mixture.soft import (
    SoftMixtureOfAttentionHeadsProjector,
)
from emperor.augmentations.adaptive_parameters import AdditiveDynamicBiasConfig
from emperor.experts import SoftMoEConfig
from models.catalog import model_package
from tests.model_packages.test_gpt_hierarchical_embedding import TOKENS
from tests.unit.test_expert_soft_moe import scoring_config

GPT_EXPERT_PACKAGES = ("gpt/expert_linear", "gpt/expert_linear_adaptive")


def configuration(package_name, *, soft_attention=True, **overrides):
    attention_routing = None
    if soft_attention:
        scoring_stack = scoring_config(layers=3)
        attention_routing = SoftMoEConfig(
            num_experts=2, slots_per_expert=2, model_config=scoring_stack
        )
    values = dict(
        input_dim=len(TOKENS),
        output_dim=len(TOKENS),
        batch_size=2,
        sequence_length=6,
        hidden_dim=8,
        stack_num_layers=2,
        stack_dropout_probability=0.0,
        embedding_dropout_probability=0.0,
        attn_num_heads=2,
        num_experts=3,
        top_k=2,
        capacity_factor=0.0,
        expert_stack_hidden_dim=8,
        router_stack_hidden_dim=8,
        ff_stack_hidden_dim=8,
        attention_soft_moe_config=attention_routing,
    )
    if "adaptive" in package_name:
        values["adaptive_generator_stack_hidden_dim"] = 8
        values["bias_option_flag"] = True
        values["bias_option"] = AdditiveDynamicBiasConfig
    values.update(overrides)
    package = model_package(package_name)
    return package.build_configuration(config_overrides=values)


@pytest.mark.parametrize("package_name", GPT_EXPERT_PACKAGES)
@pytest.mark.parametrize("soft_attention", [False, True])
def test_gpt_attention_option_is_opt_in_and_preserves_sparse_feed_forward(
    package_name, soft_attention
):
    package = model_package(package_name)
    model_config = configuration(package_name, soft_attention=soft_attention)
    model = package.build_model(model_config).eval()
    attention_modules = [
        module
        for module in model.modules()
        if isinstance(module, MixtureOfAttentionHeads)
    ]
    assert len(attention_modules) == 2
    for attention in attention_modules:
        assert attention.causal_attention_mask_flag
        assert (
            isinstance(attention.projector, SoftMixtureOfAttentionHeadsProjector)
            == soft_attention
        )
        if soft_attention:
            assert len(attention.projector.query_model) == 2
            assert len(attention.projector.output_model) == 2
            assert attention.projector.soft_router.num_experts == 2
    feed_forward_models = [
        module
        for name, module in model.named_modules()
        if "feed_forward" in name and hasattr(module, "num_experts")
    ]
    assert feed_forward_models
    assert all(module.num_experts == 3 for module in feed_forward_models)
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    logits, auxiliary_loss = model(input_ids)
    assert logits.shape == (2, 4, len(TOKENS))
    assert torch.isfinite(logits).all() and torch.isfinite(auxiliary_loss).all()


@pytest.mark.parametrize("package_name", GPT_EXPERT_PACKAGES)
def test_gpt_soft_attention_is_causal_and_all_attention_components_receive_gradients(
    package_name,
):
    torch.manual_seed(83)
    package = model_package(package_name)
    model = package.build_model(configuration(package_name)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 0]])
    logits, auxiliary_loss = model(input_ids, attention_mask=attention_mask)
    changed_future = input_ids.clone()
    changed_future[:, 3] = 9
    changed_logits, _ = model(changed_future, attention_mask=attention_mask)
    prefix_logits, _ = model(input_ids[:, :3], attention_mask=attention_mask[:, :3])
    torch.testing.assert_close(
        logits[:, :3], changed_logits[:, :3], atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(logits[:, :3], prefix_logits, atol=2e-5, rtol=2e-5)
    total_loss = logits.square().mean() + auxiliary_loss.sum()
    total_loss.backward()
    for attention in model.modules():
        if not isinstance(attention, MixtureOfAttentionHeads):
            continue
        for component in (
            attention.projector.soft_router,
            attention.projector.query_model,
            attention.projector.key_model,
            attention.projector.value_model,
            attention.projector.output_model,
        ):
            gradients = [parameter.grad for parameter in component.parameters()]
            assert gradients and all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in gradients
            )
            assert any(gradient.abs().sum() > 0 for gradient in gradients)
        assert attention.projector.combine_weights is None


@pytest.mark.parametrize("package_name", GPT_EXPERT_PACKAGES)
def test_gpt_soft_attention_checkpoint_and_greedy_generation(package_name):
    package = model_package(package_name)
    model_config = configuration(package_name)
    model = package.build_model(model_config).eval()
    prompt = torch.tensor([[1, 2], [3, 4]])
    expected = prompt
    with torch.no_grad():
        for _ in range(2):
            logits, _ = model(expected)
            next_tokens = logits[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat((expected, next_tokens), dim=1)
    generated = model.generate(prompt, max_new_tokens=2)
    torch.testing.assert_close(generated, expected)
    restored_config = copy.deepcopy(model_config)
    restored = package.build_model(restored_config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored.generate(prompt, max_new_tokens=2), generated)


@pytest.mark.parametrize("package_name", GPT_EXPERT_PACKAGES)
def test_gpt_soft_attention_composes_with_reasoning_growth_and_available_embeddings(
    package_name,
):
    package = model_package(package_name)
    embedding_options = {}
    if "adaptive" in package_name:
        embedding_options = dict(
            hierarchical_embedding_flag=True,
            hierarchical_embedding_max_token_bytes=8,
            lm_head_weight_tying_flag=False,
        )
    model_config = configuration(
        package_name,
        recurrent_flag=True,
        recurrent_initial_iterations=2,
        recurrent_max_steps=3,
        recurrent_no_gradient_transition_count=0,
        recurrent_iteration_increment=1,
        recurrent_forward_calls_before_iteration_increment=2,
        recurrent_smooth_iteration_growth_flag=True,
        **embedding_options,
    )
    model = package.build_model(model_config).eval()
    if embedding_options:
        model.set_token_vocabulary(TOKENS)
    schedule = model.transformer.recurrent_iteration_schedule
    assert schedule.active_iterations == 2
    input_ids = torch.tensor([[1, 2, 3], [3, 4, 5]])
    for _ in range(3):
        logits, auxiliary_loss = model(input_ids)
    assert schedule.active_iterations == 3
    total_loss = logits.square().mean() + auxiliary_loss.sum()
    total_loss.backward()
    assert torch.isfinite(total_loss)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.token_embedding.parameters()
    )


@pytest.mark.parametrize("package_name", GPT_EXPERT_PACKAGES)
def test_gpt_runtime_defaults_accept_soft_attention_and_reject_expert_keys_values(
    package_name,
):
    package = model_package(package_name)
    scoring_stack = scoring_config(layers=2)
    routing = SoftMoEConfig(
        num_experts=2, slots_per_expert=2, model_config=scoring_stack
    )
    assert (
        package.bind_runtime_defaults({"attention_soft_moe_config": routing})
        is not None
    )
    invalid_config = configuration(
        package_name, expert_attention_use_kv_expert_models_flag=True
    )
    with pytest.raises(ValueError, match="expert key/value projections"):
        package.build_model(invalid_config)
