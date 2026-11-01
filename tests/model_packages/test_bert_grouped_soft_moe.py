"""Five-token adaptive generation throughout BERT Soft MoE attention and MLPs."""

import math

import pytest
import torch
import torch.nn.functional as F

from emperor.attention._variants.mixture.soft import (
    SoftMixtureOfAttentionHeadsProjector,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    CombinedDynamicDiagonalConfig,
    DualModelDynamicWeightConfig,
    GeneratorDynamicBiasConfig,
    SumGroupingConfig,
)
from emperor.augmentations.adaptive_parameters._linear_adapter import (
    AdaptiveLinearLayer,
)
from emperor.halting import StickBreakingConfig
from models.catalog import model_package
from tests.model_packages.test_bert_hierarchical_soft_moe import (
    TOKEN_TEXTS,
    configuration,
)
from tests.model_packages.test_bert_soft_attention import attention_routing_config


def grouped_model():
    config = configuration(
        "bert/expert_linear_adaptive",
        hierarchical=False,
        sequence_length=100,
        attention_soft_moe_config=attention_routing_config(),
        attn_source_memory_flag=True,
        expert_attention_use_kv_expert_models_flag=False,
        grouping_scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
        chunk_size=5,
        grouping_sequence_length=100,
        grouping_input_order=AdaptiveParameterInputOrderOptions.BATCH_FIRST,
        grouping_method=SumGroupingConfig,
        attn_grouping_scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
        attn_chunk_size=5,
        attn_grouping_sequence_length=100,
        attn_grouping_input_order=AdaptiveParameterInputOrderOptions.SEQUENCE_FIRST,
        attn_grouping_method=SumGroupingConfig,
        recurrent_flag=True,
        recurrent_initial_iterations=2,
        recurrent_max_steps=3,
        recurrent_forward_calls_before_iteration_increment=20_000,
        recurrent_min_steps=2,
        recurrent_stack_halting_flag=True,
        recurrent_halting_option=StickBreakingConfig,
        recurrent_ponder_cost_weight=0.01,
        weight_option_flag=True,
        weight_option=DualModelDynamicWeightConfig,
        bias_option_flag=True,
        bias_option=GeneratorDynamicBiasConfig,
        diagonal_option_flag=True,
        diagonal_option=CombinedDynamicDiagonalConfig,
        mask_option_flag=False,
    )
    return model_package("bert/expert_linear_adaptive").build_model(config)


def token_batch():
    token_ids = torch.randint(5, len(TOKEN_TEXTS), (2, 100))
    active_tokens = torch.ones_like(token_ids)
    active_tokens[0, 93:] = 0
    active_tokens[1, 87:] = 0
    token_types = torch.zeros_like(token_ids)
    token_types[:, 50:] = 1
    return token_ids, active_tokens, token_types


@pytest.mark.training
def test_every_adaptive_layer_groups_and_trains_with_soft_attention_and_memory():
    torch.manual_seed(25)
    model = grouped_model().train()
    adaptive_layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, AdaptiveLinearLayer)
    }
    assert adaptive_layers
    observed_context_counts = {name: [] for name in adaptive_layers}
    expected_sums = {}
    handles = []
    for name, layer in adaptive_layers.items():
        grouping = layer.adaptive_behaviour.grouping_config
        assert isinstance(grouping, SumGroupingConfig)
        assert grouping.chunk_size == 5
        expected_contexts = 2 * math.ceil(grouping.sequence_length / 5)

        def record_input(_module, arguments, name=name, grouping=grouping):
            rows = arguments[0].detach()
            sequence_length = grouping.sequence_length
            if (
                grouping.input_order
                is AdaptiveParameterInputOrderOptions.SEQUENCE_FIRST
            ):
                tokens = rows.reshape(sequence_length, 2, -1).transpose(0, 1)
            else:
                tokens = rows.reshape(2, sequence_length, -1)
            padding_length = (-sequence_length) % 5
            padded_tokens = F.pad(tokens, (0, 0, 0, padding_length))
            expected_sums[name] = padded_tokens.reshape(-1, 5, rows.shape[-1]).sum(1)

        handles.append(layer.register_forward_pre_hook(record_input))

        def record_parameters(
            _module, arguments, output, name=name, expected=expected_contexts
        ):
            assert arguments[1].shape == (expected, _module.input_dim)
            torch.testing.assert_close(arguments[1].detach(), expected_sums[name])
            assert output.shape[0] == expected
            observed_context_counts[name].append(expected)

        handles.append(
            layer.adaptive_behaviour.weight_model.register_forward_hook(
                record_parameters
            )
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    try:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            token_ids, active_tokens, token_types = token_batch()
            labels = torch.full_like(token_ids, -100)
            labels[:, 1:85:5] = token_ids[:, 1:85:5]
            token_ids[:, 1:85:5] = 4
            logits, sentence_logits, auxiliary_loss = model(
                token_ids, active_tokens, token_types
            )
            assert logits.shape == (2, 100, len(TOKEN_TEXTS))
            loss = F.cross_entropy(logits.transpose(1, 2), labels)
            loss = (
                loss
                + F.cross_entropy(sentence_logits, torch.tensor([0, 1]))
                + auxiliary_loss.sum()
            )
            assert torch.isfinite(loss)
            loss.backward()
            assert all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            for layer in adaptive_layers.values():
                assert any(
                    parameter.grad is not None and parameter.grad.abs().sum() > 0
                    for parameter in layer.parameters()
                )
            optimizer.step()
    finally:
        for handle in handles:
            handle.remove()
    assert all(observed_context_counts.values())
    assert {
        count for counts in observed_context_counts.values() for count in counts
    } == {2, 40}


def test_grouped_soft_bert_preserves_padding_and_sample_independence():
    torch.manual_seed(26)
    model = grouped_model().eval()
    token_ids, active_tokens, token_types = token_batch()
    with torch.no_grad():
        original = model(token_ids, active_tokens, token_types)
        changed_ids = token_ids.clone()
        changed_ids[active_tokens == 0] = 5
        changed = model(changed_ids, active_tokens, token_types)
        torch.testing.assert_close(
            original[0][active_tokens.bool()], changed[0][active_tokens.bool()]
        )
        torch.testing.assert_close(original[1], changed[1])
        for sample in range(2):
            separate = model(
                token_ids[sample : sample + 1],
                active_tokens[sample : sample + 1],
                token_types[sample : sample + 1],
            )
            torch.testing.assert_close(original[0][sample], separate[0][0])
            torch.testing.assert_close(original[1][sample], separate[1][0])
    projector = next(
        module
        for module in model.modules()
        if isinstance(module, SoftMixtureOfAttentionHeadsProjector)
    )
    assert projector.source_memory_flag
