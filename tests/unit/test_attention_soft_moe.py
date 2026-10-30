"""Independent numerical and lifecycle contracts for Soft MoE attention."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from emperor.attention import MixtureOfAttentionHeadsConfig
from emperor.attention._monitoring.callback import _AttentionDiagnosticsTrackerManager
from emperor.attention._monitoring.diagnostics import (
    _AttentionDiagnostics,
    _resolve_attention_monitor_adapter,
)
from emperor.embedding.relative import DynamicPositionalBiasConfig
from emperor.layers import LayerState
from tests.support.attention import build_attention_config
from tests.unit.test_expert_soft_moe import expert_config


def attention_config(*, scoring_layers=2, experts=3, slots=2, **overrides):
    configuration = build_attention_config(
        MixtureOfAttentionHeadsConfig,
        batch_size=3,
        num_heads=2,
        embedding_dim=8,
        query_key_projection_dim=6,
        value_projection_dim=4,
        target_sequence_length=5,
        source_sequence_length=5,
        use_kv_expert_models_flag=False,
    )
    configuration.batch_first_flag = True
    configuration.experts_config = expert_config(
        dimension=8,
        output_dim=8,
        experts=experts,
        slots=slots,
        scoring_layers=scoring_layers,
    )
    for option_name, option_value in overrides.items():
        setattr(configuration, option_name, option_value)
    return configuration


def reference_stack(stack, features, *, expert_index=None):
    hidden = features
    for layer_index, layer in enumerate(stack.layers):
        weights = layer.model.weight_params
        bias = layer.model.bias_params
        if expert_index is not None:
            weights = weights[expert_index]
            if bias is not None:
                bias = bias[expert_index]
        hidden = hidden @ weights
        if bias is not None:
            hidden = hidden + bias
        if layer_index < len(stack.layers) - 1:
            hidden = F.gelu(hidden)
    return hidden


def explicit_reference(model, inputs, padding_mask):
    """Loop over experts, slots, and heads without production routing/forwards."""
    projector = model.projector
    router = projector.soft_router
    active_tokens = ~padding_mask
    masked_inputs = inputs.masked_fill(padding_mask.unsqueeze(-1), 0)
    unit_inputs = (
        masked_inputs / (masked_inputs.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    )
    expert_logits = []
    for expert_index in range(router.num_experts):
        logits = reference_stack(
            router.slot_logit_model.model, unit_inputs, expert_index=expert_index
        )
        expert_logits.append(logits * router.scale)
    token_slot_logits = torch.stack(expert_logits, dim=2)
    slot_mask = active_tokens[:, :, None, None]
    dispatch_logits = token_slot_logits.masked_fill(~slot_mask, -torch.inf)
    dispatch_logits = torch.where(slot_mask.any(1, keepdim=True), dispatch_logits, 0)
    dispatch_weights = dispatch_logits.softmax(1).masked_fill(~slot_mask, 0)
    combine_weights = (
        token_slot_logits.flatten(2).softmax(-1).reshape_as(token_slot_logits)
    )
    combine_weights = combine_weights.masked_fill(~slot_mask, 0)
    keys = reference_stack(projector.key_model, masked_inputs)
    values = reference_stack(projector.value_model, masked_inputs)
    source_mask = padding_mask
    if model.add_key_value_bias_flag:
        key_bias = model.bias.key_bias_vector.expand(inputs.size(0), -1, -1)
        value_bias = model.bias.value_bias_vector.expand(inputs.size(0), -1, -1)
        keys = torch.cat((keys, key_bias), dim=1)
        values = torch.cat((values, value_bias), dim=1)
        source_mask = F.pad(source_mask, (0, 1))
    if model.zero_attention_flag:
        keys = torch.cat((keys, torch.zeros_like(keys[:, :1])), dim=1)
        values = torch.cat((values, torch.zeros_like(values[:, :1])), dim=1)
        source_mask = F.pad(source_mask, (0, 1))
    query_head_width = keys.size(-1) // model.num_heads
    value_head_width = values.size(-1) // model.num_heads
    output_slots = []
    for expert_index in range(router.num_experts):
        expert_slots = []
        for slot_index in range(router.slots_per_expert):
            token_weights = dispatch_weights[:, :, expert_index, slot_index, None]
            pooled_tokens = (masked_inputs * token_weights).sum(1)
            query = reference_stack(projector.query_model[expert_index], pooled_tokens)
            head_outputs = []
            for head_index in range(model.num_heads):
                query_slice = slice(
                    head_index * query_head_width, (head_index + 1) * query_head_width
                )
                value_slice = slice(
                    head_index * value_head_width, (head_index + 1) * value_head_width
                )
                logits = (query[:, None, query_slice] * keys[:, :, query_slice]).sum(-1)
                logits = logits / query_head_width**0.5
                logits = logits.masked_fill(source_mask, -torch.inf)
                empty_rows = source_mask.all(-1, keepdim=True)
                safe_logits = logits.masked_fill(empty_rows, 0)
                attention_weights = safe_logits.softmax(-1).masked_fill(empty_rows, 0)
                head_output = (
                    attention_weights.unsqueeze(-1) * values[:, :, value_slice]
                ).sum(1)
                head_outputs.append(head_output)
            attended_slot = torch.cat(head_outputs, dim=-1)
            output_slot = reference_stack(
                projector.output_model[expert_index], attended_slot
            )
            expert_slots.append(output_slot)
        output_slots.append(torch.stack(expert_slots, dim=1))
    token_outputs = torch.zeros_like(inputs)
    for expert_index, expert_slots in enumerate(output_slots):
        for slot_index in range(router.slots_per_expert):
            token_weights = combine_weights[:, :, expert_index, slot_index, None]
            token_outputs = (
                token_outputs + token_weights * expert_slots[:, None, slot_index]
            )
    return token_outputs


def explicit_prefix_reference(model, inputs, padding_mask):
    prefix_outputs = []
    for prefix_length in range(1, inputs.shape[1] + 1):
        prefix_inputs = inputs[:, :prefix_length]
        prefix_padding_mask = padding_mask[:, :prefix_length]
        prefix_output = explicit_reference(model, prefix_inputs, prefix_padding_mask)
        prefix_outputs.append(prefix_output[:, -1])
    return torch.stack(prefix_outputs, dim=1)


@pytest.mark.parametrize("scoring_layers", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_attention_and_gradients_match_independent_reference(
    scoring_layers, dtype
):
    torch.manual_seed(24)
    configuration = attention_config(scoring_layers=scoring_layers, target_dtype=dtype)
    model = configuration.build().eval()
    inputs = torch.randn(2, 5, 8, dtype=dtype, requires_grad=True)
    padding_mask = torch.tensor([[False] * 5, [False, False, True, True, True]])
    output, weights, loss = model(inputs, inputs, inputs, padding_mask)
    expected = explicit_reference(model, inputs, padding_mask)
    torch.testing.assert_close(output, expected)
    assert weights is None and loss.item() == 0
    coefficients = torch.randn_like(output)
    parameters = (inputs, *tuple(model.parameters()))
    actual_gradients = torch.autograd.grad(
        (output * coefficients).sum(), parameters, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        (expected * coefficients).sum(), parameters
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=2e-6, rtol=2e-4
        )
        assert torch.isfinite(actual_gradient).all()
        assert actual_gradient.abs().sum() > 0


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("additive_mask", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_padding_batch_independence_and_runtime_layout(
    batch_first, additive_mask, causal
):
    configuration = attention_config(
        batch_first_flag=batch_first, causal_attention_mask_flag=causal
    )
    model = configuration.build().eval()
    batch_inputs = torch.randn(2, 4, 16)[..., ::2]
    padding_mask = torch.tensor([[False, False, True, True], [True] * 4])
    attention_padding_mask = padding_mask
    if additive_mask:
        attention_padding_mask = torch.zeros_like(padding_mask, dtype=torch.float32)
        attention_padding_mask.masked_fill_(padding_mask, -torch.inf)
    inputs = batch_inputs
    if not batch_first:
        inputs = batch_inputs.transpose(0, 1)
    output = model(inputs, inputs, inputs, attention_padding_mask)[0]
    batch_output = output
    if not batch_first:
        batch_output = output.transpose(0, 1)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(batch_output[padding_mask]) == 0
    changed_batch = batch_inputs.clone()
    changed_batch[padding_mask] = float("nan")
    changed_inputs = changed_batch
    if not batch_first:
        changed_inputs = changed_batch.transpose(0, 1)
    changed_output = model(
        changed_inputs, changed_inputs, changed_inputs, attention_padding_mask
    )[0]
    torch.testing.assert_close(output, changed_output)
    for batch_index in range(2):
        sample = batch_inputs[batch_index : batch_index + 1]
        if not batch_first:
            sample = sample.transpose(0, 1)
        sample_mask = attention_padding_mask[batch_index : batch_index + 1]
        sample_output = model(sample, sample, sample, sample_mask)[0]
        if not batch_first:
            sample_output = sample_output.transpose(0, 1)
        torch.testing.assert_close(sample_output[0], batch_output[batch_index])
    output.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    "add_bias,add_zero", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("causal", [False, True])
def test_source_extensions_match_reference(add_bias, add_zero, causal):
    configuration = attention_config(
        add_key_value_bias_flag=add_bias,
        zero_attention_flag=add_zero,
        target_dtype=torch.float64,
        causal_attention_mask_flag=causal,
    )
    model = configuration.build().eval()
    inputs = torch.randn(2, 4, 8, dtype=torch.float64)
    padding_mask = torch.tensor([[False, False, False, True], [True] * 4])
    expected = explicit_reference(model, inputs, padding_mask)
    if causal:
        expected = explicit_prefix_reference(model, inputs, padding_mask)
    output = model(inputs, inputs, inputs, padding_mask)[0]
    torch.testing.assert_close(output, expected)
    assert torch.count_nonzero(output[padding_mask]) == 0
    output.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("causal", [False, True])
def test_routing_and_shared_projections_run_once_and_transient_state_is_cleared(causal):
    model = attention_config(causal_attention_mask_flag=causal).build()
    calls = []
    handles = []
    for name in ("soft_router", "key_model", "value_model"):
        module = getattr(model.projector, name)
        handle = module.register_forward_hook(
            lambda module, arguments, output, name=name: calls.append(name)
        )
        handles.append(handle)
    inputs = torch.randn(2, 4, 8)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        model(inputs, inputs, inputs)[0].sum().backward()
        assert model.projector.combine_weights is None
        assert model.projector.auxiliary_loss is None
    assert calls == ["soft_router", "key_model", "value_model"] * 2
    for handle in handles:
        handle.remove()

    def fail_projection(module, arguments):
        raise RuntimeError("deliberate projection failure")

    handle = model.projector.value_model.register_forward_pre_hook(fail_projection)
    with pytest.raises(RuntimeError, match="deliberate projection failure"):
        model(inputs, inputs, inputs)
    handle.remove()
    assert model.projector.combine_weights is None
    assert model.projector.auxiliary_loss is None
    assert torch.isfinite(model(inputs, inputs, inputs)[0]).all()


@pytest.mark.parametrize("experts,slots", [(1, 1), (3, 7)])
@pytest.mark.parametrize("causal", [False, True])
def test_checkpoint_round_trip_and_unbatched_inputs(experts, slots, causal):
    configuration = attention_config(
        experts=experts, slots=slots, causal_attention_mask_flag=causal
    )
    model = configuration.build().eval()
    inputs = torch.randn(3, 8)
    output = model(inputs, inputs, inputs)[0]
    assert output.shape == inputs.shape
    restored = copy.deepcopy(configuration).build().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(output, restored(inputs, inputs, inputs)[0])
    assert not any("combine_weights" in name for name in model.state_dict())


class ProjectionWithLoss(nn.Module):
    def __init__(self, projection, loss_value):
        super().__init__()
        self.projection = projection
        self.loss_value = loss_value

    def forward(self, inputs):
        state = self.projection(inputs)
        return LayerState(
            hidden=state.hidden, loss=state.hidden.new_tensor(self.loss_value)
        )


@pytest.mark.parametrize("causal", [False, True])
def test_projection_auxiliary_losses_are_accumulated_once(causal):
    model = attention_config(causal_attention_mask_flag=causal).build()
    for expert_index in range(3):
        model.projector.query_model[expert_index] = ProjectionWithLoss(
            model.projector.query_model[expert_index], 1
        )
        model.projector.output_model[expert_index] = ProjectionWithLoss(
            model.projector.output_model[expert_index], 2
        )
    model.projector.key_model = ProjectionWithLoss(model.projector.key_model, 4)
    model.projector.value_model = ProjectionWithLoss(model.projector.value_model, 8)
    inputs = torch.randn(2, 4, 8)
    for _ in range(2):
        assert model(inputs, inputs, inputs)[2].item() == 21


@pytest.mark.parametrize(
    "options,message",
    [
        ({"use_kv_expert_models_flag": True}, "expert key/value"),
        ({"batch_first_flag": None}, "explicit batch_first_flag"),
        ({"return_attention_weights_flag": True}, "returning attention_weights"),
        (
            {
                "relative_positional_embedding_config": DynamicPositionalBiasConfig(
                    num_heads=2, embedding_dim=6, max_positions=8
                )
            },
            "relative positions",
        ),
    ],
)
def test_unsupported_configuration_has_specific_error(options, message):
    configuration = attention_config(**options)
    with pytest.raises(ValueError, match=message):
        configuration.build()


def test_unsupported_runtime_inputs_have_specific_errors():
    model = attention_config().build()
    inputs = torch.randn(2, 4, 8)
    with pytest.raises(ValueError, match="self-attention"):
        model(inputs, inputs.clone(), inputs)
    for static_option in ("static_k", "static_v"):
        with pytest.raises(ValueError, match="static key/value"):
            model(inputs, inputs, inputs, **{static_option: torch.zeros(4, 4, 3)})
    with pytest.raises(ValueError, match="token-pair"):
        model(inputs, inputs, inputs, attention_mask=torch.zeros(4, 4))
    for invalid_value in (-0.5, 1.0, float("nan"), float("inf")):
        invalid_mask = torch.full((2, 4), invalid_value)
        with pytest.raises(ValueError, match="0/-inf"):
            model(inputs, inputs, inputs, invalid_mask)


@pytest.mark.parametrize("causal", [False, True])
def test_cpu_bfloat16_autocast_preserves_finite_outputs_and_gradients(causal):
    model = attention_config(causal_attention_mask_flag=causal).build()
    inputs = torch.randn(2, 4, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(inputs, inputs, inputs)[0]
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    output.float().square().sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_monitoring_captures_slot_attention_and_preserves_dropout(training, causal):
    configuration = attention_config(
        dropout_probability=0.5, causal_attention_mask_flag=causal
    )
    model = configuration.build().train(training)
    inputs = torch.randn(2, 4, 8)
    torch.manual_seed(63)
    unmonitored_output = model(inputs, inputs, inputs)[0]
    observations = []
    manager = _AttentionDiagnosticsTrackerManager()
    manager.attach(
        "attention",
        model,
        lambda: True,
        lambda name, module, observation: observations.append(observation),
    )
    try:
        torch.manual_seed(63)
        monitored_output = model(inputs, inputs, inputs)[0]
        torch.testing.assert_close(monitored_output, unmonitored_output)
        assert len(observations) == 1
        observation = observations[0]
        query_count = 2
        if causal:
            query_count = 8
        assert observation.projected_inputs.query.shape == (query_count, 2, 3, 6)
        assert observation.exact_attention_weights.shape == (12, query_count, 4)
        assert observation.restored_output.shape == inputs.shape
        assert not observation.exact_attention_weights.requires_grad
        normalized_weights = observation.normalized_attention_weights
        exact_weights = observation.exact_attention_weights
        if training:
            retained_weights = exact_weights != 0
            assert retained_weights.any() and not retained_weights.all()
            torch.testing.assert_close(
                exact_weights[retained_weights],
                2 * normalized_weights[retained_weights],
            )
        else:
            torch.testing.assert_close(exact_weights, normalized_weights)
        adapter = _resolve_attention_monitor_adapter(model)
        metrics = _AttentionDiagnostics(adapter).calculate(
            observation, num_heads=2, configured_dropout_probability=0.5
        )
        assert metrics.weight_source == "exact"
        assert metrics.per_head_entropy.shape == (2,)
        assert torch.isfinite(metrics.per_head_entropy).all()
        assert model.projector.combine_weights is None
    finally:
        manager.detach()
    assert manager.hook_count == manager.subscription_count == 0
    assert "compute_qkv_projections" not in vars(model.projector)
    assert "compute_attention" not in vars(model.processor)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scoring_layers", [1, 3])
def test_causal_attention_and_gradients_match_independent_prefix_reference(
    dtype, scoring_layers
):
    torch.manual_seed(82)
    configuration = attention_config(
        target_dtype=dtype,
        scoring_layers=scoring_layers,
        causal_attention_mask_flag=True,
    )
    model = configuration.build().eval()
    inputs = torch.randn(3, 5, 8, dtype=dtype, requires_grad=True)
    padding_mask = torch.tensor(
        [
            [False, False, False, True, True],
            [True, True, False, False, False],
            [True, True, True, True, True],
        ]
    )
    output = model(inputs, inputs, inputs, padding_mask)[0]
    expected = explicit_prefix_reference(model, inputs, padding_mask)
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-4)
    assert torch.count_nonzero(output[padding_mask]) == 0
    coefficients = torch.randn_like(output)
    parameters = (inputs, *tuple(model.parameters()))
    actual_gradients = torch.autograd.grad(
        (output * coefficients).sum(), parameters, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        (expected * coefficients).sum(), parameters
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=2e-6, rtol=3e-4
        )


def test_causal_prefixes_ignore_future_tokens_and_future_input_gradients():
    configuration = attention_config(
        causal_attention_mask_flag=True, target_dtype=torch.float64
    )
    model = configuration.build().eval()
    inputs = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    output = model(inputs, inputs, inputs)[0]
    for prefix_length in range(1, 6):
        prefix = inputs[:, :prefix_length]
        prefix_output = model(prefix, prefix, prefix)[0]
        torch.testing.assert_close(output[:, :prefix_length], prefix_output)
        changed_future = inputs.detach().clone()
        changed_future[:, prefix_length:] *= -1000
        changed_output = model(changed_future, changed_future, changed_future)[0]
        torch.testing.assert_close(
            output[:, :prefix_length], changed_output[:, :prefix_length]
        )
    earlier_output_gradients = torch.autograd.grad(
        output[:, :3].square().sum(), inputs
    )[0]
    assert torch.count_nonzero(earlier_output_gradients[:, 3:]) == 0
    assert torch.count_nonzero(earlier_output_gradients[:, :3]) > 0


def test_causal_pooling_handles_future_logits_that_would_underflow_global_dispatch():
    configuration = attention_config(
        causal_attention_mask_flag=True, experts=1, slots=1, scoring_layers=1
    )
    model = configuration.build().eval()
    scoring_layer = model.projector.soft_router.slot_logit_model.model.layers[0].model
    with torch.no_grad():
        scoring_layer.weight_params.zero_()
        scoring_layer.weight_params[:, 0, :] = 10000
        scoring_layer.bias_params.zero_()
    inputs = torch.zeros(1, 3, 8, requires_grad=True)
    with torch.no_grad():
        inputs[0, :, 0] = torch.tensor([-1, -1, 1])
        inputs[0, 1, 1] = 0.5
    output = model(inputs, inputs, inputs)[0]
    padding_mask = torch.zeros(1, 3, dtype=torch.bool)
    expected = explicit_prefix_reference(model, inputs, padding_mask)
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-4)
    output.sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
