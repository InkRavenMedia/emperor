"""Behavioral contracts for temporary memory sources in token attention."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from emperor.attention import (
    AttentionSourceMemoryConfig,
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.attention._ops.source_memory import AttentionSourceMemory
from emperor.attention._runtime import AttentionRuntimeLayout, MultiHeadAttentionInputs
from emperor.augmentations.adaptive_parameters import SumGroupingConfig
from emperor.embedding.relative import DynamicPositionalBiasConfig
from emperor.layers import Layer
from emperor.linears import LinearLayerConfig
from emperor.memory._base import DynamicMemoryAbstract
from support.attention import build_attention_config


def source_memory_config(chunk_size=1):
    return AttentionSourceMemoryConfig(
        grouping_config=SumGroupingConfig(chunk_size=chunk_size)
    )


def test_source_memory_builds_and_runs_without_a_projector():
    memory = AttentionSourceMemory(attention_config()).double().eval()
    inputs = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
    projected, skip_mask, loss = memory(inputs)
    assert projected.shape == (6, 4)
    assert skip_mask.shape == (6, 1)
    assert skip_mask.all()
    assert loss is None
    projected.square().sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in memory.parameters()
    )


def test_memory_uses_its_own_linear_generator_without_input_blending():
    config = build_attention_config(
        SelfAttentionConfig,
        batch_size=2,
        num_heads=2,
        embedding_dim=4,
        return_attention_weights_flag=True,
        self_attention_projection_strategy=SelfAttentionProjectionStrategy.SEPARATE,
    )
    config = replace(
        config,
        batch_first_flag=True,
        source_memory_flag=True,
        source_memory_config=source_memory_config(),
    )
    model = config.build().double().eval()
    memory = model.projector.source_memory.model
    assert isinstance(memory.cfg, LinearLayerConfig)
    key_parameters = dict(model.projector.key_model.named_parameters())
    memory_parameters = dict(memory.named_parameters())
    assert {p.data_ptr() for p in memory_parameters.values()}.isdisjoint(
        p.data_ptr() for p in key_parameters.values()
    )
    assert not any(
        isinstance(module, DynamicMemoryAbstract) for module in memory.modules()
    )
    with torch.no_grad():
        for projection in (
            model.projector.query_model,
            model.projector.key_model,
            model.projector.value_model,
            model.projector.output_model,
        ):
            projection.layers[0].model.weight_params.copy_(torch.eye(4))
            projection.layers[0].model.bias_params.zero_()
        for parameter in memory.parameters():
            parameter.zero_()

    inputs = torch.randn(2, 3, 4, dtype=torch.float64)
    actual, weights, _ = model(inputs, inputs, inputs)
    queries = inputs.reshape(2, 3, 2, 2).transpose(1, 2)
    sources = torch.cat((inputs, torch.zeros_like(inputs)), dim=1)
    sources = sources.reshape(2, 6, 2, 2).transpose(1, 2)
    expected_weights = (queries @ sources.transpose(-1, -2) / 2**0.5).softmax(-1)
    expected = (expected_weights @ sources).transpose(1, 2).reshape_as(inputs)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(actual, expected)


def attention_config(**overrides):
    config = build_attention_config(
        SelfAttentionConfig,
        batch_size=3,
        num_heads=2,
        embedding_dim=4,
        target_sequence_length=5,
        source_sequence_length=5,
        return_attention_weights_flag=True,
        self_attention_projection_strategy=SelfAttentionProjectionStrategy.SEPARATE,
    )
    return replace(
        config,
        **dict(
            batch_first_flag=True,
            source_memory_flag=True,
            source_memory_config=source_memory_config(),
        )
        | overrides,
    )


@pytest.mark.parametrize(
    "mode",
    ["fused", "fused_kv", "separate", "independent", "shared_kv", "expert_kv", "soft"],
)
def test_projector_returns_memory_in_keys_and_values_before_head_reshaping(mode):
    config_class = (
        IndependentAttentionConfig
        if mode == "independent"
        else MixtureOfAttentionHeadsConfig
        if mode in {"shared_kv", "expert_kv", "soft"}
        else SelfAttentionConfig
    )
    config = build_attention_config(
        config_class,
        batch_size=2,
        num_heads=2,
        embedding_dim=4,
        target_sequence_length=3,
        source_sequence_length=3,
        use_kv_expert_models_flag=mode == "expert_kv",
        experts_top_k=2,
        experts_num_experts=3,
        self_attention_projection_strategy={
            "fused": SelfAttentionProjectionStrategy.FUSED,
            "fused_kv": SelfAttentionProjectionStrategy.FUSED_KEY_VALUE,
        }.get(mode, SelfAttentionProjectionStrategy.SEPARATE),
    )
    if mode == "soft":
        from tests.unit.test_expert_soft_moe import expert_config

        config.experts_config = expert_config(
            dimension=4, output_dim=4, experts=2, slots=2
        )
    config = replace(config, batch_first_flag=False)
    projector = (
        replace(
            config,
            source_memory_flag=True,
            source_memory_config=source_memory_config(2),
        )
        .build()
        .eval()
        .projector
    )
    plain = replace(config, source_memory_flag=False).build().eval().projector
    assert isinstance(projector.source_memory, AttentionSourceMemory)
    assert plain.source_memory is None
    plain.load_state_dict(
        {
            name: value
            for name, value in projector.state_dict().items()
            if not name.startswith("source_memory.")
        }
    )
    source = torch.randn(3, 2, 4)
    inputs = MultiHeadAttentionInputs(
        query=source,
        key=source,
        value=source,
        runtime_layout=AttentionRuntimeLayout(
            batch_size=2, target_sequence_length=3, source_sequence_length=3
        ),
    )
    generated = []
    handle = projector.source_memory.register_forward_hook(
        lambda _module, _args, output: generated.append(output[0])
    )
    expected = plain.compute_qkv_projections(inputs)
    actual = projector.compute_qkv_projections(inputs)
    handle.remove()
    assert len(generated) == 1
    memory = generated[0].reshape(2, 2, 4)
    if expected.key.dim() == 4:
        memory = memory.unsqueeze(2).expand(-1, -1, expected.key.size(2), -1)
    torch.testing.assert_close(actual.query, expected.query)
    torch.testing.assert_close(actual.key, torch.cat((expected.key, memory), dim=0))
    torch.testing.assert_close(actual.value, torch.cat((expected.value, memory), dim=0))
    assert actual.runtime_layout.source_sequence_length == 5
    assert actual.runtime_layout.real_source_sequence_length == 3
    assert actual.runtime_layout.target_sequence_length == expected.query.size(0)
    assert inputs.key.shape == source.shape


@pytest.mark.parametrize("mask_kind", ["none", "bool", "float", "heads"])
def test_memory_is_one_additional_source_for_both_keys_and_values(mask_kind):
    torch.manual_seed(57)
    model = attention_config().build().double().eval()
    with torch.no_grad():
        for projection in (
            model.projector.query_model,
            model.projector.key_model,
            model.projector.value_model,
            model.projector.output_model,
        ):
            linear = projection.layers[0].model
            linear.weight_params.copy_(torch.eye(4))
            linear.bias_params.zero_()

    memories = []
    memory = model.projector.source_memory
    handle = memory.register_forward_hook(
        lambda _module, _args, output: memories.append(output[0])
    )
    inputs = torch.randn(2, 3, 4, dtype=torch.float64)
    mask = None
    if mask_kind != "none":
        mask = torch.zeros(3, 3, dtype=torch.float64)
        mask[0, :] = -torch.inf
        mask[1, 2] = -torch.inf
        if mask_kind == "bool":
            mask = mask.isneginf()
        else:
            mask[2, 0] = -1.5
        if mask_kind == "heads":
            mask = mask.repeat(4, 1, 1)
            mask[0, 2, 1] = 0.75
    output, weights, loss = model(inputs, inputs, inputs, attention_mask=mask)
    handle.remove()
    assert len(memories) == 1
    generated = memories[0].reshape(3, 2, 4).transpose(0, 1)
    queries = inputs.reshape(2, 3, 2, 2).transpose(1, 2)
    sources = torch.cat((inputs, generated), dim=1).reshape(2, 6, 2, 2).transpose(1, 2)
    scores = queries @ sources.transpose(-1, -2) / 2**0.5
    if mask is not None:
        bias = (
            torch.zeros_like(mask, dtype=torch.float64).masked_fill(mask, -torch.inf)
            if mask.dtype == torch.bool
            else mask
        )
        if bias.ndim == 3:
            bias = bias.reshape(2, 2, 3, 3)
        memory_bias = torch.zeros_like(bias).masked_fill(bias.isneginf(), -torch.inf)
        scores = scores + torch.cat((bias, memory_bias), dim=-1)
    excluded = scores.isneginf().all(-1, keepdim=True)
    expected_weights = (
        scores.masked_fill(excluded, 0).softmax(-1).masked_fill(excluded, 0)
    )
    expected_output = (expected_weights @ sources).transpose(1, 2).reshape(2, 3, 4)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(output, expected_output)
    assert output.shape == inputs.shape
    assert loss is None


def test_memory_respects_causal_and_padding_masks():
    torch.manual_seed(41)
    config = attention_config(
        causal_attention_mask_flag=True,
    )
    model = config.build().eval()
    inputs = torch.randn(2, 4, 4)
    padding = torch.tensor([[False, False, False, True], [False, False, False, True]])
    output, weights, _ = model(inputs, inputs, inputs, k_padding_mask=padding)
    changed = inputs.clone()
    changed[:, 2:] += 10
    changed_output, _, _ = model(changed, changed, changed, k_padding_mask=padding)
    torch.testing.assert_close(output[:, :2], changed_output[:, :2])
    # Memory columns duplicate each source's visibility, including padded sources.
    assert torch.count_nonzero(weights[..., 3]) == 0
    assert torch.count_nonzero(weights[..., 7]) == 0
    assert torch.count_nonzero(weights[..., :2, 6:]) == 0
    assert torch.count_nonzero(weights[..., 1, 5]) > 0


@pytest.mark.parametrize("mode", ["independent", "shared_kv", "expert_kv", "soft"])
def test_memory_branch_runs_and_differentiates_in_each_attention_path(mode):
    torch.manual_seed(62)
    config = build_attention_config(
        IndependentAttentionConfig
        if mode == "independent"
        else MixtureOfAttentionHeadsConfig,
        batch_size=3,
        num_heads=2,
        embedding_dim=4,
        target_sequence_length=5,
        source_sequence_length=5,
        use_kv_expert_models_flag=mode == "expert_kv",
        experts_top_k=2,
        experts_num_experts=3,
    )
    config.batch_first_flag = True
    config.source_memory_flag = True
    config.source_memory_config = source_memory_config(2)
    if mode == "soft":
        from tests.unit.test_expert_soft_moe import expert_config

        config.experts_config = expert_config(
            dimension=4, output_dim=4, experts=2, slots=2
        )
    model = config.build().eval()
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    output, _, auxiliary_loss = model(inputs, inputs, inputs)
    assert output.shape == inputs.shape
    loss = output.square().sum()
    if auxiliary_loss is not None:
        loss = loss + auxiliary_loss
    loss.backward()
    memory = model.projector.source_memory.model
    assert isinstance(memory.cfg, LinearLayerConfig)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in memory.parameters()
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize(
    "bias,zero", [(False, False), (True, False), (False, True), (True, True)]
)
def test_explicit_memory_mask_recovers_memory_free_attention(bias, zero):
    torch.manual_seed(19)
    config = attention_config(
        causal_attention_mask_flag=True,
        add_key_value_bias_flag=bias,
        zero_attention_flag=zero,
    )
    model = config.build().eval()
    plain = replace(config, source_memory_flag=False).build().eval()
    plain.load_state_dict(
        {
            k: v
            for k, v in model.state_dict().items()
            if not k.startswith("projector.source_memory.model.")
        },
        strict=True,
    )
    inputs = torch.randn(2, 3, 4)
    mask = torch.zeros(3, 6, dtype=torch.bool)
    mask[:, 3:] = True
    expected, expected_weights, _ = plain(inputs, inputs, inputs)
    actual, weights, _ = model(inputs, inputs, inputs, attention_mask=mask)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.cat((weights[..., :3], weights[..., 6:]), dim=-1), expected_weights
    )
    assert torch.count_nonzero(weights[..., 3:6]) == 0
    unmasked, _, _ = model(inputs, inputs, inputs)
    assert not torch.allclose(actual, unmasked)


def test_source_memory_rejects_incompatible_projection_widths_at_construction():
    config = build_attention_config(
        IndependentAttentionConfig,
        num_heads=2,
        embedding_dim=4,
        query_key_projection_dim=2,
        value_projection_dim=4,
    )
    config.source_memory_flag = True
    with pytest.raises(ValueError, match="source memory.*width"):
        config.build()


@pytest.mark.parametrize("static", [False, True])
def test_source_memory_rejects_cross_and_static_inputs(static):
    model = attention_config().build()
    inputs = torch.randn(2, 3, 4)
    with pytest.raises(
        ValueError, match="source memory.*self-attention|source memory.*static"
    ):
        if static:
            model(inputs, inputs, inputs, static_k=torch.randn(4, 3, 2))
        else:
            context = inputs.clone()
            model(inputs, context, context)


@pytest.mark.parametrize("positions", ["rotary", "relative"])
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_chunk_memory_has_no_copied_token_position(positions, chunk_size):
    config = attention_config(
        causal_attention_mask_flag=True,
        source_memory_config=source_memory_config(chunk_size),
    )
    if positions == "rotary":
        config.rotary_base = 10000.0
    else:
        config.relative_positional_embedding_config = DynamicPositionalBiasConfig(
            embedding_dim=4, num_heads=2, max_positions=5
        )
    model = config.build().double().eval()
    with torch.no_grad():
        for projection in (
            model.projector.query_model,
            model.projector.key_model,
            model.projector.value_model,
            model.projector.output_model,
        ):
            projection.layers[0].model.weight_params.copy_(torch.eye(4))
            projection.layers[0].model.bias_params.zero_()
        model.projector.source_memory.model.weight_params.copy_(torch.eye(4))
        model.projector.source_memory.model.bias_params.zero_()
    inputs = torch.randn(2, 5, 4, dtype=torch.float64)
    contexts = torch.stack(
        [
            inputs[:, start : start + chunk_size].sum(1)
            for start in range(0, 5, chunk_size)
        ],
        dim=1,
    )
    token_heads = inputs.reshape(2, 5, 2, 2).transpose(1, 2)
    memory_heads = contexts.reshape(2, -1, 2, 2).transpose(1, 2)
    queries = token_heads
    token_keys = token_heads
    if positions == "rotary":
        angles = torch.arange(5, dtype=torch.float64)[None, None, :]
        queries = torch.stack(
            (
                token_heads[..., 0] * angles.cos() - token_heads[..., 1] * angles.sin(),
                token_heads[..., 0] * angles.sin() + token_heads[..., 1] * angles.cos(),
            ),
            dim=-1,
        )
        token_keys = queries
    keys = torch.cat((token_keys, memory_heads), dim=2)
    values = torch.cat((token_heads, memory_heads), dim=2)
    scores = queries @ keys.transpose(-1, -2) / 2**0.5
    if positions == "relative":
        token_bias = model.processor.relative_positional_embedding(
            queries / 2**0.5, sequence_length=5
        )
        memory_bias = token_bias.new_zeros(2, 2, 5, contexts.size(1))
        scores = scores + torch.cat((token_bias, memory_bias), dim=-1)
    chunk_ends = (torch.arange(contexts.size(1)) * chunk_size + chunk_size - 1).clamp(
        max=4
    )
    source_positions = torch.cat((torch.arange(5), chunk_ends))
    visible = torch.arange(5)[:, None] >= source_positions[None, :]
    expected_weights = scores.masked_fill(~visible, -torch.inf).softmax(-1)
    expected = (expected_weights @ values).transpose(1, 2).reshape(2, 5, 4)
    actual, weights, _ = model(inputs, inputs, inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(weights, expected_weights)


def test_memory_gradients_checkpoint_and_call_local_outputs():
    config = attention_config()
    model = config.build().train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    output, weights, _ = model(inputs, inputs, inputs)
    assert weights.shape[-1] == 6
    output.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.projector.source_memory.model.parameters()
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    restored = config.build().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.inference_mode():
        expected = model(inputs, inputs, inputs)[0]
        changed = inputs + 5
        model(changed, changed, changed)
        actual = model(inputs, inputs, inputs)[0]
        reloaded = restored(inputs, inputs, inputs)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(reloaded, expected)


@pytest.mark.parametrize("strategy", list(SelfAttentionProjectionStrategy))
@pytest.mark.parametrize("layout", ["batch", "sequence", "unbatched"])
def test_projection_strategies_and_input_layouts(strategy, layout):
    config = attention_config(projection_strategy=strategy)
    model = config.build().eval()
    reference = torch.randn(2, 3, 4)
    expected = model(reference, reference, reference)[0]
    if layout == "unbatched":
        inputs = reference[0]
        expected = expected[0]
    elif layout == "sequence":
        inputs = reference.transpose(0, 1)
        expected = expected.transpose(0, 1)
        sequence_model = replace(config, batch_first_flag=False).build().eval()
        sequence_model.load_state_dict(model.state_dict(), strict=True)
        model = sequence_model
    else:
        inputs = reference
    actual = model(inputs, inputs, inputs)[0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("grouped", [False, True])
def test_memory_uses_its_explicit_adaptive_generator_without_kv_grouping(grouped):
    from tests.unit.test_attention_adaptive_grouping import (
        grouped_weight_projection_model_config,
    )

    stack = grouped_weight_projection_model_config()
    if not grouped:
        stack.layer_config.layer_model_config.adaptive_augmentation_config.grouping_config = None
    generator_stack = deepcopy(stack)
    generator_stack.layer_config.layer_model_config.adaptive_augmentation_config.grouping_config = None
    memory_config = source_memory_config(2)
    memory_config.model_config = generator_stack
    config = attention_config(
        target_sequence_length=4,
        source_sequence_length=4,
        projection_model_config=stack,
        source_memory_config=memory_config,
        causal_attention_mask_flag=not grouped,
    )
    model = config.build().eval()
    key = model.projector.key_model
    memory = model.projector.source_memory.model
    assert type(memory.layers[0].model) is type(key.layers[0].model)
    assert memory is not key
    assert memory.layers[0].model.adaptive_behaviour.grouping_config is None
    inputs = torch.randn(2, 4, 4)
    contexts = inputs.reshape(2, 2, 2, 4).sum(2)
    expected_memory = Layer.run_model_from_hidden(
        memory, contexts.reshape(-1, 4)
    ).hidden
    expected_memory = expected_memory.reshape(2, 2, 4).transpose(0, 1).reshape(-1, 4)
    memory_output, _, _ = model.projector.source_memory(inputs.transpose(0, 1))
    torch.testing.assert_close(memory_output, expected_memory)
    actual = model(inputs, inputs, inputs)[0]
    if not grouped:
        changed_future = inputs.clone()
        changed_future[:, 2:] += 10
        changed_output = model(changed_future, changed_future, changed_future)[0]
        torch.testing.assert_close(actual[:, :2], changed_output[:, :2])
    actual.square().mean().backward()
    weight_model = memory.layers[0].model.adaptive_behaviour.weight_model
    for generator in (weight_model.input_model, weight_model.output_model):
        gradients = [parameter.grad for parameter in generator.parameters()]
        assert all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        )
        assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_expert_memory_head_order_matches_explicit_per_sample_reference():
    torch.manual_seed(13)
    config = build_attention_config(
        MixtureOfAttentionHeadsConfig,
        batch_size=3,
        num_heads=2,
        embedding_dim=4,
        target_sequence_length=5,
        source_sequence_length=5,
        use_kv_expert_models_flag=True,
        experts_top_k=2,
        experts_num_experts=3,
    )
    config.batch_first_flag = True
    config.source_memory_flag = True
    config.source_memory_config = source_memory_config(2)
    model = config.build().double().eval()
    observed = {}
    handles = []
    for name in ("query", "key", "value"):
        module = getattr(model.projector, f"{name}_model")
        handles.append(
            module.register_forward_hook(
                lambda _m, _a, out, name=name: observed.__setitem__(name, out[0])
            )
        )
    handles.append(
        model.projector.source_memory.register_forward_hook(
            lambda _m, _a, out: observed.__setitem__("memory", out[0])
        )
    )
    handles.append(
        model.projector.output_model.register_forward_pre_hook(
            lambda _m, args: observed.__setitem__("weighted", args[0])
        )
    )
    inputs = torch.randn(2, 3, 4, dtype=torch.float64)
    model(inputs, inputs, inputs)
    for handle in handles:
        handle.remove()
    projections = {
        name: observed[name].reshape(3, 2, 2, 4) for name in ("query", "key", "value")
    }
    memories = observed["memory"].reshape(2, 2, 4)
    expected = torch.empty(3, 2, 2, 4, dtype=torch.float64)
    for batch in range(2):
        for expert in range(2):
            for head in range(2):
                features = slice(2 * head, 2 * head + 2)
                query = projections["query"][:, batch, expert, features]
                key = projections["key"][:, batch, expert, features]
                value = projections["value"][:, batch, expert, features]
                memory = memories[:, batch, features]
                keys = torch.cat((key, memory))
                values = torch.cat((value, memory))
                weights = (query @ keys.T / 2**0.5).softmax(-1)
                expected[:, batch, expert, features] = weights @ values
    torch.testing.assert_close(observed["weighted"], expected.reshape(-1, 4))


def test_explicit_memory_mask_cannot_reopen_excluded_or_future_sources():
    model = attention_config(causal_attention_mask_flag=True).build().eval()
    inputs = torch.randn(2, 3, 4)
    mask = torch.zeros(3, 6)
    mask[:, 1] = -torch.inf
    mask[:, 3:] = 100  # Try to favor even masked/future memory positions.
    _, weights, _ = model(inputs, inputs, inputs, attention_mask=mask)
    assert torch.count_nonzero(weights[..., 1]) == 0
    assert torch.count_nonzero(weights[..., 4]) == 0
    assert torch.count_nonzero(weights[..., :2, 5]) == 0
    assert torch.count_nonzero(weights[..., 0, 3]) > 0


@pytest.mark.parametrize(
    "config_class", [SelfAttentionConfig, IndependentAttentionConfig]
)
def test_source_memory_cpu_autocast_preserves_finite_backward(config_class):
    config = build_attention_config(
        config_class,
        batch_size=2,
        num_heads=2,
        embedding_dim=4,
        target_sequence_length=3,
        source_sequence_length=3,
    )
    config.batch_first_flag = True
    config.source_memory_flag = True
    config.source_memory_config = source_memory_config(2)
    model = config.build()
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(inputs, inputs, inputs)[0]
        objective = output.square().mean()
    objective.backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs.grad).all()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.projector.source_memory.model.parameters()
    )
