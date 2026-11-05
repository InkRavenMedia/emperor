import pickle
from dataclasses import replace

import pytest
import torch

from emperor.attention import AttentionSourceMemoryConfig
from emperor.attention._memory.model import AttentionSourceMemory
from emperor.augmentations.adaptive_parameters import SumGroupingConfig
from tests.unit.test_attention_source_memory import attention_config


@pytest.fixture(autouse=True)
def isolated_random_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(705)
        yield


def grouped_config(*, chunk=2, **overrides):
    return attention_config(
        source_memory_config=AttentionSourceMemoryConfig(
            grouping_config=SumGroupingConfig(chunk_size=chunk),
        ),
        **overrides,
    )


def test_memory_config_preserves_its_previous_serialized_owner():
    legacy_config_reference = (
        b"cemperor.attention._config\nAttentionSourceMemoryConfig\n."
    )
    assert pickle.loads(legacy_config_reference) is AttentionSourceMemoryConfig
    config = AttentionSourceMemoryConfig(
        grouping_config=SumGroupingConfig(chunk_size=2)
    )
    restored = pickle.loads(pickle.dumps(config))
    assert type(restored) is AttentionSourceMemoryConfig
    assert restored == config


def test_standalone_memory_requires_explicit_context_configuration():
    with pytest.raises(ValueError, match="source_memory_config.*chunk_size"):
        AttentionSourceMemory(attention_config(source_memory_config=None))


@pytest.mark.parametrize("variant", ["self", "independent", "mixture"])
def test_enabled_attention_memory_cannot_fall_back_to_token_projections(variant):
    from emperor.attention import (
        IndependentAttentionConfig,
        MixtureOfAttentionHeadsConfig,
        SelfAttentionConfig,
    )
    from tests.support.attention import build_attention_config

    config_type = {
        "self": SelfAttentionConfig,
        "independent": IndependentAttentionConfig,
        "mixture": MixtureOfAttentionHeadsConfig,
    }[variant]
    config = build_attention_config(config_type)
    config.source_memory_flag = True
    config.source_memory_config = None
    with pytest.raises(ValueError, match="source_memory_config.*chunk_size"):
        config.build()


@pytest.mark.parametrize("chunk_size", [1, 2, 10, 128])
def test_grouped_memory_generates_one_embedding_per_chunk_and_keeps_parameters(
    chunk_size,
):
    memory = AttentionSourceMemory(grouped_config(chunk=chunk_size)).double()
    with torch.no_grad():
        memory.model.weight_params.copy_(torch.eye(4))
        memory.model.bias_params.zero_()
    parameters = {name: id(value) for name, value in memory.named_parameters()}
    checkpoint = {name: value.clone() for name, value in memory.state_dict().items()}
    assert not list(memory.context.parameters())
    for length in (1, 3, 5, 7, 95, 100):
        source = (
            torch.arange(length * 2 * 4, dtype=torch.float64)
            .reshape(length, 2, 4)
            .requires_grad_()
        )
        output, mask, _ = memory(source)
        chunks = [
            source[start : start + chunk_size].sum(0)
            for start in range(0, length, chunk_size)
        ]
        expected = torch.stack(chunks)
        assert output.shape == (len(chunks) * 2, 4)
        torch.testing.assert_close(output.reshape(len(chunks), 2, 4), expected)
        assert mask.shape == (len(chunks) * 2, 1)
        assert mask.all()
        output.square().mean().backward()
        assert source.grad is not None
    assert parameters == {name: id(value) for name, value in memory.named_parameters()}
    torch.testing.assert_close(memory.state_dict(), checkpoint, rtol=0, atol=0)


def test_memory_generator_receives_chunk_summaries_without_slot_conditioning():
    memory = AttentionSourceMemory(grouped_config()).double()
    with torch.no_grad():
        memory.model.weight_params.copy_(2 * torch.eye(4))
        memory.model.bias_params.fill_(3)
    source = torch.arange(40, dtype=torch.float64).reshape(5, 2, 4)
    context = torch.stack([source[:2].sum(0), source[2:4].sum(0), source[4]])
    output, _, _ = memory(source)
    torch.testing.assert_close(output.reshape(3, 2, 4), 2 * context + 3)


@pytest.mark.parametrize("layout", ["sequence_first", "unbatched"])
def test_memory_preserves_layout_with_bias_and_zero_attention_sources(layout):
    config = grouped_config(add_key_value_bias_flag=True, zero_attention_flag=True)
    reference = config.build().double().eval()
    source = torch.randn(2, 5, 4, dtype=torch.float64)
    expected = reference(source, source, source)[0]
    model = replace(config, batch_first_flag=False).build().double().eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    inputs = source.transpose(0, 1) if layout == "sequence_first" else source[0]
    outputs, weights, _ = model(inputs, inputs, inputs)
    expected = expected.transpose(0, 1) if layout == "sequence_first" else expected[0]
    torch.testing.assert_close(outputs, expected)
    assert weights.shape[-1] == 5 + 3 + 2


def test_invalid_sequence_does_not_poison_a_later_valid_forward():
    memory = AttentionSourceMemory(grouped_config())
    parameters = dict(memory.named_parameters())
    checkpoint = {name: value.clone() for name, value in memory.state_dict().items()}
    with pytest.raises(ValueError, match="sequence_length must be a positive integer"):
        memory(torch.zeros(0, 2, 4))
    memory_rows, valid_memory_rows, _ = memory(torch.ones(2, 1, 4))
    assert memory_rows.shape == (1, 4)
    assert valid_memory_rows.all()
    assert parameters.keys() == dict(memory.named_parameters()).keys()
    assert all(
        parameters[name] is parameter for name, parameter in memory.named_parameters()
    )
    torch.testing.assert_close(memory.state_dict(), checkpoint, rtol=0, atol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_attention_memory_resolves_each_runtime_layout_independently(causal):
    config = grouped_config(causal_attention_mask_flag=causal)
    model = config.build().double().eval()
    for batch_size, sequence_length in ((2, 5), (1, 3), (2, 4)):
        source = torch.randn(batch_size, sequence_length, 4, dtype=torch.float64)
        padding = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
        padding[:, -1] = True
        memory_length = (sequence_length + 1) // 2
        mask = torch.zeros(
            sequence_length, sequence_length + memory_length, dtype=torch.bool
        )
        mask[0, 1] = True
        expected_model = config.build().double().eval()
        expected_model.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            actual = model(source, source, source, padding, mask)
            expected = expected_model(source, source, source, padding, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual[1].shape[-1] == sequence_length + memory_length


def test_generator_output_width_is_validated_before_restoring_memory_rows(monkeypatch):
    memory = AttentionSourceMemory(grouped_config())
    monkeypatch.setattr(memory.model, "forward", lambda inputs: inputs[:, :3])
    with pytest.raises(ValueError, match="must preserve row count and embedding width"):
        memory(torch.ones(5, 2, 4))


@pytest.mark.parametrize("causal", [False, True])
def test_grouped_attention_padding_causality_and_gradients(causal):
    torch.manual_seed(42)
    model = grouped_config(causal_attention_mask_flag=causal).build().double().eval()
    inputs = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    padding = torch.tensor([[False, False, False, True, True], [True] * 5])
    outputs, weights, _ = model(inputs, inputs, inputs, padding)
    assert outputs.shape == inputs.shape
    assert weights.shape == (2, 2, 5, 8)
    assert torch.isfinite(outputs).all()
    assert not weights[1].any()
    assert not weights[0, :, :, -1].any()
    changed = inputs.detach().clone()
    changed[padding] = 1000
    torch.testing.assert_close(
        model(changed, changed, changed, padding)[0][~padding], outputs[~padding]
    )
    if causal:
        changed = inputs.detach().clone()
        changed[:, 2:] += 100
        torch.testing.assert_close(
            model(changed, changed, changed, padding)[0][:, :2], outputs[:, :2]
        )
        assert not weights[:, :, 0, 5:].any()
    outputs.square().mean().backward()
    assert inputs.grad is not None
    assert model.projector.source_memory.model.weight_params.grad.abs().sum() > 0


@pytest.mark.parametrize("method", ["Sum", "Mean", "RMS", "MeanStd", "Attention"])
def test_each_reduction_matches_independent_reference_with_padding(method):
    from emperor import layers
    from emperor.augmentations import adaptive_parameters as grouping
    from emperor.linears import LinearLayerConfig

    reducer_type = getattr(grouping, method + "GroupingConfig")
    reducer = reducer_type(chunk_size=2)
    if method in {"MeanStd", "Attention"}:
        reducer.model_config = layers.LayerConfig(
            activation=layers.ActivationOptions.DISABLED,
            dropout_probability=0.0,
            layer_norm_position=layers.LayerNormPositionOptions.DISABLED,
            layer_model_config=LinearLayerConfig(bias_flag=False),
        )
    config = grouped_config()
    config.source_memory_config.grouping_config = reducer
    memory = AttentionSourceMemory(config).double()
    with torch.no_grad():
        memory.model.weight_params.copy_(torch.eye(4))
        memory.model.bias_params.zero_()
        if method == "Attention":
            memory.context.grouper.scorer.model.weight_params.zero_()
        if method == "MeanStd":
            memory.context.grouper.projection.model.weight_params.copy_(
                torch.cat((torch.eye(4), 0.5 * torch.eye(4)))
            )
    source = torch.tensor(
        [
            [[1.0, -3.0, 2.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
            [[3.0, 1.0, -2.0, 2.0], [3.0, 2.0, 1.0, 0.0]],
            [[2.0, -4.0, 6.0, 8.0], [5.0, 4.0, 3.0, 2.0]],
            [[900.0, 900.0, 900.0, 900.0], [7.0, 6.0, 5.0, 4.0]],
            [[900.0, 900.0, 900.0, 900.0], [9.0, 8.0, 7.0, 6.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    valid = torch.tensor(
        [[True, True], [True, True], [True, True], [False, True], [False, True]]
    )
    expected = torch.zeros(3, 2, 4, dtype=torch.float64)
    for batch in range(2):
        for chunk in range(3):
            tokens = source[chunk * 2 : chunk * 2 + 2, batch]
            tokens = tokens[valid[chunk * 2 : chunk * 2 + 2, batch]]
            if not len(tokens):
                continue
            context = (
                tokens.sum(0)
                if method == "Sum"
                else tokens.square().mean(0).sqrt()
                if method == "RMS"
                else tokens.mean(0) + 0.5 * tokens.std(0, correction=0)
                if method == "MeanStd"
                else tokens.mean(0)
            )
            expected[chunk, batch] = context
    output, validity, _ = memory(source, skip_mask=valid.reshape(-1, 1))
    torch.testing.assert_close(output.reshape(3, 2, 4), expected)
    assert not validity.reshape(3, 2)[2, 0]
    output.square().sum().backward()
    assert torch.isfinite(source.grad).all()
    if method in {"Attention", "MeanStd"}:
        parameters = list(memory.context.grouper.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)


@pytest.mark.parametrize("heads", [False, True])
def test_explicit_token_and_memory_masks_preserve_chunk_visibility(heads):
    model = grouped_config().build().double().eval()
    source = torch.randn(2, 5, 4, dtype=torch.float64)
    mask = torch.zeros((5, 8), dtype=torch.float64)
    mask[0, 1] = -torch.inf
    mask[0, 7] = -torch.inf
    mask[1, 6] = -1.5
    if heads:
        mask = mask.expand(4, -1, -1).clone()
        mask[2, 0, 2] = -torch.inf
    _, weights, _ = model(source, source, source, attention_mask=mask)
    assert not weights[:, :, 0, 5].any()
    assert not weights[:, :, 0, 7].any()
    if heads:
        assert not weights[1, 0, 0, 6].any()
        assert weights[0, 0, 0, 6] > 0
    changed = source.clone()
    changed[:, 1] += 100
    original = model(source, source, source, attention_mask=mask)[0]
    actual = model(changed, changed, changed, attention_mask=mask)[0]
    torch.testing.assert_close(actual[:, 0], original[:, 0])


@pytest.mark.parametrize(
    "mode",
    ["fused", "fused_kv", "separate", "independent", "shared_kv", "expert_kv", "soft"],
)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_paths_shapes_gradients_batch_independence_and_checkpoints(
    mode, causal
):
    from emperor.attention import (
        IndependentAttentionConfig,
        MixtureOfAttentionHeadsConfig,
        SelfAttentionConfig,
        SelfAttentionProjectionStrategy,
    )
    from tests.support.attention import build_attention_config
    from tests.unit.test_expert_soft_moe import expert_config

    cls = (
        IndependentAttentionConfig
        if mode == "independent"
        else MixtureOfAttentionHeadsConfig
        if mode in {"shared_kv", "expert_kv", "soft"}
        else SelfAttentionConfig
    )
    config = build_attention_config(
        cls,
        batch_size=2,
        num_heads=2,
        embedding_dim=4,
        target_sequence_length=5,
        source_sequence_length=5,
        experts_num_experts=3,
        experts_top_k=2,
        use_kv_expert_models_flag=mode == "expert_kv",
        self_attention_projection_strategy={
            "fused": SelfAttentionProjectionStrategy.FUSED,
            "fused_kv": SelfAttentionProjectionStrategy.FUSED_KEY_VALUE,
        }.get(mode, SelfAttentionProjectionStrategy.SEPARATE),
    )
    config = replace(
        config,
        batch_first_flag=True,
        source_memory_flag=True,
        causal_attention_mask_flag=causal,
        source_memory_config=grouped_config().source_memory_config,
    )
    if mode == "soft":
        config.experts_config = expert_config(
            dimension=4, output_dim=4, experts=2, slots=10
        )
    model = config.build().double().eval()
    source = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    output = model(source, source, source)[0]
    assert output.shape == source.shape
    assert torch.isfinite(output).all()
    changed = source.detach().clone()
    changed[1] += 100
    torch.testing.assert_close(model(changed, changed, changed)[0][0], output[0])
    if causal:
        changed = source.detach().clone()
        changed[:, 2:] += 100
        torch.testing.assert_close(
            model(changed, changed, changed)[0][:, :2], output[:, :2]
        )
    restored = config.build().double().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(source, source, source)[0], output)
    output.square().sum().backward()
    assert model.projector.source_memory.model.weight_params.grad.abs().sum() > 0
    assert torch.isfinite(source.grad).all()


@pytest.mark.parametrize("chunk", [0, True, -1, 2.5])
def test_invalid_chunk_sizes(chunk):
    with pytest.raises((TypeError, ValueError), match="positive"):
        AttentionSourceMemory(grouped_config(chunk=chunk))


def test_group_count_and_context_mixing_models_are_rejected():
    config = grouped_config()
    config.source_memory_config.grouping_config.group_count = 3
    with pytest.raises(ValueError, match="chunk_size only"):
        AttentionSourceMemory(config)
    config.source_memory_config.grouping_config.group_count = None
    config.source_memory_config.model_config = attention_config()
    with pytest.raises(ValueError, match="independent rows"):
        AttentionSourceMemory(config)
