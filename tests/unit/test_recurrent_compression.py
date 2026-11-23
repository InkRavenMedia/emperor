"""Token-space contracts across each recurrent block boundary."""

from dataclasses import replace

import pytest
import torch

from emperor.attention import AttentionLayerState
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    MeanGroupingConfig,
)
from emperor.experts import MixtureOfExpertsLayerState
from emperor.layers import LayerState, RecurrentCompressionConfig
from emperor.linears import LinearLayer, LinearLayerConfig
from support.adaptive_grouping_variants import GROUPING_CONFIGS, grouping_config
from support.layers import linear_stack_config
from support.recurrent import RecordingBlockConfig, inner_thinking_config
from tests.unit.test_linears import make_bias_config
from tests.unit.test_recurrent_composition_config import _minimal_variant_configs


def adaptive_scorer(width, slots, grouped=False):
    grouping = None
    if grouped:
        grouping = MeanGroupingConfig(
            scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
            chunk_size=2,
            sequence_length=8,
            input_order=AdaptiveParameterInputOrderOptions.BATCH_FIRST,
        )
    return AdaptiveLinearLayerConfig(
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            bias_config=make_bias_config(width, slots), grouping_config=grouping
        ),
    )


@pytest.mark.parametrize("adaptive", [False, True])
def test_two_softmax_axes_match_reference_and_gradients(adaptive):
    torch.manual_seed(31)
    config = RecurrentCompressionConfig(
        input_dim=3,
        max_slots=2,
        scorer_config=adaptive_scorer(3, 2)
        if adaptive
        else LinearLayerConfig(bias_flag=False),
    )
    compressor = config.build().double()
    block = RecordingBlockConfig().build().double()
    inputs = torch.randn(2, 5, 3, dtype=torch.double, requires_grad=True)
    result = compressor(block, LayerState(inputs, loss=inputs.new_tensor(1.0)))
    scores = torch.stack([compressor.scorer(row) for row in inputs])
    pooled = torch.einsum("bts,btd->bsd", scores.softmax(1), inputs)
    reference = torch.einsum("bts,bsd->btd", scores.softmax(2), pooled * block.scale)
    torch.testing.assert_close(block.calls[0].hidden, pooled)
    torch.testing.assert_close(result.hidden, reference)
    assert result.loss.item() == 1.25
    parameters = (inputs, *compressor.parameters(), block.scale)
    weights = torch.randn_like(inputs)
    actual = torch.autograd.grad(
        (result.hidden * weights).sum(), parameters, allow_unused=True
    )
    expected = torch.autograd.grad(
        (reference * weights).sum(), parameters, allow_unused=True
    )
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)
    assert any(grad is not None and grad.abs().sum() > 0 for grad in actual[1:-1])
    if adaptive:
        assert any(
            grad is not None and grad.abs().sum() > 0
            for (name, _), grad in zip(
                compressor.named_parameters(), actual[1:-1], strict=True
            )
            if "adaptive_behaviour" in name
        )
    restored = config.build().double()
    restored.load_state_dict(compressor.state_dict(), strict=True)
    torch.testing.assert_close(
        restored(block, LayerState(inputs)).hidden, result.hidden
    )


@pytest.mark.parametrize("grouped", [False, True])
def test_padding_short_and_empty_examples_match_separate_valid_inputs(grouped):
    torch.manual_seed(9)
    config = RecurrentCompressionConfig(
        input_dim=3,
        max_slots=3,
        scorer_config=adaptive_scorer(3, 3, grouped=True) if grouped else None,
    )
    compressor = config.build()
    block = RecordingBlockConfig().build()
    padding = torch.tensor(
        [[0, 0, 1, 0, 0], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.bool
    )
    features = (
        torch.randn(3, 5, 3)
        .masked_fill(padding[..., None], float("nan"))
        .requires_grad_()
    )
    result = compressor(block, AttentionLayerState(features, key_padding_mask=padding))
    assert sorted(call.hidden.size(-2) for call in block.calls) == [1, 3]
    assert result.key_padding_mask is padding
    for row in (0, 1):
        valid = ~padding[row]
        reference = compressor(block, LayerState(features[row, valid])).hidden
        torch.testing.assert_close(result.hidden[row, valid], reference)
    assert (result.hidden[padding] == 0).all()
    result.hidden.square().sum().backward()
    assert torch.isfinite(features.grad).all() and (features.grad[padding] == 0).all()


@pytest.mark.parametrize("grouped", [False, True])
def test_causal_compression_matches_independently_scored_prefixes(grouped):
    torch.manual_seed(7)
    config = RecurrentCompressionConfig(
        input_dim=2,
        max_slots=2,
        scorer_config=adaptive_scorer(2, 2, grouped=True) if grouped else None,
    )
    compressor = config.build()
    block = RecordingBlockConfig().build()
    features = torch.randn(2, 5, 2, requires_grad=True)
    causal = torch.ones(5, 5, dtype=torch.bool).triu(1)
    result = compressor(block, AttentionLayerState(features, attention_mask=causal))
    expected = torch.cat(
        [
            compressor(block, LayerState(features[:, :end])).hidden[:, -1:]
            for end in range(1, 6)
        ],
        dim=1,
    )
    torch.testing.assert_close(result.hidden, expected)
    result.hidden[:, :3].sum().backward()
    assert (features.grad[:, 3:] == 0).all()


@pytest.mark.parametrize("mask_kind", ["bool", "additive", "skip"])
def test_causal_compression_runs_only_prefixes_with_a_valid_target(mask_kind):
    padding = torch.tensor(
        [[0, 0, 1, 1, 1], [0, 1, 0, 0, 1], [1, 1, 1, 1, 1]], dtype=torch.bool
    )
    tokens = torch.randn(3, 5, 2).masked_fill(padding[..., None], float("nan"))
    tokens.requires_grad_()
    if mask_kind == "skip":
        state = MixtureOfExpertsLayerState(
            tokens, skip_mask=~padding[..., None], causal_prefix_flag=True
        )
    else:
        mask = (
            padding
            if mask_kind == "bool"
            else torch.zeros_like(tokens[..., 0]).masked_fill(padding, -torch.inf)
        )
        state = AttentionLayerState(
            tokens,
            key_padding_mask=mask,
            attention_mask=torch.ones(5, 5, dtype=torch.bool).triu(1),
        )
    state.loss = torch.tensor(2.0)
    block = RecordingBlockConfig().build()
    compressor = RecurrentCompressionConfig(input_dim=2, max_slots=4).build()
    result = compressor(block, state)
    assert [call.hidden.shape[:2] for call in block.calls] == [
        (2, 1),
        (1, 2),
        (1, 2),
        (1, 3),
    ]
    torch.testing.assert_close(result.loss, state.loss + 4 * 0.25)
    assert (result.hidden[padding] == 0).all()
    result.hidden.sum().backward()
    assert torch.isfinite(tokens.grad).all() and (tokens.grad[padding] == 0).all()


@pytest.mark.parametrize(
    "config", _minimal_variant_configs(), ids=lambda cfg: type(cfg).__name__
)
def test_each_variant_routes_each_transition_and_keeps_full_state(config):
    torch.manual_seed(21)
    slots = 3
    if hasattr(config, "high_block_config"):
        config = replace(
            config,
            low_block_config=RecordingBlockConfig(),
            high_block_config=RecordingBlockConfig(),
            low_cycles=2,
        )
        expected_lengths = [3, 3, 3]
    elif hasattr(config, "latent_updates_per_answer_update"):
        config = replace(
            config,
            block_config=RecordingBlockConfig(),
            latent_updates_per_answer_update=2,
        )
        expected_lengths = [3, 3, 3]
    elif hasattr(config, "sampler_config"):
        config = inner_thinking_config(
            max_steps=2, initial_iterations=2, selection_ratio=0.25
        )
        expected_lengths = [3, 2]
    else:
        config = replace(
            config,
            max_steps=2,
            initial_iterations=2,
            block_config=RecordingBlockConfig(),
        )
        expected_lengths = [3, 3]
    model = replace(
        config, compression_config=RecurrentCompressionConfig(max_slots=slots)
    ).build()
    inputs = torch.randn(2, 8, 2)
    scored = []
    model.step.compressor.scorer.register_forward_pre_hook(
        lambda _, args: scored.append(args[0].detach().clone())
    )
    lengths = []
    for module in model.modules():
        if isinstance(module, RecordingBlockConfig().registry_owner()):
            module.register_forward_pre_hook(
                lambda _, args: lengths.append(args[0].hidden.size(-2))
            )
    result = model(LayerState(inputs))
    assert lengths == expected_lengths
    assert len(scored) == len(expected_lengths)
    assert not torch.equal(scored[0], scored[1])
    assert result.hidden.shape == inputs.shape
    result.hidden.square().sum().backward()
    assert model.step.compressor.scorer.weight_params.grad is not None
    assert not any("compressor" in key for key in config.build().state_dict())


def test_invalid_compression_config_and_unsupported_mask_are_actionable():
    for slots in (0, True):
        with pytest.raises(ValueError, match="max_slots"):
            RecurrentCompressionConfig(input_dim=2, max_slots=slots).build()
    with pytest.raises(ValueError, match="output_dim"):
        RecurrentCompressionConfig(
            input_dim=2, max_slots=3, scorer_config=LinearLayerConfig(output_dim=2)
        ).build()
    compressor = RecurrentCompressionConfig(input_dim=2, max_slots=2).build()
    state = AttentionLayerState(
        torch.zeros(3, 2), attention_mask=torch.eye(3, dtype=torch.bool)
    )
    with pytest.raises(ValueError, match="causal self-attention masks"):
        compressor(RecordingBlockConfig().build(), state)


def test_inner_thinking_causal_selection_is_prefix_local_with_padding():
    torch.manual_seed(32)
    config = inner_thinking_config(
        selection_ratio=0.75, compression_config=RecurrentCompressionConfig(max_slots=2)
    )
    model = config.build().eval()
    features = torch.randn(2, 6, 2)
    padding = torch.tensor([[False] * 6, [True, True, False, True, True, True]])

    def run(hidden, mask):
        count = hidden.size(1)
        return model(
            AttentionLayerState(
                hidden,
                key_padding_mask=mask,
                attention_mask=torch.ones(count, count, dtype=torch.bool).triu(1),
            )
        )

    full = run(features, padding)
    for end in range(1, 7):
        torch.testing.assert_close(
            run(features[:, :end], padding[:, :end]).hidden[:, -1],
            full.hidden[:, end - 1],
        )
    padded = run(
        torch.nn.functional.pad(features, (0, 0, 0, 2)),
        torch.nn.functional.pad(padding, (0, 2), value=True),
    )
    torch.testing.assert_close(padded.hidden[:, :6], full.hidden)
    torch.testing.assert_close(padded.loss, full.loss)


def test_extra_batch_axes_are_independent_and_original_layout_is_restored():
    compressor = RecurrentCompressionConfig(input_dim=2, max_slots=2).build()
    block = RecordingBlockConfig().build()
    features = torch.randn(2, 3, 4, 2)
    padding = torch.zeros(2, 3, 4, dtype=torch.bool)
    padding[1, 1, 2:] = True
    result = compressor(block, AttentionLayerState(features, key_padding_mask=padding))
    expected = torch.stack(
        [
            compressor(block, AttentionLayerState(row, key_padding_mask=mask)).hidden
            for row, mask in zip(features, padding, strict=True)
        ]
    )
    torch.testing.assert_close(result.hidden, expected)
    assert result.key_padding_mask is padding


@pytest.mark.parametrize(
    "scope",
    [
        AdaptiveParameterGroupingScopeOptions.ROWS,
        AdaptiveParameterGroupingScopeOptions.SEQUENCE,
    ],
)
def test_group_count_scorer_handles_uneven_groups_and_single_token(scope):
    torch.manual_seed(12)
    scorer_config = adaptive_scorer(2, 2)
    grouping = MeanGroupingConfig(group_count=3, scope=scope)
    if scope is AdaptiveParameterGroupingScopeOptions.SEQUENCE:
        grouping = replace(
            grouping,
            sequence_length=6,
            input_order=AdaptiveParameterInputOrderOptions.BATCH_FIRST,
        )
    scorer_config.adaptive_augmentation_config.grouping_config = grouping
    compressor = RecurrentCompressionConfig(
        input_dim=2, max_slots=2, scorer_config=scorer_config
    ).build()
    block = RecordingBlockConfig().build()
    features = torch.randn(1, 5, 2)
    equal_groups = replace(grouping, group_count=1)
    scorer_config.adaptive_augmentation_config.grouping_config = equal_groups
    reference = RecurrentCompressionConfig(
        input_dim=2, max_slots=2, scorer_config=scorer_config
    ).build()
    reference.load_state_dict(compressor.state_dict())
    # Five positions divide into contiguous groups of two, two, and one.
    expected_scores = torch.cat(
        [reference._score(group).hidden for group in features.tensor_split(3, dim=1)],
        dim=1,
    )
    torch.testing.assert_close(compressor._score(features).hidden, expected_scores)
    single = compressor(block, LayerState(features[:, :1])).hidden
    torch.testing.assert_close(single, features[:, :1])


@pytest.mark.parametrize(
    "base", _minimal_variant_configs(), ids=lambda cfg: type(cfg).__name__
)
@pytest.mark.parametrize("grouping_type", GROUPING_CONFIGS)
def test_each_variant_inherits_the_adaptive_layer_settings(base, grouping_type):
    block = linear_stack_config(2)
    template = replace(adaptive_scorer(2, 2), input_dim=2, output_dim=2)
    template.adaptive_augmentation_config.grouping_config = grouping_config(
        grouping_type,
        scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
        sequence_length=6,
        input_order=AdaptiveParameterInputOrderOptions.BATCH_FIRST,
    )
    block.layer_config.layer_model_config = template
    transitions = (
        {"high_block_config": block, "low_block_config": linear_stack_config(2)}
        if hasattr(base, "high_block_config")
        else {"block_config": block}
    )
    compression = RecurrentCompressionConfig(max_slots=3)
    config = replace(base, compression_config=compression, **transitions)
    model = config.build()
    compressor = model.step.compressor
    inherited = compressor.cfg.scorer_config
    assert inherited == replace(template, output_dim=3)
    assert (
        inherited.adaptive_augmentation_config
        is not template.adaptive_augmentation_config
    )
    assert compression.scorer_config is None and template.output_dim == 2
    assert isinstance(compressor.scorer, template.registry_owner())
    result = model(LayerState(torch.randn(2, 6, 2)))
    assert result.hidden.shape == (2, 6, 2)
    result.hidden.square().sum().backward()
    assert any(
        p.grad is not None for p in compressor.scorer.adaptive_behaviour.parameters()
    )


def test_plain_block_bias_is_inherited_and_opaque_block_uses_default():
    base = _minimal_variant_configs()[0]
    config = replace(base, compression_config=RecurrentCompressionConfig(max_slots=2))
    assert config.build().step.compressor.scorer.bias_params is not None
    model = replace(
        config, block_config=linear_stack_config(2, bias_flag=False)
    ).build()
    assert model.step.compressor.scorer.bias_params is None


def test_linear_router_can_run_an_adaptive_block_without_mixing_examples():
    block = linear_stack_config(3)
    block.layer_config.layer_model_config = adaptive_scorer(3, 3, grouped=True)
    base = _minimal_variant_configs()[0]
    model = (
        replace(
            base,
            input_dim=3,
            output_dim=3,
            block_config=block,
            compression_config=RecurrentCompressionConfig(
                max_slots=2, scorer_config=LinearLayerConfig(bias_flag=True)
            ),
        )
        .build()
        .eval()
    )
    assert isinstance(model.step.compressor.scorer, LinearLayer)
    features = torch.randn(2, 5, 3)
    batch = model.forward_sequence(features).hidden
    individual = torch.stack([model(LayerState(row)).hidden for row in features])
    torch.testing.assert_close(batch, individual)
    batch.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in model.named_parameters()
        if "adaptive_behaviour" in name
    )


def test_sequence_first_uses_batch_padding_and_preserves_layout():
    from types import SimpleNamespace

    class SequenceFirst(torch.nn.Module):
        cfg = SimpleNamespace(batch_first_flag=False)

        def forward(self, state):
            # Make swapping sequence and batch axes observable.
            return replace(
                state, hidden=state.hidden + state.hidden.mean(dim=0, keepdim=True)
            )

    config = RecurrentCompressionConfig(input_dim=2, max_slots=3)
    compressor = config.build()
    block = SequenceFirst()
    hidden = torch.randn(5, 2, 2)
    pad = torch.tensor(
        [[False, False, False, False, False], [True, False, False, True, True]]
    )
    result = compressor(block, AttentionLayerState(hidden, key_padding_mask=pad))
    for idx in range(2):
        ref = compressor(block, LayerState(hidden[~pad[idx], idx])).hidden
        torch.testing.assert_close(result.hidden[~pad[idx], idx], ref)
    assert result.hidden.shape == hidden.shape and result.key_padding_mask is pad


def test_legacy_attention_layout_is_inferred_before_shortening_the_sequence():
    from types import SimpleNamespace

    from emperor.attention._ops.batching import BatchDimensionManager
    from emperor.attention._runtime import MultiHeadAttentionInputs

    class LegacyLayoutBlock(torch.nn.Module):
        cfg = SimpleNamespace(batch_first_flag=None, batch_size=2)

        def forward(self, state):
            tensor = state.hidden
            manager = BatchDimensionManager(self.cfg)
            internal = manager.convert_inputs_to_internal_layout(
                MultiHeadAttentionInputs(query=tensor, key=tensor, value=tensor)
            )
            value = internal.query + internal.query.mean(0)
            return replace(state, hidden=manager.restore_output_layout(value, internal))

    block = LegacyLayoutBlock()
    compressor = RecurrentCompressionConfig(input_dim=2, max_slots=2).build()
    hidden = torch.randn(2, 5, 2)
    expected = torch.stack(
        [compressor(block, LayerState(row)).hidden for row in hidden]
    )
    # Both axes of the inner block are now 2, so re-inferring here would swap them.
    torch.testing.assert_close(compressor(block, LayerState(hidden)).hidden, expected)
    torch.testing.assert_close(
        compressor(block, LayerState(hidden.transpose(0, 1))).hidden,
        expected.transpose(0, 1),
    )
