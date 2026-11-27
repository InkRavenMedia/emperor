import pytest
import torch

from emperor.nn import SoftSlotRouting


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_pool_shared_bank_matches_independent_routes_and_gradients(causal, masked):
    logits = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    bank = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[[True], [True], [False]], [[True], [False], [True]]])
    routing = SoftSlotRouting(mask if masked else None, causal_prefix_flag=causal)
    actual = routing.pool_tokens_with_dispatch_weights(
        bank, routing.compute_dispatch_weights(logits)
    )
    expected = routing.compress_tokens(bank.expand(2, -1, -1), logits)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.square().sum(), (bank, logits)),
        torch.autograd.grad(expected.square().sum(), (bank, logits)),
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_saved_route_matches_independent_token_mixtures_and_gradients(causal, grouped):
    generator = torch.Generator().manual_seed(17)
    scores = torch.randn(
        2, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    tokens = torch.randn(
        2, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    active = torch.tensor([[False, True, True, False, True], [False] * 5])
    mask = active.unsqueeze(-1)
    logits = scores.masked_fill(~mask, torch.nan)
    if grouped:
        logits = logits.unflatten(-1, (2, 3))
    routing = SoftSlotRouting(mask, causal_prefix_flag=causal)
    slots = routing.compress_tokens(tokens.masked_fill(~mask, torch.nan), logits)
    slot_length = 5 if causal else 1
    assert slots.shape == (
        (2, 2, 3 * slot_length, 4) if grouped else (2, 6 * slot_length, 4)
    )
    actual = routing.decompress_slots(slots.tanh())

    expected = torch.zeros_like(tokens)
    for index in active[0].nonzero().flatten().tolist():
        selected = active[0].clone()
        if causal:
            selected[index + 1 :] = False
        pooled = scores[0, selected].softmax(0).T @ tokens[0, selected]
        expected[0, index] = scores[0, index].softmax(-1) @ pooled.tanh()
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.square().sum(), (scores, tokens)),
        torch.autograd.grad(expected.square().sum(), (scores, tokens)),
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_interleaved_routes_decompress_with_their_own_weights():
    tokens = torch.tensor([[[1.0], [3.0], [8.0]]])
    logits = torch.tensor([[[2.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]])
    first = SoftSlotRouting()
    first_slots = first.compress_tokens(tokens, logits)
    second = SoftSlotRouting()
    second_slots = second.compress_tokens(tokens, -logits)
    for route, slots, scores in (
        (second, second_slots, -logits),
        (first, first_slots, logits),
    ):
        expected_slots = scores.softmax(1).transpose(1, 2) @ tokens
        torch.testing.assert_close(
            route.decompress_slots(slots.square()),
            scores.softmax(-1) @ expected_slots.square(),
        )


def test_route_rejects_mismatched_batch_or_expert_axes():
    logits = torch.zeros(2, 5, 2, 3)
    routing = SoftSlotRouting()
    with pytest.raises(ValueError, match="batch and token"):
        routing.compress_tokens(torch.ones(1, 5, 4), logits)
    routing.compress_tokens(torch.ones(2, 5, 4), logits)
    with pytest.raises(ValueError, match="batch and slot"):
        routing.decompress_slots(torch.ones(2, 3, 2, 4))


@pytest.mark.parametrize(
    "invalid_mask", [torch.ones(2, 5, 1), torch.ones(2, 5, dtype=torch.bool)]
)
def test_compression_rejects_invalid_token_masks(invalid_mask):
    routing = SoftSlotRouting(invalid_mask)
    with pytest.raises(ValueError, match="Active token mask"):
        routing.compress_tokens(torch.ones(2, 5, 4), torch.zeros(2, 5, 3))


def test_only_a_successful_compression_replaces_the_saved_weights():
    tokens = torch.randn(2, 5, 4)
    logits = torch.randn(2, 5, 3)
    routing = SoftSlotRouting()
    with pytest.raises(RuntimeError, match="Compress tokens before"):
        routing.decompress_slots(torch.zeros(2, 3, 4))
    slots = routing.compress_tokens(tokens, logits)
    with pytest.raises(ValueError, match="batch and token"):
        routing.compress_tokens(tokens[:1], -logits)
    torch.testing.assert_close(
        routing.decompress_slots(slots), logits.softmax(-1) @ slots
    )
    new_slots = routing.compress_tokens(tokens, -logits)
    torch.testing.assert_close(
        routing.decompress_slots(new_slots), (-logits).softmax(-1) @ new_slots
    )


def test_precomputed_dispatch_and_combine_weights_must_match():
    routing = SoftSlotRouting()
    with pytest.raises(ValueError, match="Dispatch and combine weights"):
        routing.compress_with_routing_weights(
            torch.ones(2, 5, 4), torch.ones(2, 5, 3), torch.ones(2, 5, 1)
        )
