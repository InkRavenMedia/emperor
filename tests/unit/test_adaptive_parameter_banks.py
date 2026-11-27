"""Independent selection/reduction oracles for complete parameter banks."""

import pytest
import torch

from support.adaptive_generation import (
    bias_mixture_config,
    matrix_sampler_config,
    weight_mixture_config,
)


@pytest.mark.parametrize("with_indices", [True, False])
def test_singleton_bias_bank_preserves_independent_contexts(with_indices, monkeypatch):
    from emperor.augmentations.adaptive_parameters import MatrixBiasMixtureConfig
    from support.adaptive_grouping import linear_stack_config

    model = (
        MatrixBiasMixtureConfig(
            input_dim=2,
            output_dim=3,
            num_experts=1,
            top_k=1,
            sampler_config=matrix_sampler_config(num_experts=1, top_k=1),
            model_config=linear_stack_config(2, 1),
        )
        .build()
        .double()
    )
    probabilities = torch.tensor(
        [[0.2], [0.8]], dtype=torch.float64, requires_grad=True
    )
    indices = torch.zeros(2, 1, dtype=torch.long) if with_indices else None
    expected = torch.stack([p * model.parameter_bank[0] for p in probabilities[:, 0]])
    monkeypatch.setattr(
        model.sampler,
        "sample_probabilities_and_indices",
        lambda context, skip_mask=None: (
            probabilities,
            indices,
            None,
            context.new_zeros(()),
        ),
    )
    output = model(
        torch.zeros_like(model.parameter_bank[0]), torch.ones(2, 2, dtype=torch.float64)
    )
    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    assert probabilities.grad.abs().sum() > 0
    assert model.parameter_bank.grad.abs().sum() > 0


def test_independent_banks_add_dense_weight_and_sparse_bias_updates_to_base():
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveLinearLayerConfig,
        AdaptiveParameterAugmentationConfig,
    )

    model = (
        AdaptiveLinearLayerConfig(
            input_dim=2,
            output_dim=3,
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                weight_config=weight_mixture_config(),
                bias_config=bias_mixture_config(top_k=1),
            ),
        )
        .build()
        .double()
    )
    weight = model.adaptive_behaviour.weight_model
    bias = model.adaptive_behaviour.bias_model
    with torch.no_grad():
        for scorer, distribution in (
            (weight.model, [0.25, 0.05, 0.7]),
            (bias.sampler.router.model, [0.6, 0.3, 0.1]),
        ):
            router = scorer[0].model
            router.weight_params.zero_()
            router.bias_params.copy_(torch.tensor(distribution).log())
    x = torch.tensor([[2.0, 1.0], [-1.0, 3.0]], dtype=torch.float64)
    expected_weight = (
        0.25 * weight.parameter_bank[0]
        + 0.05 * weight.parameter_bank[1]
        + 0.7 * weight.parameter_bank[2]
    )
    expected_bias = 0.6 * bias.parameter_bank[0]
    torch.testing.assert_close(
        model(x),
        x @ (model.weight_params + expected_weight) + model.bias_params + expected_bias,
    )
    assert model.weight_params is not None and model.bias_params is not None
    assert "weight_params" in model.state_dict() and "bias_params" in model.state_dict()


@pytest.mark.parametrize("experts,k", [(1, 1), (3, 1), (3, 2), (3, 3)])
@pytest.mark.parametrize("normalize", [False, True])
def test_routes_follow_existing_sampler_probabilities_and_gradients(
    experts, k, normalize
):
    cfg = bias_mixture_config(
        num_experts=experts,
        top_k=k,
        sampler_config=matrix_sampler_config(
            num_experts=experts, top_k=k, normalize_probabilities_flag=normalize
        ),
    )
    if k == 1 and normalize:
        with pytest.raises(
            ValueError, match="normalize_probabilities_flag must be False"
        ):
            cfg.build()
        return
    model = cfg.build().double()
    x = torch.tensor([[0.1, 1.7], [-2.0, 0.6]], dtype=torch.float64, requires_grad=True)
    logits = model.sampler.router.compute_logit_scores(x)
    full = logits.softmax(-1)
    if k == experts:
        indices = torch.arange(experts).expand(2, -1)
        alpha = full
    else:
        alpha, indices = full.topk(k, dim=-1)
    if normalize:
        alpha = alpha / (alpha.sum(-1, keepdim=True) + 1e-6).detach()
    expected_w = torch.stack(
        [
            sum(alpha[c, j] * model.parameter_bank[indices[c, j]] for j in range(k))
            for c in range(2)
        ]
    )
    actual = model(torch.zeros_like(model.parameter_bank[0]), x)
    torch.testing.assert_close(actual, expected_w)
    inputs = (x, model.parameter_bank, *model.sampler.router.parameters())
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
    expected_grads = torch.autograd.grad(expected_w.square().sum(), inputs)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)
    if experts > 1:
        assert any(gradient.abs().sum() > 0 for gradient in actual_grads[2:])
    if k < experts:
        selected = set(indices.flatten().tolist())
        for index in set(range(experts)) - selected:
            assert actual_grads[1][index].count_nonzero() == 0


@pytest.mark.parametrize("expert_count", [1, 4])
def test_dense_weight_router_trains_from_task_loss_for_every_token(expert_count):
    mixture = (
        weight_mixture_config(input_dim=4, output_dim=5, num_experts=expert_count)
        .build()
        .double()
    )
    sequences = torch.eye(4, dtype=torch.float64)[[[0, 1, 2], [3, 1, 0]], :]
    tokens = sequences.flatten(0, 1).requires_grad_()
    with torch.no_grad():
        scorer = mixture.model[0].model
        scorer.weight_params.copy_(torch.eye(4, expert_count, dtype=torch.float64))
        scorer.bias_params.zero_()
        mixture.parameter_bank.copy_(
            torch.arange(expert_count * 20).reshape(expert_count, 4, 5) / 20
        )
    base = torch.zeros(4, 5, dtype=torch.float64)
    scores = tokens @ scorer.weight_params + scorer.bias_params
    shares = scores.softmax(-1)
    expected = torch.einsum(
        "bte,eio->btio", shares.reshape(2, 3, expert_count), mixture.parameter_bank
    )
    actual = mixture(base, tokens).reshape(2, 3, 4, 5)
    torch.testing.assert_close(actual, expected)
    if expert_count > 1:
        assert not torch.allclose(actual[0, 0], actual[0, 1])
    assert mixture.get_auxiliary_loss() is None
    task_loss = torch.einsum("bti,btio->bto", sequences, actual).square().mean()
    task_loss.backward()
    assert (mixture.parameter_bank.grad.flatten(1).abs().sum(1) > 0).all()
    if expert_count > 1:
        assert scorer.weight_params.grad.abs().sum() > 0


def test_dense_weight_mixture_rejects_top_k_router_checkpoint():
    mixture = weight_mixture_config().build()
    top_k_checkpoint = {
        ("sampler.router." + key if key.startswith("model.") else key): value
        for key, value in mixture.state_dict().items()
    }
    with pytest.raises(
        RuntimeError, match=r"Unexpected key\(s\).*sampler\.router\.model\."
    ):
        mixture.load_state_dict(top_k_checkpoint)


@pytest.mark.parametrize(
    "bank_dtype,router_dtype",
    [(torch.float32, torch.float64), (torch.float64, torch.float32)],
)
def test_full_weight_bank_preserves_mixed_precision_per_token(bank_dtype, router_dtype):
    model = weight_mixture_config(input_dim=4, output_dim=5, num_experts=4)
    mixture = model.build().to(bank_dtype)
    mixture.model.to(router_dtype)
    tokens = torch.linspace(-1, 1, 64, dtype=router_dtype).reshape(16, 4)
    probabilities = mixture.model.forward_hidden(tokens).softmax(-1)
    base = torch.zeros_like(mixture.parameter_bank[0])
    actual = mixture(base, tokens)
    expected = (probabilities[:, :, None, None] * mixture.parameter_bank).sum(dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=1e-7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": -0.1},
        {"num_topk_samples": 3},
        {"noisy_topk_flag": "yes"},
        {"switch_loss_weight": -0.2},
        {"coefficient_of_variation_loss_weight": -0.1},
        {"zero_centred_loss_weight": -0.1},
        {"mutual_information_loss_weight": -0.1},
        {"num_experts": 5},
        {"top_k": 3},
    ],
)
def test_invalid_or_disagreeing_sampler_options_fail_before_initialization(kwargs):
    rng = torch.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError)):
        bias_mixture_config(sampler_config=matrix_sampler_config(**kwargs)).build()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_experts": True},
        {"num_experts": 0},
        {"top_k": 0},
        {"top_k": 4},
    ],
)
def test_invalid_weight_mixture_configuration_fails_early(kwargs):
    with pytest.raises((TypeError, ValueError)):
        weight_mixture_config(**kwargs).build()


@pytest.mark.parametrize(
    "bad_probs,bad_indices",
    [
        (torch.ones(2, 2), torch.tensor([[0, 3], [0, 1]])),
        (torch.ones(2, 2), torch.tensor([[0.0, 1.0], [0.0, 1.0]])),
        (torch.ones(2, 2), torch.tensor([[0], [1]])),
        (torch.ones(3, 2), torch.tensor([[0, 1], [1, 2]])),
        (torch.tensor([[float("nan"), 1.0]]), torch.tensor([[0, 1]])),
        (None, torch.tensor([[0, 1]])),
        (torch.ones(2, 2), None),
    ],
)
def test_malformed_sampler_routes_cannot_broadcast_silently(
    bad_probs, bad_indices, monkeypatch
):
    model = bias_mixture_config().build()
    monkeypatch.setattr(
        model.sampler,
        "sample_probabilities_and_indices",
        lambda context, skip_mask=None: (
            bad_probs,
            bad_indices,
            None,
            context.new_zeros(()),
        ),
    )
    with pytest.raises((TypeError, ValueError)):
        model(torch.zeros_like(model.parameter_bank[0]), torch.ones(2, 2))


@pytest.mark.parametrize("bias", [False, True])
def test_linear_owns_bias_policy_and_constructs_base_parameters(bias, monkeypatch):
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveLinearLayerConfig,
        AdaptiveParameterAugmentationConfig,
    )
    from emperor.linears import LinearAbstract

    calls = []
    original = LinearAbstract._create_weight_parameters

    def observe(model):
        calls.append(model)
        return original(model)

    monkeypatch.setattr(LinearAbstract, "_create_weight_parameters", observe)
    layer = AdaptiveLinearLayerConfig(
        input_dim=2,
        output_dim=3,
        bias_flag=bias,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            weight_config=weight_mixture_config(),
            bias_config=bias_mixture_config() if bias else None,
        ),
    ).build()
    assert sum(owner is layer for owner in calls) == 1
    assert layer.weight_params is not None
    assert (layer.bias_params is not None) is bias
    assert (layer.adaptive_behaviour.bias_model is not None) is bias
    assert layer(torch.randn(2, 2)).shape == (2, 3)


@pytest.mark.parametrize("overrides", [{"input_dim": 4}, {"output_dim": 4}])
@pytest.mark.parametrize(
    "slot, config_factory",
    [("weight_config", weight_mixture_config), ("bias_config", bias_mixture_config)],
)
def test_mixture_cannot_disagree_with_owning_linear(overrides, slot, config_factory):
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveLinearLayerConfig,
        AdaptiveParameterAugmentationConfig,
    )

    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="match"):
        AdaptiveLinearLayerConfig(
            input_dim=2,
            output_dim=3,
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                **{slot: config_factory(**overrides)}
            ),
        ).build()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_parametric_application_rejects_a_second_parameter_set_owner():
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterAugmentationConfig,
    )
    from emperor.parametric import (
        AdaptiveRouterOptions,
        MatrixWeightsMixtureConfig,
        ParametricLayerConfig,
    )

    sampler = matrix_sampler_config()
    cfg = ParametricLayerConfig(
        input_dim=2,
        output_dim=3,
        weight_mixture_config=MatrixWeightsMixtureConfig(
            input_dim=2,
            output_dim=3,
            num_experts=3,
            top_k=2,
            weighted_parameters_flag=True,
        ),
        routing_initialization_mode=AdaptiveRouterOptions.SHARED_ROUTER,
        router_config=sampler.router_config,
        sampler_config=sampler,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            weight_config=weight_mixture_config()
        ),
    )
    with pytest.raises(ValueError, match="weight_config"):
        cfg.build()


def test_adaptive_and_parametric_matrix_implementations_are_separate():
    from dataclasses import fields

    from emperor import parametric
    from emperor.augmentations import adaptive_parameters
    from emperor.augmentations.adaptive_parameters import MatrixWeightsMixtureConfig

    expected = {"input_dim", "output_dim", "num_experts", "model_config"}
    assert expected <= {field.name for field in fields(MatrixWeightsMixtureConfig)}
    assert "clip_parameter_option" not in {
        field.name for field in fields(MatrixWeightsMixtureConfig)
    }
    assert all(
        field.default is None and field.metadata["help"]
        for field in fields(MatrixWeightsMixtureConfig)
    )
    for name in (
        "MatrixWeightsMixtureConfig",
        "MatrixBiasMixtureConfig",
        "MatrixWeightsMixture",
        "MatrixBiasMixture",
    ):
        adaptive = getattr(adaptive_parameters, name)
        external = getattr(parametric, name)
        assert adaptive is not external
        assert adaptive.__module__.startswith(
            "emperor.augmentations.adaptive_parameters."
        )
        assert external.__module__.startswith("emperor.parametric._mixtures.")
