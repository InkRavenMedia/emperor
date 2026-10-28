"""Adaptive matrix mixtures honor explicit routing and auxiliary-loss settings."""

from copy import deepcopy

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    MeanGroupingConfig,
)
from emperor.layers import LayerState
from support.adaptive_generation import (
    bias_mixture_config,
    matrix_sampler_config,
    weight_mixture_config,
)
from support.adaptive_grouping import linear_stack_config

LOSS_OPTIONS = (
    "coefficient_of_variation_loss_weight",
    "switch_loss_weight",
    "zero_centred_loss_weight",
    "mutual_information_loss_weight",
)
SAMPLER_OPTIONS = (
    "num_experts",
    "top_k",
    "threshold",
    "filter_above_threshold",
    "num_topk_samples",
    "noisy_topk_flag",
    "normalize_probabilities_flag",
    *LOSS_OPTIONS,
)


@pytest.mark.parametrize("config_factory", [weight_mixture_config, bias_mixture_config])
@pytest.mark.parametrize("option", SAMPLER_OPTIONS)
def test_sampler_options_must_be_explicit_before_parameter_initialization(
    config_factory, option
):
    config = config_factory()
    setattr(config.sampler_config, option, None)
    rng_state = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match=option):
        config.build()
    torch.testing.assert_close(torch.get_rng_state(), rng_state, rtol=0, atol=0)


@pytest.mark.parametrize("config_factory", [weight_mixture_config, bias_mixture_config])
@pytest.mark.parametrize(
    "option, value",
    [
        ("input_dim", None),
        ("input_dim", 4),
        ("num_experts", None),
        ("num_experts", 4),
        ("noisy_topk_flag", None),
        ("noisy_topk_flag", True),
        ("model_config", None),
    ],
)
def test_router_fields_are_required_and_conflicts_fail_without_mutation(
    config_factory, option, value
):
    config = config_factory()
    setattr(config.sampler_config.router_config, option, value)
    original = deepcopy(config)
    rng_state = torch.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError), match=option):
        config.build()
    assert config == original
    torch.testing.assert_close(torch.get_rng_state(), rng_state, rtol=0, atol=0)


@pytest.mark.parametrize("config_factory", [weight_mixture_config, bias_mixture_config])
@pytest.mark.parametrize("loss_option", LOSS_OPTIONS)
def test_configured_auxiliary_losses_reach_router_gradients(
    config_factory, loss_option
):
    config = config_factory(sampler_config=matrix_sampler_config(**{loss_option: 0.5}))
    original = deepcopy(config)
    model = config.build().double()
    router = model.sampler.router.model[0].model
    with torch.no_grad():
        router.weight_params.copy_(torch.tensor([[2.0, -1.5, 0.2], [-0.5, 1.4, 1.0]]))
        router.bias_params.copy_(torch.tensor([0.3, -0.2, 0.7]))
    context = torch.tensor(
        [[1.0, 0.25], [-0.5, 2.0], [0.75, -1.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    base = torch.zeros(model.parameter_bank.shape[1:], dtype=torch.float64)
    model(base, context)
    loss = model.get_auxiliary_loss()
    assert config == original
    assert getattr(model.cfg.sampler_config, loss_option) == 0.5
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert router.weight_params.grad.abs().sum() > 0


@pytest.mark.parametrize("config_factory", [weight_mixture_config, bias_mixture_config])
def test_noise_and_random_sampling_options_are_preserved(config_factory):
    config = config_factory(
        sampler_config=matrix_sampler_config(
            noisy_topk_flag=True,
            num_topk_samples=1,
            normalize_probabilities_flag=False,
        )
    )
    original = deepcopy(config)
    model = config.build()
    assert config == original
    assert model.cfg.sampler_config.noisy_topk_flag is True
    assert model.cfg.sampler_config.router_config.noisy_topk_flag is True
    assert model.cfg.sampler_config.num_topk_samples == 1
    assert model.cfg.sampler_config.normalize_probabilities_flag is False
    assert model.sampler.router.model.output_dim == 2 * model.num_experts
    output = model(torch.ones(model.parameter_bank.shape[1:]), torch.ones(4, 2))
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("config_factory", [weight_mixture_config, bias_mixture_config])
@pytest.mark.parametrize("threshold, filter_above", [(0.9, False), (0.2, True)])
def test_threshold_skips_generated_update_and_retains_base(
    config_factory, threshold, filter_above
):
    model = config_factory(
        sampler_config=matrix_sampler_config(
            threshold=threshold, filter_above_threshold=filter_above
        )
    ).build()
    with torch.no_grad():
        for parameter in model.sampler.router.parameters():
            parameter.zero_()
        model.parameter_bank.fill_(1.0)
    base = torch.full(model.parameter_bank.shape[1:], 0.25)
    output = model(base, torch.ones(4, 2))
    torch.testing.assert_close(output, base.expand_as(output))
    assert torch.count_nonzero(model.sampler.get_updated_skip_mask()) == 0


@pytest.mark.parametrize("grouped", [False, True])
def test_layer_state_accumulates_both_matrix_losses_without_resampling(grouped):
    config = linear_stack_config(2, 3)
    config.layer_config.layer_model_config = AdaptiveLinearLayerConfig(
        input_dim=2,
        output_dim=3,
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            grouping_config=MeanGroupingConfig(
                scope=AdaptiveParameterGroupingScopeOptions.ROWS, chunk_size=2
            )
            if grouped
            else None,
            weight_config=weight_mixture_config(
                sampler_config=matrix_sampler_config(zero_centred_loss_weight=0.3)
            ),
            bias_config=bias_mixture_config(
                sampler_config=matrix_sampler_config(switch_loss_weight=0.2)
            ),
        ),
    )
    model = config.build().double()
    augmentation = model[0].model.adaptive_behaviour
    router_calls = []
    hooks = [
        variant.sampler.router.model.register_forward_hook(
            lambda _module, _inputs, _output: router_calls.append(True)
        )
        for variant in (augmentation.weight_model, augmentation.bias_model)
    ]
    initial_loss = torch.tensor(0.75, dtype=torch.float64)
    context = torch.tensor([[1.0, 0.5], [-0.5, 1.5]], dtype=torch.float64)
    try:
        result = model(LayerState(hidden=context, loss=initial_loss))
    finally:
        for hook in hooks:
            hook.remove()
    assert len(router_calls) == 2
    expected = (
        initial_loss
        + augmentation.weight_model.sampler.get_auxiliary_loss()
        + augmentation.bias_model.sampler.get_auxiliary_loss()
    )
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    for variant in (augmentation.weight_model, augmentation.bias_model):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in variant.sampler.router.parameters()
        )
    next_result = model(LayerState(hidden=context * 0.5))
    next_loss = (
        augmentation.weight_model.sampler.get_auxiliary_loss()
        + augmentation.bias_model.sampler.get_auxiliary_loss()
    )
    torch.testing.assert_close(next_result.loss, next_loss)
    next_result.loss.backward()
    if grouped:
        empty_result = model(LayerState(hidden=context[:0], loss=initial_loss))
        torch.testing.assert_close(empty_result.loss, initial_loss)
