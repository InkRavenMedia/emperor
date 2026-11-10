"""Behavioral contracts for normalized per-depth broadcast-add weight updates."""

from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from math import exp, expm1, log

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    BroadcastSumDynamicWeightConfig,
    MeanGroupingConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationConfig,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from support.adaptive_grouping import linear_stack_config


def broadcast_config(**overrides):
    normalization = WeightNormalizationConfig(
        normalization_option=overrides.pop(
            "normalization_option", WeightNormalizationOptions.SOFTSIGN
        ),
        normalization_learnable_scale_flag=overrides.pop(
            "normalization_learnable_scale_flag", True
        ),
        normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
    )
    return BroadcastSumDynamicWeightConfig(
        **(
            dict(
                input_dim=2,
                output_dim=3,
                generator_depth=2,
                normalization_config=normalization,
                decay_schedule=WeightDecayScheduleOptions.DISABLED,
                decay_rate=0.0,
                decay_warmup_batches=0,
                model_config=linear_stack_config(2, 3),
            )
            | overrides
        )
    )


def assign_two_depth_generators(model):
    input_mapping = model.input_model.model[0].model
    output_mapping = model.output_model.model[0].model
    projection = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    with torch.no_grad():
        input_mapping.weight_params.copy_(torch.stack([torch.eye(2), 2 * torch.eye(2)]))
        input_mapping.bias_params.copy_(torch.tensor([[0.0, 0.0], [1.0, -1.0]]))
        output_mapping.weight_params.copy_(torch.stack([projection, -projection]))
        output_mapping.bias_params.copy_(
            torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 1.0]])
        )


@pytest.mark.parametrize("options", [{}, {"normalization_config": None}])
def test_missing_normalization_is_rejected(options):
    config = BroadcastSumDynamicWeightConfig(
        input_dim=2,
        output_dim=3,
        generator_depth=2,
        decay_schedule=WeightDecayScheduleOptions.DISABLED,
        decay_rate=0.0,
        decay_warmup_batches=0,
        model_config=linear_stack_config(2, 3),
        **options,
    )
    with pytest.raises(
        ValueError,
        match="BroadcastSumDynamicWeight requires normalization_config",
    ):
        config.build()


def test_dimension_overrides_preserve_selected_normalization():
    config = broadcast_config(normalization_option=WeightNormalizationOptions.RMS)
    overrides = BroadcastSumDynamicWeightConfig(input_dim=3, output_dim=4)
    model = config.build(overrides)
    assert (
        model.normalization_policy.normalization_option
        is WeightNormalizationOptions.RMS
    )
    assert model(torch.zeros(3, 4), torch.ones(1, 3)).shape == (1, 3, 4)
    assert config.input_dim == 2 and config.output_dim == 3


@pytest.mark.parametrize(
    "original_normalization", [None, WeightNormalizationOptions.CLAMP]
)
def test_explicit_normalization_override_is_used_before_validation(
    original_normalization,
):
    config = broadcast_config(normalization_option=original_normalization)
    overrides = BroadcastSumDynamicWeightConfig(
        normalization_config=WeightNormalizationConfig(
            normalization_learnable_scale_flag=False,
            normalization_option=WeightNormalizationOptions.RMS,
            normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
        ),
    )
    model = config.build(overrides)
    assert (
        model.normalization_policy.normalization_option
        is WeightNormalizationOptions.RMS
    )
    assert config.normalization_config.normalization_option is original_normalization


@pytest.mark.parametrize(
    "normalization",
    [
        option
        for option in WeightNormalizationOptions
        if option is not WeightNormalizationOptions.DISABLED
    ],
)
def test_selected_normalization_transforms_each_matrix_before_depth_sum(normalization):
    model = broadcast_config(normalization_option=normalization).build().double()
    assign_two_depth_generators(model)
    with torch.no_grad():
        model.normalization_policy.raw_scale.copy_(
            torch.tensor([log(expm1(0.4)), log(expm1(1.4))], dtype=torch.float64)
        )
        model.normalization_policy.clamp_limit.fill_(2.0)
    inputs = torch.tensor([[1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    # Independently calculated matrices for the two native generator depths.
    matrices = torch.tensor(
        [[[[0.0, 3.0, 6.0], [1.0, 4.0, 7.0]], [[6.0, 0.0, -1.0], [6.0, 0.0, -1.0]]]],
        dtype=torch.float64,
    )
    scales = torch.tensor([0.5, 1.5], dtype=torch.float64).reshape(1, 2, 1, 1)
    expected_by_option = {
        WeightNormalizationOptions.CLAMP: matrices.clamp(-2.0, 2.0),
        WeightNormalizationOptions.L2_SCALE: torch.nn.functional.normalize(
            matrices, dim=-1
        )
        * scales,
        WeightNormalizationOptions.SOFT_CLAMP: 2.0 * torch.tanh(matrices / 2.0),
        WeightNormalizationOptions.RMS: matrices
        / (matrices.square().mean(dim=-1, keepdim=True).sqrt() + 1e-8)
        * scales,
        WeightNormalizationOptions.SIGMOID_SCALE: (2.0 * matrices.sigmoid() - 1.0)
        * scales,
        WeightNormalizationOptions.SOFTSIGN: matrices / (1.0 + matrices.abs()) * scales,
    }
    observed_shapes = []
    handle = model.normalization_policy.register_forward_hook(
        lambda _module, args, _output: observed_shapes.append(tuple(args[0].shape))
    )
    try:
        output = model(torch.zeros(2, 3, dtype=torch.float64), inputs)
    finally:
        handle.remove()
    torch.testing.assert_close(output, expected_by_option[normalization].sum(dim=1))
    assert observed_shapes == [(1, 2, 2, 3)]
    output.square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    if normalization is WeightNormalizationOptions.SOFTSIGN:
        scale_gradient = model.normalization_policy.raw_scale.grad
        assert scale_gradient is not None and torch.isfinite(scale_gradient).all()
        softsign_matrices = matrices / (1 + matrices.abs())
        softplus_derivatives = torch.tensor(
            [1 - exp(-0.4), 1 - exp(-1.4)], dtype=torch.float64
        )
        expected_gradient = (2 * output.detach().unsqueeze(1) * softsign_matrices).sum(
            dim=(0, 2, 3)
        ) * softplus_derivatives
        torch.testing.assert_close(scale_gradient, expected_gradient)
        assert torch.all(scale_gradient.abs() > 0)
        assert scale_gradient[0] != scale_gradient[1]


def test_disabled_normalization_and_configurable_placement_are_rejected():
    with pytest.raises(ValueError, match="requires an enabled normalization_option"):
        broadcast_config(
            normalization_option=WeightNormalizationOptions.DISABLED
        ).build()
    config = broadcast_config()
    for position in WeightNormalizationPositionOptions:
        if position is WeightNormalizationPositionOptions.DISABLED:
            continue
        with pytest.raises(ValueError, match="fixed normalization stage"):
            replace(
                config,
                normalization_config=replace(
                    config.normalization_config, normalization_position_option=position
                ),
            ).build()


def test_signed_rectangular_updates_are_hand_calculated_softsign_sums():
    model = broadcast_config().build()
    assign_two_depth_generators(model)
    inputs = torch.tensor([[1.0, 2.0], [-1.0, 1.0]])
    base = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    original_base = base.clone()
    expected = base + torch.tensor(
        [
            [[6 / 7, 3 / 4, 5 / 14], [19 / 14, 4 / 5, 3 / 8]],
            [[0, -3 / 4, -1 / 12], [1 / 3, 1 / 6, 3 / 10]],
        ]
    )
    torch.testing.assert_close(model(base, inputs), expected)
    torch.testing.assert_close(base, original_base, rtol=0, atol=0)
    changed = inputs.clone()
    changed[1] += 10
    torch.testing.assert_close(model(base, changed)[0], expected[0])


def test_softsign_precedes_depth_reduction_and_preserves_depth_contributions():
    model = broadcast_config().build()
    mappings = [model.input_model.model[0].model, model.output_model.model[0].model]
    with torch.no_grad():
        for mapping in mappings:
            mapping.weight_params.zero_()
            mapping.bias_params.fill_(0.5)
    base = torch.zeros(2, 3)
    inputs = torch.zeros(1, 2)
    # Each depth adds 0.5 + 0.5, then Softsign(1) = 0.5; two depths sum to 1.
    torch.testing.assert_close(model(base, inputs), torch.ones(1, 2, 3))

    with torch.no_grad():
        for mapping in mappings:
            mapping.bias_params[0].zero_()
            mapping.bias_params[1].fill_(1.0)
    # The vector sums are unchanged, but Softsign(0) + Softsign(2) = 2/3.
    torch.testing.assert_close(model(base, inputs), torch.full((1, 2, 3), 2 / 3))


@pytest.mark.parametrize("depth", [1, 5])
@pytest.mark.parametrize("input_dim,output_dim", [(2, 3), (3, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_depth_reduction_preserves_outputs_and_both_generator_gradients(
    depth, input_dim, output_dim, dtype
):
    config = broadcast_config(
        input_dim=input_dim,
        output_dim=output_dim,
        generator_depth=depth,
        model_config=replace(linear_stack_config(input_dim, output_dim), num_layers=2),
    )
    model = config.build().to(dtype=dtype)
    reference = deepcopy(model)
    inputs = torch.randn(3, input_dim, dtype=dtype, requires_grad=True)
    base = torch.randn(input_dim, output_dim, dtype=dtype, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_base = base.detach().clone().requires_grad_()
    output = model(base, inputs)
    u = reference.input_model(reference_inputs)
    v = reference.output_model(reference_inputs)
    depth_matrices = [u[:, k, :, None] + v[:, k, None, :] for k in range(depth)]
    expected = reference_base + torch.stack(
        [matrix / (1 + matrix.abs()) for matrix in depth_matrices]
    ).sum(dim=0)
    torch.testing.assert_close(output, expected)
    assert u.shape == (3, depth, input_dim)
    assert v.shape == (3, depth, output_dim)
    assert model.input_model.model_config.num_layers == 2
    assert model.output_model.model_config.num_layers == 2
    assert not set(model.input_model.parameters()) & set(
        model.output_model.parameters()
    )
    output.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    torch.testing.assert_close(base.grad, reference_base.grad)
    for name in ("input_model", "output_model"):
        for actual, wanted in zip(
            getattr(model, name).parameters(),
            getattr(reference, name).parameters(),
            strict=True,
        ):
            assert actual.grad is not None and torch.isfinite(actual.grad).all()
            torch.testing.assert_close(actual.grad, wanted.grad)


@pytest.mark.parametrize("schedule", list(WeightDecayScheduleOptions))
def test_decay_schedules_warmup_eval_and_checkpoint_preserve_generated_update(schedule):
    model = broadcast_config(
        decay_schedule=schedule, decay_rate=0.25, decay_warmup_batches=2
    ).build()
    assign_two_depth_generators(model)
    inputs = torch.tensor([[1.0, 2.0], [-1.0, 1.0]])
    base = torch.full((2, 3), 10.0)
    expected_update = torch.tensor(
        [
            [[6 / 7, 3 / 4, 5 / 14], [19 / 14, 4 / 5, 3 / 8]],
            [[0, -3 / 4, -1 / 12], [1 / 3, 1 / 6, 3 / 10]],
        ]
    )

    def factor(step):
        return {
            WeightDecayScheduleOptions.DISABLED: 1.0,
            WeightDecayScheduleOptions.EXPONENTIAL: exp(-0.25 * step),
            WeightDecayScheduleOptions.LINEAR: max(0.0, 1.0 - 0.25 * step),
            WeightDecayScheduleOptions.MULTIPLICATIVE: 0.75**step,
        }[schedule]

    for step in (0, 0, 0, 1, 2, 3, 4, 5):
        torch.testing.assert_close(
            model(base, inputs), base * factor(step) + expected_update
        )
    disabled = schedule is WeightDecayScheduleOptions.DISABLED
    policy = model._decay_policy
    assert policy.warmup_step.item() == (0 if disabled else 2)
    assert policy.decay_step.item() == (0 if disabled else 6)
    model.eval()
    for _ in range(2):
        torch.testing.assert_close(
            model(base, inputs), base * factor(6) + expected_update
        )
    assert policy.decay_step.item() == (0 if disabled else 6)
    checkpoint = BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = deepcopy(model.cfg).build()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    restored.eval()
    torch.testing.assert_close(restored(base, inputs), model(base, inputs))
    model.train()
    restored.train()
    torch.testing.assert_close(restored(base, inputs), model(base, inputs))
    assert restored._decay_policy.decay_step.item() == (0 if disabled else 7)
    torch.testing.assert_close(base, torch.full_like(base, 10.0), rtol=0, atol=0)


def test_grouped_rows_share_compact_weights_without_cross_group_gradients():
    layer = AdaptiveLinearLayerConfig(
        input_dim=2,
        output_dim=3,
        bias_flag=False,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            weight_config=broadcast_config(
                decay_schedule=WeightDecayScheduleOptions.MULTIPLICATIVE,
                decay_rate=0.25,
            ),
            grouping_config=MeanGroupingConfig(
                scope=AdaptiveParameterGroupingScopeOptions.ROWS,
                chunk_size=2,
            ),
        ),
    ).build()
    generator = layer.adaptive_behaviour.weight_model
    assign_two_depth_generators(generator)
    with torch.no_grad():
        layer.weight_params.zero_()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 1.0]], requires_grad=True)
    observed_shapes = []
    hook = generator.register_forward_hook(
        lambda _module, _args, output: observed_shapes.append(tuple(output.shape))
    )
    try:
        output = layer(inputs)
    finally:
        hook.remove()
    # Mean contexts [2, 3] and [-1, 1]; the final padded chunk has one real row.
    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                [9 / 2, 73 / 28, 20 / 39],
                [187 / 18, 85 / 14, 31 / 26],
                [1 / 3, 11 / 12, 23 / 60],
            ]
        ),
    )
    assert observed_shapes == [(2, 2, 3)]
    assert generator._decay_policy.decay_step.item() == 1
    output[0].sum().backward()
    assert torch.count_nonzero(inputs.grad[1]) > 0
    torch.testing.assert_close(inputs.grad[2], torch.zeros(2), rtol=0, atol=0)
    changed = inputs.detach().clone()
    changed[2] += 20
    torch.testing.assert_close(layer(changed)[:2], output[:2])


def test_explicit_overrides_and_generator_postprocessing_are_preserved():
    config = broadcast_config()
    original = deepcopy(config)
    overrides = BroadcastSumDynamicWeightConfig(
        input_dim=3, output_dim=4, generator_depth=5
    )
    model = config.build(overrides)
    assert model(torch.ones(3, 4), torch.ones(2, 3)).shape == (2, 3, 4)
    assert config == original
    from emperor.layers import ActivationOptions

    generator = linear_stack_config(2, 3)
    generator.apply_output_postprocessing_flag = True
    generator.layer_config.activation = ActivationOptions.TANH
    model = broadcast_config(model_config=generator).build()
    assign_two_depth_generators(model)
    inputs = torch.tensor([[1.0, 2.0]])
    u = model.input_model(inputs)
    v = model.output_model(inputs)
    assert u.abs().max() <= 1 and v.abs().max() <= 1
    depth_matrices = u.unsqueeze(-1) + v.unsqueeze(-2)
    torch.testing.assert_close(
        model(torch.zeros(2, 3), inputs),
        (depth_matrices / (1 + depth_matrices.abs())).sum(dim=1),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"generator_depth": 0},
        {"generator_depth": True},
        {"input_dim": 0},
        {"decay_schedule": WeightDecayScheduleOptions.EXPONENTIAL, "decay_rate": -0.1},
    ],
)
def test_invalid_dimensions_depth_and_decay_keep_standard_validation(overrides):
    with pytest.raises((ValueError, TypeError)):
        broadcast_config(**overrides).build()
