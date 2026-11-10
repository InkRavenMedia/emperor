import copy
import unittest
from dataclasses import replace
from math import exp, expm1, log

import pytest
import torch
import torch.nn.functional as F

from emperor.augmentations.adaptive_parameters import (
    BankExpansionFactorOptions,
    DualModelDynamicWeightConfig,
    DynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationConfig,
    WeightNormalizationOptions,
    WeightNormalizationPolicy,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters._weights.variants.dual_model import (
    DualModelDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.hypernetwork import (
    HypernetworkDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.layered_weighted_bank import (  # noqa: E501
    LayeredWeightedBankDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.low_rank import (
    LowRankDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.single_model import (
    SingleModelDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.soft_weighted_bank import (  # noqa: E501
    SoftWeightedBankDynamicWeight,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.nn import Module


@pytest.mark.parametrize(
    "normalization",
    [
        WeightNormalizationOptions.L2_SCALE,
        WeightNormalizationOptions.RMS,
        WeightNormalizationOptions.SIGMOID_SCALE,
        WeightNormalizationOptions.SOFTSIGN,
    ],
)
@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 5), (2, 3, 4, 5)])
def test_each_depth_has_an_independent_positive_normalization_scale(
    normalization, shape
):
    policy = (
        WeightNormalizationConfig(
            generator_depth=3,
            normalization_option=normalization,
            normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
            normalization_learnable_scale_flag=True,
        )
        .build()
        .double()
    )
    assert policy.raw_scale.shape == (3,)
    torch.testing.assert_close(policy.scale, torch.ones(3, dtype=torch.float64))
    effective_scales = torch.tensor([0.5, 1.5, 3.0], dtype=torch.float64)
    with torch.no_grad():
        policy.raw_scale.copy_(torch.log(torch.expm1(effective_scales - 0.1)))
    vectors = (
        torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
        .expand(shape)
        .clone()
        .requires_grad_()
    )
    unscaled = {
        WeightNormalizationOptions.L2_SCALE: F.normalize(vectors, dim=-1),
        WeightNormalizationOptions.RMS: vectors
        / (vectors.square().mean(dim=-1, keepdim=True).sqrt() + 1e-8),
        WeightNormalizationOptions.SIGMOID_SCALE: 2 * vectors.sigmoid() - 1,
        WeightNormalizationOptions.SOFTSIGN: vectors / (1 + vectors.abs()),
    }[normalization]
    depth_axis = 0 if len(shape) == 2 else 1
    scale_shape = [1] * len(shape)
    scale_shape[depth_axis] = 3
    expected = unscaled * effective_scales.reshape(scale_shape)
    actual = policy(vectors)
    torch.testing.assert_close(actual, expected)

    # A loss on one depth must update only that depth's scale.
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    actual.select(depth_axis, 1).square().sum().backward()
    assert policy.raw_scale.grad[0].item() == 0
    assert policy.raw_scale.grad[1].item() > 0
    assert policy.raw_scale.grad[2].item() == 0
    optimizer.step()
    torch.testing.assert_close(policy.scale[[0, 2]], effective_scales[[0, 2]])
    assert 0.1 < policy.scale[1].item() < effective_scales[1].item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_depth_scales_preserve_low_precision_vector_dtype(dtype):
    policy = WeightNormalizationConfig(
        generator_depth=3,
        normalization_option=WeightNormalizationOptions.SOFTSIGN,
        normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
        normalization_learnable_scale_flag=True,
    ).build()
    vectors = torch.ones(2, 3, 4, dtype=dtype, requires_grad=True)
    result = policy(vectors)
    assert result.dtype is dtype
    torch.testing.assert_close(result, torch.full_like(vectors, 0.5))
    result.sum().backward()
    assert torch.isfinite(policy.raw_scale.grad).all()
    assert torch.all(policy.raw_scale.grad > 0)


@pytest.mark.parametrize("shape", [(), (3,), (2, 5), (2, 5, 3), (2, 5, 4, 3)])
def test_depth_scales_reject_shapes_without_the_configured_depth_axis(shape):
    policy = WeightNormalizationConfig(
        generator_depth=3,
        normalization_option=WeightNormalizationOptions.SOFTSIGN,
        normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
        normalization_learnable_scale_flag=True,
    ).build()
    with pytest.raises(ValueError, match="Normalization expects 3 depths"):
        policy(torch.ones(shape))


@pytest.mark.parametrize("depth", [0, -1])
def test_normalization_rejects_nonpositive_generator_depth(depth):
    with pytest.raises(ValueError, match="generator_depth"):
        WeightNormalizationConfig(
            generator_depth=depth,
            normalization_option=WeightNormalizationOptions.SOFTSIGN,
            normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
            normalization_learnable_scale_flag=True,
        ).build()


@pytest.mark.parametrize("normalization", list(WeightNormalizationOptions))
def test_after_depth_sum_rejects_learnable_scales(normalization):
    config = WeightNormalizationConfig(
        generator_depth=3,
        normalization_option=normalization,
        normalization_position_option=WeightNormalizationPositionOptions.AFTER_DEPTH_SUM,
        normalization_learnable_scale_flag=True,
    )
    with pytest.raises(
        ValueError,
        match="normalization_learnable_scale_flag must be False for AFTER_DEPTH_SUM",
    ):
        config.build()


def test_depth_sum_scale_restriction_validates_merged_overrides():
    config = WeightNormalizationConfig(
        generator_depth=3,
        normalization_option=WeightNormalizationOptions.SOFTSIGN,
        normalization_position_option=WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
        normalization_learnable_scale_flag=True,
    )
    with pytest.raises(ValueError, match="must be False for AFTER_DEPTH_SUM"):
        config.build(
            WeightNormalizationConfig(
                normalization_position_option=WeightNormalizationPositionOptions.AFTER_DEPTH_SUM
            )
        )
    after_sum = config.build(
        WeightNormalizationConfig(
            normalization_position_option=WeightNormalizationPositionOptions.AFTER_DEPTH_SUM,
            normalization_learnable_scale_flag=False,
        )
    )
    assert after_sum.raw_scale is None
    assert after_sum.scale is None
    assert "raw_scale" not in after_sum.state_dict()
    assert after_sum.clamp_limit.requires_grad is False
    matrix = torch.tensor([[[-2.0, 0.5], [1.0, 3.0]]])
    torch.testing.assert_close(
        after_sum.normalize_after_depth_sum(matrix), matrix / (1 + matrix.abs())
    )
    with pytest.raises(ValueError, match="must be False for AFTER_DEPTH_SUM"):
        after_sum.cfg.build(
            WeightNormalizationConfig(normalization_learnable_scale_flag=True)
        )
    restored = after_sum.cfg.build(
        WeightNormalizationConfig(
            normalization_position_option=WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
            normalization_learnable_scale_flag=True,
        )
    )
    assert restored.raw_scale.shape == (3,)
    assert restored.raw_scale.requires_grad
    assert (
        config.normalization_position_option
        is WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT
    )
    assert config.normalization_learnable_scale_flag is True


class TestWeightHandlerForward(unittest.TestCase):
    def preset(
        self,
        config_cls: type[DynamicWeightConfig] = DualModelDynamicWeightConfig,
        input_dim: int = 12,
        hidden_dim: int = 36,
        output_dim: int = 24,
        bias_flag: bool = True,
        generator_depth: int = 2,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag: bool = True,
        normalization_option: WeightNormalizationOptions = (
            WeightNormalizationOptions.L2_SCALE
        ),
        normalization_position_option: WeightNormalizationPositionOptions = (
            WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
        ),
        normalization_learnable_scale_flag: bool = True,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_dropout_probability: float = 0.1,
        bank_expansion_factor: BankExpansionFactorOptions | None = None,
        decay_schedule: WeightDecayScheduleOptions = (
            WeightDecayScheduleOptions.DISABLED
        ),
        decay_rate: float = 0.0,
        decay_warmup_batches: int = 0,
    ) -> DynamicWeightConfig:
        common_kwargs = dict(
            input_dim=input_dim,
            output_dim=output_dim,
            generator_depth=generator_depth,
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
            model_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=2,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_postprocessing_flag=apply_output_postprocessing_flag,
                layer_config=LayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    activation=stack_activation,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=stack_dropout_probability,
                    gate_config=None,
                    halting_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                    ),
                ),
            ),
        )
        if config_cls in {
            SingleModelDynamicWeightConfig,
            DualModelDynamicWeightConfig,
            LowRankDynamicWeightConfig,
            HypernetworkDynamicWeightConfig,
        }:
            normalization_config = None
            if normalization_option is not WeightNormalizationOptions.DISABLED:
                position = normalization_position_option
                if config_cls in (
                    LowRankDynamicWeightConfig,
                    HypernetworkDynamicWeightConfig,
                ):
                    position = WeightNormalizationPositionOptions.DISABLED
                normalization_config = WeightNormalizationConfig(
                    normalization_option=normalization_option,
                    normalization_learnable_scale_flag=normalization_learnable_scale_flag,
                    normalization_position_option=position,
                )
            return config_cls(
                **common_kwargs,
                normalization_config=normalization_config,
            )
        if config_cls in {
            LayeredWeightedBankDynamicWeightConfig,
            SoftWeightedBankDynamicWeightConfig,
        }:
            return config_cls(
                **common_kwargs,
                bank_expansion_factor=bank_expansion_factor,
            )
        return config_cls(**common_kwargs)

    def weight_cases(
        self,
    ) -> list[tuple[type[DynamicWeightConfig], type, int]]:
        input_dim = 12
        output_dim = 24
        return [
            (
                SingleModelDynamicWeightConfig,
                SingleModelDynamicWeight,
                input_dim,
            ),
            (
                DualModelDynamicWeightConfig,
                DualModelDynamicWeight,
                output_dim,
            ),
            (
                LowRankDynamicWeightConfig,
                LowRankDynamicWeight,
                output_dim,
            ),
            (
                HypernetworkDynamicWeightConfig,
                HypernetworkDynamicWeight,
                output_dim,
            ),
            (
                LayeredWeightedBankDynamicWeightConfig,
                LayeredWeightedBankDynamicWeight,
                output_dim,
            ),
            (
                SoftWeightedBankDynamicWeightConfig,
                SoftWeightedBankDynamicWeight,
                output_dim,
            ),
        ]

    def test_disabled_method_has_no_policy_parameters_or_checkpoint_state(self):
        for config_type, _, _ in self.weight_cases():
            with self.subTest(config_type=config_type):
                is_bank = config_type in (
                    LayeredWeightedBankDynamicWeightConfig,
                    SoftWeightedBankDynamicWeightConfig,
                )
                model = self.preset(
                    config_cls=config_type,
                    input_dim=3,
                    output_dim=3,
                    generator_depth=2,
                    normalization_option=WeightNormalizationOptions.DISABLED,
                    bank_expansion_factor=(
                        BankExpansionFactorOptions.FACTOR_OF_TWO if is_bank else None
                    ),
                ).build()
                self.assertIsNone(model.normalization_policy)
                for names in (
                    dict(model.named_modules()),
                    dict(model.named_parameters()),
                    model.state_dict(),
                ):
                    self.assertFalse(
                        any(name.startswith("normalization_policy") for name in names)
                    )
                inputs = torch.randn(2, 2, 3, requires_grad=True)
                outputs = torch.randn(2, 2, 3, requires_grad=True)
                self.assertIs(model._apply_normalization_transform(inputs), inputs)
                raw_matrices = inputs.unsqueeze(-1) * outputs.unsqueeze(-2)
                actual = model._compute_dynamic_weights(
                    model._compute_outer_product(inputs, outputs)
                )
                expected = raw_matrices.sum(dim=1)
                torch.testing.assert_close(actual, expected)
                actual_gradients = torch.autograd.grad(
                    actual.square().sum(), (inputs, outputs)
                )
                expected_gradients = torch.autograd.grad(
                    expected.square().sum(), (inputs, outputs)
                )
                for actual_gradient, expected_gradient in zip(
                    actual_gradients, expected_gradients, strict=True
                ):
                    torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_disabled_method_override_omits_policy_without_mutating_config(self):
        config = self.preset(normalization_option=WeightNormalizationOptions.SOFTSIGN)
        model = replace(config, normalization_config=None).build()
        self.assertIsNone(model.normalization_policy)
        self.assertIs(
            config.normalization_config.normalization_option,
            WeightNormalizationOptions.SOFTSIGN,
        )
        self.assertIsInstance(
            config.build().normalization_policy, WeightNormalizationPolicy
        )

    def test_nested_normalization_config_controls_policy_and_inherits_depth(self):
        normalization = WeightNormalizationConfig(
            normalization_option=WeightNormalizationOptions.SOFTSIGN,
            normalization_position_option=WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
            normalization_learnable_scale_flag=True,
        )
        config = replace(self.preset(), normalization_config=normalization)
        model = config.build()
        self.assertIs(model.normalization_config, normalization)
        self.assertEqual(
            model.normalization_policy.generator_depth, config.generator_depth
        )
        self.assertEqual(
            model.normalization_policy.raw_scale.shape, (config.generator_depth,)
        )
        self.assertIsNone(normalization.generator_depth)
        disabled = replace(config, normalization_config=None).build()
        self.assertIsNone(disabled.normalization_config)
        self.assertIsNone(disabled.normalization_policy)
        self.assertFalse(
            any(
                name.startswith("normalization_policy")
                for name in disabled.state_dict()
            )
        )
        with self.assertRaisesRegex(ValueError, "normalization_config=None"):
            replace(
                config,
                normalization_config=replace(
                    normalization,
                    normalization_option=WeightNormalizationOptions.DISABLED,
                ),
            ).build()
        with self.assertRaisesRegex(TypeError, "normalization_config"):
            replace(config, normalization_config=object()).build()
        with self.assertRaisesRegex(
            ValueError, "normalization_learnable_scale_flag is required"
        ):
            replace(
                config,
                normalization_config=replace(
                    normalization, normalization_learnable_scale_flag=None
                ),
            ).build()

    def test_unsupported_weight_variants_reject_any_normalization_config(self):
        from support.adaptive_generation import weight_mixture_config

        configurations = [
            self.preset(
                config_cls=variant,
                bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_ONE,
            )
            for variant in (
                LayeredWeightedBankDynamicWeightConfig,
                SoftWeightedBankDynamicWeightConfig,
            )
        ]
        configurations.append(weight_mixture_config())
        for config in configurations:
            self.assertIsNone(config.build().normalization_config)
            for method in WeightNormalizationOptions:
                with self.subTest(variant=type(config), method=method):
                    normalization = WeightNormalizationConfig(
                        normalization_option=method,
                        normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
                        normalization_learnable_scale_flag=False,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        type(config).__name__
                        + " does not support normalization_config",
                    ):
                        replace(config, normalization_config=normalization).build()

    def test_fixed_stage_variants_reject_configurable_normalization_positions(self):
        from support.adaptive_generation import modulated_config

        configurations = [
            self.preset(config_cls=variant)
            for variant in (LowRankDynamicWeightConfig, HypernetworkDynamicWeightConfig)
        ]
        configurations.append(
            modulated_config(normalization_option=WeightNormalizationOptions.SOFTSIGN)
        )
        for config in configurations:
            for position in WeightNormalizationPositionOptions:
                if position is WeightNormalizationPositionOptions.DISABLED:
                    continue
                with self.subTest(variant=type(config), position=position):
                    with self.assertRaisesRegex(
                        ValueError, "fixed normalization stage"
                    ):
                        replace(
                            config,
                            normalization_config=replace(
                                config.normalization_config,
                                normalization_position_option=position,
                            ),
                        ).build()

    def test_single_model_handler_forward(self):
        batch_size = 2
        dim = 12
        cfg = self.preset(
            config_cls=SingleModelDynamicWeightConfig,
            input_dim=dim,
            output_dim=dim,
        )
        weight_params = Module()._init_parameter_bank((dim, dim))
        input_tensor = torch.randn(batch_size, dim)
        model = SingleModelDynamicWeight(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, dim, dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))

    def test_dual_model_handler_forward(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            config_cls=DualModelDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        input_tensor = torch.randn(batch_size, input_dim)
        model = DualModelDynamicWeight(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))

    def test_low_rank_handler_forward(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            config_cls=LowRankDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        input_tensor = torch.randn(batch_size, input_dim)
        model = LowRankDynamicWeight(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))

    def test_hypernetwork_handler_forward(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            config_cls=HypernetworkDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        input_tensor = torch.randn(batch_size, input_dim)
        model = HypernetworkDynamicWeight(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))

    def test_weighted_bank_handler_forward(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        bank_expansion_factor = BankExpansionFactorOptions.FACTOR_OF_THREE
        cfg = self.preset(
            config_cls=LayeredWeightedBankDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
            generator_depth=2,
            bank_expansion_factor=bank_expansion_factor,
        )
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        input_tensor = torch.randn(batch_size, input_dim)
        model = LayeredWeightedBankDynamicWeight(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))

    def test_soft_weighted_bank_handler_forward(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        bank_expansion_factor = BankExpansionFactorOptions.FACTOR_OF_THREE
        cfg = self.preset(
            config_cls=SoftWeightedBankDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
            generator_depth=2,
            bank_expansion_factor=bank_expansion_factor,
        )
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        input_tensor = torch.randn(batch_size, input_dim)
        model = SoftWeightedBankDynamicWeight(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))

    def test_compute_raw_outer_product_values(self):
        batch_size = 2
        generator_depth = 3
        input_dim = 4
        output_dim = 5
        cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
        model = DualModelDynamicWeight(cfg)
        input_vectors = torch.randn(batch_size, generator_depth, input_dim)
        output_vectors = torch.randn(batch_size, generator_depth, output_dim)
        result = model._compute_raw_outer_product(input_vectors, output_vectors)
        self.assertEqual(
            result.shape, (batch_size, generator_depth, input_dim, output_dim)
        )
        for b in range(batch_size):
            for k in range(generator_depth):
                expected = input_vectors[b, k].unsqueeze(1) * output_vectors[
                    b, k
                ].unsqueeze(0)
                self.assertTrue(
                    torch.allclose(result[b, k], expected, atol=1e-6),
                    f"batch={b}, depth={k}",
                )

    def test_compute_dynamic_weights_sums_over_depth(self):
        batch_size = 2
        generator_depth = 3
        input_dim = 4
        output_dim = 5
        outer_product = torch.randn(batch_size, generator_depth, input_dim, output_dim)
        cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
        model = DualModelDynamicWeight(cfg)
        result = model._compute_dynamic_weights(outer_product)
        expected = outer_product.sum(dim=1)
        self.assertEqual(result.shape, (batch_size, input_dim, output_dim))
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_compute_outer_product_raises_on_unknown_position(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            normalization_option=WeightNormalizationOptions.L2_SCALE,
            normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
        )
        model = DualModelDynamicWeight(cfg)
        model.normalization_policy.normalization_position_option = "invalid_position"
        input_vectors = torch.randn(batch_size, generator_depth, input_dim)
        output_vectors = torch.randn(batch_size, generator_depth, output_dim)
        with self.assertRaises(ValueError):
            model._compute_outer_product(input_vectors, output_vectors)

    def test_normalization_position_after_outer_product(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        valid_normalizations = [
            WeightNormalizationOptions.L2_SCALE,
            WeightNormalizationOptions.RMS,
            WeightNormalizationOptions.CLAMP,
            WeightNormalizationOptions.SOFT_CLAMP,
            WeightNormalizationOptions.SIGMOID_SCALE,
        ]
        for normalization in valid_normalizations:
            message = f"normalization={normalization}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    generator_depth=generator_depth,
                    normalization_option=normalization,
                    normalization_position_option=WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
                )
                model = DualModelDynamicWeight(cfg)
                input_vectors = torch.randn(batch_size, generator_depth, input_dim)
                output_vectors = torch.randn(batch_size, generator_depth, output_dim)
                result = model._compute_outer_product(input_vectors, output_vectors)
                raw = model._compute_raw_outer_product(input_vectors, output_vectors)
                expected = model._apply_normalization_transform(raw)
                self.assertTrue(torch.equal(result, expected))

    def test_normalization_position_before_outer_product(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        valid_normalizations = [
            WeightNormalizationOptions.L2_SCALE,
            WeightNormalizationOptions.RMS,
            WeightNormalizationOptions.CLAMP,
            WeightNormalizationOptions.SOFT_CLAMP,
            WeightNormalizationOptions.SIGMOID_SCALE,
        ]
        for normalization in valid_normalizations:
            message = f"normalization={normalization}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    generator_depth=generator_depth,
                    normalization_option=normalization,
                    normalization_position_option=WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT,
                )
                model = DualModelDynamicWeight(cfg)
                input_vectors = torch.randn(batch_size, generator_depth, input_dim)
                output_vectors = torch.randn(batch_size, generator_depth, output_dim)
                result = model._compute_outer_product(input_vectors, output_vectors)
                normalized_input = model._apply_normalization_transform(input_vectors)
                normalized_output = model._apply_normalization_transform(output_vectors)
                expected = model._compute_raw_outer_product(
                    normalized_input, normalized_output
                )
                self.assertTrue(torch.equal(result, expected))

    def test_normalization_position_disabled_uses_raw_outer_product(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            normalization_option=WeightNormalizationOptions.L2_SCALE,
            normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
        )
        model = DualModelDynamicWeight(cfg)
        input_vectors = torch.randn(batch_size, generator_depth, input_dim)
        output_vectors = torch.randn(batch_size, generator_depth, output_dim)
        result = model._compute_outer_product(input_vectors, output_vectors)
        expected = model._compute_raw_outer_product(input_vectors, output_vectors)
        self.assertTrue(torch.equal(result, expected))

    def test_normalization_placement_matches_explicit_outer_products_and_gradients(
        self,
    ):
        for config_type in (
            SingleModelDynamicWeightConfig,
            DualModelDynamicWeightConfig,
        ):
            for position in WeightNormalizationPositionOptions:
                for normalization in WeightNormalizationOptions:
                    with self.subTest(
                        config_type=config_type,
                        position=position,
                        normalization=normalization,
                    ):
                        model = (
                            self.preset(
                                config_cls=config_type,
                                input_dim=3,
                                output_dim=3,
                                normalization_option=normalization,
                                normalization_position_option=position,
                                normalization_learnable_scale_flag=(
                                    position
                                    is not WeightNormalizationPositionOptions.AFTER_DEPTH_SUM
                                ),
                            )
                            .build()
                            .double()
                        )
                        policy = model.normalization_policy
                        if normalization is WeightNormalizationOptions.DISABLED:
                            self.assertIsNone(policy)
                            matrices = torch.randn(2, 2, 3, 3, dtype=torch.float64)
                            torch.testing.assert_close(
                                model._compute_dynamic_weights(matrices),
                                matrices.sum(dim=1),
                            )
                            continue
                        self.assertIs(policy.normalization_position_option, position)
                        self.assertFalse(
                            hasattr(model, "normalization_position_option")
                        )
                        with torch.no_grad():
                            if policy.raw_scale is not None:
                                policy.raw_scale.copy_(torch.tensor([-0.7, 1.7]))
                            policy.clamp_limit.fill_(1.3)
                        input_vectors = torch.tensor(
                            [[[0.25, -2.0, 1.5], [3.0, -0.5, 0.75]]],
                            dtype=torch.float64,
                            requires_grad=True,
                        )
                        if config_type is SingleModelDynamicWeightConfig:
                            output_vectors = input_vectors
                            differentiated_vectors = (input_vectors,)
                        else:
                            output_vectors = torch.tensor(
                                [[[2.0, -0.75, 0.1], [-1.5, 0.3, 4.0]]],
                                dtype=torch.float64,
                                requires_grad=True,
                            )
                            differentiated_vectors = (input_vectors, output_vectors)

                        parameter = (
                            policy.clamp_limit
                            if normalization
                            in (
                                WeightNormalizationOptions.CLAMP,
                                WeightNormalizationOptions.SOFT_CLAMP,
                            )
                            else policy.raw_scale
                        )
                        scales = None
                        if policy.raw_scale is not None:
                            scales = 0.1 + torch.logaddexp(
                                policy.raw_scale, policy.raw_scale.new_zeros(())
                            )

                        def normalize(
                            values,
                            apply_scale=True,
                            normalization=normalization,
                            parameter=parameter,
                            scales=scales,
                        ):
                            if normalization is WeightNormalizationOptions.DISABLED:
                                return values
                            if normalization is WeightNormalizationOptions.CLAMP:
                                return values.clamp(-parameter.abs(), parameter.abs())
                            if normalization is WeightNormalizationOptions.SOFT_CLAMP:
                                return parameter * torch.tanh(values / parameter)
                            if normalization is WeightNormalizationOptions.SOFTSIGN:
                                normalized = values / (1 + values.abs())
                            elif (
                                normalization
                                is WeightNormalizationOptions.SIGMOID_SCALE
                            ):
                                normalized = 2 * values.sigmoid() - 1
                            elif normalization is WeightNormalizationOptions.L2_SCALE:
                                normalized = F.normalize(values, dim=-1)
                            else:
                                normalized = values / (
                                    values.square().mean(dim=-1, keepdim=True).sqrt()
                                    + 1e-8
                                )
                            if apply_scale:
                                return normalized * scales.reshape(
                                    (1, 2) + (1,) * (values.ndim - 2)
                                )
                            return normalized

                        expected_input = input_vectors
                        expected_output = output_vectors
                        if (
                            position
                            is WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
                        ):
                            expected_input = normalize(input_vectors)
                            expected_output = normalize(output_vectors)
                        else:
                            self.assertIs(
                                policy.normalize_before_outer_product(input_vectors),
                                input_vectors,
                            )
                        expected = expected_input.unsqueeze(
                            -1
                        ) * expected_output.unsqueeze(-2)
                        if (
                            position
                            is WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT
                        ):
                            expected = normalize(expected)
                        else:
                            self.assertIs(
                                policy.normalize_after_outer_product(expected), expected
                            )
                        actual = model._compute_outer_product(
                            input_vectors, output_vectors
                        )
                        torch.testing.assert_close(actual, expected)
                        actual = model._compute_dynamic_weights(actual)
                        expected = expected.sum(dim=1)
                        if (
                            position
                            is WeightNormalizationPositionOptions.AFTER_DEPTH_SUM
                        ):
                            expected = normalize(expected, apply_scale=False)
                        else:
                            self.assertIs(
                                policy.normalize_after_depth_sum(expected), expected
                            )
                        torch.testing.assert_close(actual, expected)
                        differentiated = differentiated_vectors
                        if parameter is not None and parameter.requires_grad:
                            differentiated = (*differentiated, parameter)
                        actual_gradients = torch.autograd.grad(
                            actual.square().sum(), differentiated, allow_unused=True
                        )
                        expected_gradients = torch.autograd.grad(
                            expected.square().sum(), differentiated, allow_unused=True
                        )
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients, strict=True
                        ):
                            if expected_gradient is None:
                                self.assertIsNone(actual_gradient)
                            else:
                                torch.testing.assert_close(
                                    actual_gradient, expected_gradient
                                )
                                self.assertTrue(torch.isfinite(actual_gradient).all())

    def test_after_depth_sum_normalizes_once_before_adding_decayed_base(self):
        for config_type, output_dim in (
            (SingleModelDynamicWeightConfig, 2),
            (DualModelDynamicWeightConfig, 3),
        ):
            with self.subTest(config_type=config_type):
                config = self.preset(
                    config_cls=config_type,
                    input_dim=2,
                    output_dim=output_dim,
                    generator_depth=2,
                    apply_output_postprocessing_flag=False,
                    stack_activation=ActivationOptions.DISABLED,
                    stack_dropout_probability=0.0,
                    normalization_option=WeightNormalizationOptions.SOFTSIGN,
                    normalization_position_option=WeightNormalizationPositionOptions.AFTER_DEPTH_SUM,
                    normalization_learnable_scale_flag=False,
                    decay_schedule=WeightDecayScheduleOptions.MULTIPLICATIVE,
                    decay_rate=0.5,
                )
                config.model_config.num_layers = 1
                model = config.build().double().eval()
                generators = (
                    (model.model,)
                    if config_type is SingleModelDynamicWeightConfig
                    else (model.input_model, model.output_model)
                )
                with torch.no_grad():
                    model._decay_policy.decay_step.fill_(1)
                    for generator in generators:
                        mapping = generator.model[0].model
                        mapping.weight_params.zero_()
                        mapping.bias_params.fill_(1.0)
                base = torch.full(
                    (2, output_dim), 10.0, dtype=torch.float64, requires_grad=True
                )
                contexts = torch.tensor([[1.0, 2.0], [-3.0, 0.5]], dtype=torch.float64)
                observed = []
                hook = model.normalization_policy.register_forward_hook(
                    lambda _module, args, _output, observed=observed: observed.append(
                        args[0].detach().clone()
                    )
                )
                try:
                    output = model(base, contexts)
                finally:
                    hook.remove()
                # Two unscaled all-one outer products sum to 2; Softsign(2) = 2/3.
                # The independently decayed base is 5 and is not normalized.
                expected = torch.full(
                    (2, 2, output_dim), 5.0 + 2 / 3, dtype=torch.float64
                )
                torch.testing.assert_close(output, expected)
                self.assertEqual(len(observed), 1)
                torch.testing.assert_close(observed[0], torch.full_like(expected, 2.0))
                self.assertIsNone(model.normalization_policy.raw_scale)
                self.assertIsNone(model.normalization_policy.scale)
                self.assertNotIn("normalization_policy.raw_scale", model.state_dict())
                torch.testing.assert_close(base, torch.full_like(base, 10.0))
                output.sum().backward()
                torch.testing.assert_close(base.grad, torch.ones_like(base))
                for generator in generators:
                    gradient = generator.model[0].model.bias_params.grad
                    self.assertIsNotNone(gradient)
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(gradient.abs().sum().item(), 0)

    def test_outer_product_generators_reject_learnable_scales_after_depth_sum(self):
        for config_type in (
            SingleModelDynamicWeightConfig,
            DualModelDynamicWeightConfig,
        ):
            for normalization in WeightNormalizationOptions:
                if normalization is WeightNormalizationOptions.DISABLED:
                    continue
                with self.subTest(config_type=config_type, normalization=normalization):
                    with self.assertRaisesRegex(
                        ValueError, "must be False for AFTER_DEPTH_SUM"
                    ):
                        self.preset(
                            config_cls=config_type,
                            input_dim=3,
                            output_dim=3,
                            normalization_option=normalization,
                            normalization_position_option=WeightNormalizationPositionOptions.AFTER_DEPTH_SUM,
                            normalization_learnable_scale_flag=True,
                        ).build()

    def test_normalization_policy_rejects_invalid_positions_at_each_entry(self):
        for position in (None, False, 0, "invalid_position"):
            with self.subTest(position=position):
                config = WeightNormalizationConfig(
                    generator_depth=2,
                    normalization_option=WeightNormalizationOptions.L2_SCALE,
                    normalization_position_option=position,
                    normalization_learnable_scale_flag=False,
                )
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "normalization_position_option"
                ):
                    WeightNormalizationPolicy(config)
                policy = self.preset().build().normalization_policy
                policy.normalization_position_option = position
                for stage in (
                    policy.normalize_before_outer_product,
                    policy.normalize_after_outer_product,
                    policy.normalize_after_depth_sum,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "Unsupported normalization_position_option"
                    ):
                        stage(torch.ones(1, 1, 3))

    def test_normalization_policy_requires_complete_config(self):
        required = {
            "generator_depth": 2,
            "normalization_option": WeightNormalizationOptions.SOFTSIGN,
            "normalization_position_option": WeightNormalizationPositionOptions.AFTER_DEPTH_SUM,
            "normalization_learnable_scale_flag": False,
        }
        for field in required:
            for omit in (False, True):
                with self.subTest(field=field, omit=omit):
                    values = required.copy()
                    if omit:
                        del values[field]
                    else:
                        values[field] = None
                    with self.assertRaisesRegex(ValueError, field + " is required"):
                        WeightNormalizationPolicy(WeightNormalizationConfig(**values))

    def test_normalization_policy_rejects_weight_configs_and_untyped_objects(self):
        for config in (DynamicWeightConfig(), self.preset(), object(), None):
            with self.subTest(config_type=type(config).__name__):
                with self.assertRaisesRegex(TypeError, "WeightNormalizationConfig"):
                    WeightNormalizationPolicy(config)

    def test_normalization_config_validates_after_overrides_without_mutating_input(
        self,
    ):
        config = WeightNormalizationConfig(
            generator_depth=2,
            normalization_option=WeightNormalizationOptions.SOFTSIGN,
            normalization_position_option=WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
        )
        for flag in (False, True):
            with self.subTest(flag=flag):
                policy = config.build(
                    WeightNormalizationConfig(normalization_learnable_scale_flag=flag)
                )
                self.assertIsNone(config.normalization_learnable_scale_flag)
                self.assertIs(policy.cfg.normalization_learnable_scale_flag, flag)
                self.assertIs(policy.raw_scale.requires_grad, flag)
                self.assertIs(policy.clamp_limit.requires_grad, flag)
                self.assertIs(policy.normalization_option, config.normalization_option)
                self.assertIs(
                    policy.normalization_position_option,
                    config.normalization_position_option,
                )

    def test_normalization_config_rejects_invalid_types(self):
        valid = {
            "generator_depth": 2,
            "normalization_option": WeightNormalizationOptions.SOFTSIGN,
            "normalization_position_option": WeightNormalizationPositionOptions.AFTER_DEPTH_SUM,
            "normalization_learnable_scale_flag": False,
        }
        for field, values in (
            ("generator_depth", (False, 1.5, "2")),
            ("normalization_option", (False, 0, "SOFTSIGN")),
            ("normalization_position_option", (False, 0, "AFTER_DEPTH_SUM")),
            ("normalization_learnable_scale_flag", (0, 1, "false")),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(TypeError, field):
                        WeightNormalizationConfig(**(valid | {field: value})).build()

    def test_fixed_normalization_stages_preserve_low_rank_and_hypernetwork_math(self):
        for config_type in (
            LowRankDynamicWeightConfig,
            HypernetworkDynamicWeightConfig,
        ):
            with self.subTest(config_type=config_type):
                model = (
                    self.preset(
                        config_cls=config_type,
                        input_dim=3,
                        output_dim=4,
                        normalization_option=WeightNormalizationOptions.SIGMOID_SCALE,
                        apply_output_postprocessing_flag=False,
                        stack_activation=ActivationOptions.DISABLED,
                        stack_dropout_probability=0.0,
                    )
                    .build()
                    .double()
                    .eval()
                )
                policy = model.normalization_policy
                self.assertIs(
                    policy.normalization_position_option,
                    WeightNormalizationPositionOptions.DISABLED,
                )
                with torch.no_grad():
                    policy.raw_scale.copy_(torch.tensor([-0.7, 1.7]))
                scale = 0.1 + torch.logaddexp(
                    policy.raw_scale, policy.raw_scale.new_zeros(())
                )
                scale = scale.reshape(1, 2, 1)
                context = torch.tensor(
                    [[-1.5, 0.25, 2.0], [0.5, -3.0, 0.75]],
                    dtype=torch.float64,
                    requires_grad=True,
                )
                base = torch.ones(3, 4, dtype=torch.float64, requires_grad=True)
                if config_type is LowRankDynamicWeightConfig:
                    input_factors = (
                        2 * model.input_model(context).sigmoid() - 1
                    ) * scale
                    output_factors = (
                        2 * model.output_model(context).sigmoid() - 1
                    ) * scale
                    expected = base + input_factors.transpose(-1, -2) @ output_factors
                else:
                    flat_components = (2 * model.model(context).sigmoid() - 1) * scale
                    expected = base + flat_components.sum(dim=1).reshape(2, 3, 4)
                actual = model(base, context)
                torch.testing.assert_close(actual, expected)
                differentiated = (context, base, policy.raw_scale)
                actual_gradients = torch.autograd.grad(
                    actual.square().sum(), differentiated
                )
                expected_gradients = torch.autograd.grad(
                    expected.square().sum(), differentiated
                )
                for actual_gradient, expected_gradient in zip(
                    actual_gradients, expected_gradients, strict=True
                ):
                    torch.testing.assert_close(actual_gradient, expected_gradient)
                    self.assertGreater(actual_gradient.abs().sum().item(), 0)

    def test_build_creates_model_for_each_leaf_config(self):
        input_dim = 12
        for config_cls, model_cls, effective_output_dim in self.weight_cases():
            with self.subTest(config_cls=config_cls.__name__):
                bank_factor = (
                    BankExpansionFactorOptions.FACTOR_OF_TWO
                    if config_cls
                    in {
                        LayeredWeightedBankDynamicWeightConfig,
                        SoftWeightedBankDynamicWeightConfig,
                    }
                    else None
                )
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=input_dim,
                    output_dim=effective_output_dim,
                    bank_expansion_factor=bank_factor,
                )
                model = cfg.build()
                self.assertIsInstance(model, model_cls)
                weight_shape = (input_dim, effective_output_dim)
                expected_shape = (2, input_dim, effective_output_dim)
                weight_params = Module()._init_parameter_bank(weight_shape)
                input_tensor = torch.randn(2, input_dim)
                output = model(weight_params, input_tensor)
                self.assertEqual(output.shape, expected_shape)

    def test_abstract_config_cannot_build(self):
        cfg = self.preset(config_cls=DynamicWeightConfig)
        with self.assertRaises(ValueError) as error:
            cfg.build()
        self.assertEqual(
            str(error.exception),
            "DynamicWeightConfig is abstract and has no registered "
            "DynamicWeight class; instantiate a concrete leaf config instead.",
        )

    def test_gradients_flow(self):
        batch_size = 2
        input_dim = 12
        for config_cls, _, effective_output_dim in self.weight_cases():
            with self.subTest(config_cls=config_cls.__name__):
                effective_weight_params = Module()._init_parameter_bank(
                    (input_dim, effective_output_dim)
                )
                expected_shape = (batch_size, input_dim, effective_output_dim)
                bank_factor = (
                    BankExpansionFactorOptions.FACTOR_OF_TWO
                    if config_cls
                    in {
                        LayeredWeightedBankDynamicWeightConfig,
                        SoftWeightedBankDynamicWeightConfig,
                    }
                    else None
                )
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=input_dim,
                    output_dim=effective_output_dim,
                    bank_expansion_factor=bank_factor,
                )
                model = cfg.build()
                input_tensor = torch.randn(batch_size, input_dim, requires_grad=True)
                output = model(effective_weight_params, input_tensor)
                self.assertEqual(output.shape, expected_shape)
                output.sum().backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad]
                non_none_grads = [g for g in grads if g is not None]
                self.assertTrue(len(non_none_grads) > 0)

    def test_each_leaf_preserves_float64_and_backpropagates_to_both_inputs(self):
        input_dim = 2
        batch_size = 3
        cases = (
            (SingleModelDynamicWeightConfig, SingleModelDynamicWeight, input_dim),
            (DualModelDynamicWeightConfig, DualModelDynamicWeight, 3),
            (LowRankDynamicWeightConfig, LowRankDynamicWeight, 3),
            (HypernetworkDynamicWeightConfig, HypernetworkDynamicWeight, 3),
            (
                LayeredWeightedBankDynamicWeightConfig,
                LayeredWeightedBankDynamicWeight,
                3,
            ),
            (
                SoftWeightedBankDynamicWeightConfig,
                SoftWeightedBankDynamicWeight,
                3,
            ),
        )

        for config_type, model_type, output_dim in cases:
            with self.subTest(model_type=model_type.__name__):
                torch.manual_seed(17)
                is_bank = config_type in {
                    LayeredWeightedBankDynamicWeightConfig,
                    SoftWeightedBankDynamicWeightConfig,
                }
                config = self.preset(
                    config_cls=config_type,
                    input_dim=input_dim,
                    hidden_dim=4,
                    output_dim=output_dim,
                    generator_depth=1,
                    apply_output_postprocessing_flag=False,
                    normalization_option=WeightNormalizationOptions.DISABLED,
                    normalization_position_option=(
                        WeightNormalizationPositionOptions.DISABLED
                    ),
                    stack_activation=ActivationOptions.DISABLED,
                    stack_dropout_probability=0.0,
                    bank_expansion_factor=(
                        BankExpansionFactorOptions.FACTOR_OF_TWO if is_bank else None
                    ),
                )
                model = model_type(config).double()
                for state_value in model.state_dict().values():
                    if torch.is_floating_point(state_value):
                        self.assertEqual(state_value.dtype, torch.float64)
                        self.assertEqual(state_value.device.type, "cpu")
                weight_params = torch.randn(
                    input_dim,
                    output_dim,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                input_tensor = torch.randn(
                    batch_size,
                    input_dim,
                    dtype=torch.float64,
                    requires_grad=True,
                )

                output = model(weight_params, input_tensor)
                output.square().sum().backward()

                self.assertEqual(output.dtype, torch.float64)
                self.assertEqual(output.device.type, "cpu")
                self.assertTrue(torch.isfinite(output).all())
                for gradient in (weight_params.grad, input_tensor.grad):
                    self.assertIsNotNone(gradient)
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertTrue(torch.any(gradient != 0))
                active_model_gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if name
                    not in {
                        "normalization_policy.raw_scale",
                        "normalization_policy.clamp_limit",
                    }
                    and parameter.grad is not None
                ]
                self.assertTrue(active_model_gradients)
                self.assertTrue(
                    any(torch.any(gradient != 0) for gradient in active_model_gradients)
                )
                if is_bank:
                    self.assertIsNotNone(model.weight_bank.grad)
                    self.assertTrue(torch.isfinite(model.weight_bank.grad).all())
                    self.assertTrue(torch.any(model.weight_bank.grad != 0))

    def test_active_normalization_parameters_receive_finite_nonzero_gradients(self):
        config = self.preset(
            input_dim=3,
            output_dim=4,
            normalization_option=WeightNormalizationOptions.SIGMOID_SCALE,
        )
        model = DualModelDynamicWeight(config).double()
        vectors = torch.tensor(
            [[[-3.0, -0.5, 1.5], [0.25, 2.0, -4.0]]],
            dtype=torch.float64,
        )
        cases = (
            (
                WeightNormalizationOptions.SIGMOID_SCALE,
                model.normalization_policy.raw_scale,
            ),
            (
                WeightNormalizationOptions.SOFT_CLAMP,
                model.normalization_policy.clamp_limit,
            ),
        )

        for option, active_parameter in cases:
            with self.subTest(option=option):
                model.zero_grad(set_to_none=True)
                model.normalization_policy.normalization_option = option
                differentiable_vectors = vectors.clone().requires_grad_()

                transformed = model._apply_normalization_transform(
                    differentiable_vectors
                )
                transformed.square().sum().backward()

                self.assertIsNotNone(active_parameter.grad)
                self.assertTrue(torch.isfinite(active_parameter.grad).all())
                self.assertTrue(torch.any(active_parameter.grad != 0))
                self.assertIsNotNone(differentiable_vectors.grad)
                self.assertTrue(torch.isfinite(differentiable_vectors.grad).all())
                self.assertTrue(torch.any(differentiable_vectors.grad != 0))

    @pytest.mark.training
    def test_each_leaf_strictly_restores_model_and_adam_state(self):
        input_dim = 2
        cases = (
            (SingleModelDynamicWeightConfig, input_dim),
            (DualModelDynamicWeightConfig, 3),
            (LowRankDynamicWeightConfig, 3),
            (HypernetworkDynamicWeightConfig, 3),
            (LayeredWeightedBankDynamicWeightConfig, 3),
            (SoftWeightedBankDynamicWeightConfig, 3),
        )

        for config_type, output_dim in cases:
            with self.subTest(config_type=config_type.__name__):
                torch.manual_seed(23)
                is_bank = config_type in {
                    LayeredWeightedBankDynamicWeightConfig,
                    SoftWeightedBankDynamicWeightConfig,
                }
                config = self.preset(
                    config_cls=config_type,
                    input_dim=input_dim,
                    hidden_dim=4,
                    output_dim=output_dim,
                    generator_depth=2,
                    apply_output_postprocessing_flag=False,
                    normalization_option=(
                        WeightNormalizationOptions.DISABLED
                        if is_bank
                        else WeightNormalizationOptions.SOFTSIGN
                    ),
                    normalization_position_option=(
                        WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT
                    ),
                    stack_activation=ActivationOptions.DISABLED,
                    stack_dropout_probability=0.0,
                    bank_expansion_factor=(
                        BankExpansionFactorOptions.FACTOR_OF_TWO if is_bank else None
                    ),
                    decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
                    decay_rate=0.2,
                    decay_warmup_batches=0,
                )
                source = config.build().double()
                if not is_bank:
                    with torch.no_grad():
                        source.normalization_policy.raw_scale.copy_(
                            torch.tensor([-0.7, 1.7], dtype=torch.float64)
                        )
                source_optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
                weight_params = torch.tensor(
                    [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]],
                    dtype=torch.float64,
                )[:, :output_dim]
                first_input = torch.tensor(
                    [[1.0, 2.0], [-0.5, 3.0]],
                    dtype=torch.float64,
                )
                continuation_input = torch.tensor(
                    [[2.0, -1.0], [0.75, 0.5]],
                    dtype=torch.float64,
                )

                def training_step(
                    model,
                    optimizer,
                    input_tensor,
                    base_weight_params,
                ):
                    optimizer.zero_grad()
                    output = model(base_weight_params, input_tensor)
                    output.square().mean().backward()
                    optimizer.step()
                    return output.detach()

                training_step(source, source_optimizer, first_input, weight_params)
                if not is_bank:
                    scale_state = source_optimizer.state[
                        source.normalization_policy.raw_scale
                    ]
                    self.assertEqual(scale_state["exp_avg"].shape, (2,))
                model_state = copy.deepcopy(source.state_dict())
                optimizer_state = copy.deepcopy(source_optimizer.state_dict())

                torch.manual_seed(101)
                restored = config.build().double()
                incompatible = restored.load_state_dict(model_state, strict=True)
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])
                restored_optimizer = torch.optim.Adam(
                    restored.parameters(),
                    lr=0.01,
                )
                restored_optimizer.load_state_dict(optimizer_state)

                self.assertTupleEqual(
                    tuple(source.state_dict()),
                    tuple(restored.state_dict()),
                )
                self.assertTupleEqual(
                    tuple(dict(source.named_parameters())),
                    tuple(dict(restored.named_parameters())),
                )
                source_output = training_step(
                    source,
                    source_optimizer,
                    continuation_input,
                    weight_params,
                )
                restored_output = training_step(
                    restored,
                    restored_optimizer,
                    continuation_input,
                    weight_params,
                )
                torch.testing.assert_close(restored_output, source_output)
                for name, source_value in source.state_dict().items():
                    torch.testing.assert_close(
                        restored.state_dict()[name],
                        source_value,
                    )

                source_optimizer_state = source_optimizer.state_dict()
                restored_optimizer_state = restored_optimizer.state_dict()
                self.assertEqual(
                    source_optimizer_state["param_groups"],
                    restored_optimizer_state["param_groups"],
                )
                for source_values, restored_values in zip(
                    source_optimizer_state["state"].values(),
                    restored_optimizer_state["state"].values(),
                    strict=True,
                ):
                    self.assertEqual(source_values.keys(), restored_values.keys())
                    for key, source_value in source_values.items():
                        restored_value = restored_values[key]
                        if torch.is_tensor(source_value):
                            torch.testing.assert_close(restored_value, source_value)
                        else:
                            self.assertEqual(restored_value, source_value)

    def test_positive_integer_generator_depths(self):
        batch_size = 2
        input_dim = 12
        for config_cls, _, effective_output_dim in self.weight_cases():
            is_bank_type = config_cls in {
                LayeredWeightedBankDynamicWeightConfig,
                SoftWeightedBankDynamicWeightConfig,
            }
            bank_factor = (
                BankExpansionFactorOptions.FACTOR_OF_TWO if is_bank_type else None
            )
            weight_shape = (input_dim, effective_output_dim)
            expected_shape = (batch_size, input_dim, effective_output_dim)
            weight_params = Module()._init_parameter_bank(weight_shape)
            for depth in (*range(1, 11), 32):
                msg = f"config_cls={config_cls.__name__}, depth={depth}"
                with self.subTest(msg=msg):
                    cfg = self.preset(
                        config_cls=config_cls,
                        input_dim=input_dim,
                        output_dim=effective_output_dim,
                        generator_depth=depth,
                        bank_expansion_factor=bank_factor,
                    )
                    model = cfg.build()
                    input_tensor = torch.randn(batch_size, input_dim)
                    output = model(weight_params, input_tensor)
                    self.assertEqual(output.shape, expected_shape)

    def test_bank_expansion_factor_variants(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        bank_config_classes = [
            LayeredWeightedBankDynamicWeightConfig,
            SoftWeightedBankDynamicWeightConfig,
        ]
        for config_cls in bank_config_classes:
            for factor in BankExpansionFactorOptions:
                is_disabled = factor == BankExpansionFactorOptions.DISABLED
                msg = f"config_cls={config_cls.__name__}, factor={factor}"
                with self.subTest(msg=msg):
                    cfg = self.preset(
                        config_cls=config_cls,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bank_expansion_factor=factor,
                    )
                    if is_disabled:
                        with self.assertRaises(ValueError):
                            cfg.build()
                    else:
                        model = cfg.build()
                        input_tensor = torch.randn(batch_size, input_dim)
                        output = model(weight_params, input_tensor)
                        expected_shape = (
                            batch_size,
                            input_dim,
                            output_dim,
                        )
                        self.assertEqual(output.shape, expected_shape)

    def test_bank_expansion_factor_field_absent_on_non_bank_leaves(self):
        non_bank_leaf_classes = [
            SingleModelDynamicWeightConfig,
            DualModelDynamicWeightConfig,
            LowRankDynamicWeightConfig,
            HypernetworkDynamicWeightConfig,
        ]
        for leaf_cls in non_bank_leaf_classes:
            with self.subTest(leaf=leaf_cls.__name__):
                with self.assertRaises(TypeError):
                    leaf_cls(
                        bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO
                    )

    def test_normalization_fields_absent_on_bank_leaf_configs(self):
        bank_leaf_classes = [
            LayeredWeightedBankDynamicWeightConfig,
            SoftWeightedBankDynamicWeightConfig,
        ]
        for leaf_cls in bank_leaf_classes:
            with self.subTest(leaf=leaf_cls.__name__, field="normalization_option"):
                with self.assertRaises(TypeError):
                    leaf_cls(normalization_option=WeightNormalizationOptions.L2_SCALE)
            with self.subTest(
                leaf=leaf_cls.__name__, field="normalization_position_option"
            ):
                with self.assertRaises(TypeError):
                    leaf_cls(
                        normalization_position_option=WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
                    )

    def test_normalization_position_field_absent_on_low_rank_and_hypernetwork(self):
        leaf_classes = [
            LowRankDynamicWeightConfig,
            HypernetworkDynamicWeightConfig,
        ]
        for leaf_cls in leaf_classes:
            with self.subTest(leaf=leaf_cls.__name__):
                with self.assertRaises(TypeError):
                    leaf_cls(
                        normalization_position_option=WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
                    )

    def test_soft_weighted_bank_squashes_all_expanded_rows_per_output_row(self):
        input_dim = 2
        output_dim = 3
        batch_size = 2
        depth = 1
        bank_factor = BankExpansionFactorOptions.FACTOR_OF_TWO
        expanded_bank_rows = input_dim * bank_factor.value
        cfg = self.preset(
            config_cls=SoftWeightedBankDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=6,
            generator_depth=depth,
            bank_expansion_factor=bank_factor,
        )
        model = SoftWeightedBankDynamicWeight(cfg)
        weight_params = torch.zeros(input_dim, output_dim)
        input_tensor = torch.randn(batch_size, input_dim)
        generated_logits = model.model(input_tensor)
        self.assertEqual(
            generated_logits.shape,
            (
                batch_size,
                depth,
                input_dim * expanded_bank_rows,
            ),
        )
        self.assertEqual(
            model.weight_bank.shape,
            (depth, expanded_bank_rows, output_dim),
        )
        raw_logits = torch.tensor(
            [[[-8.0, -8.0, -8.0, 8.0, 8.0, -8.0, -8.0, -8.0]]],
            dtype=weight_params.dtype,
        )

        class StaticDepthMapper(torch.nn.Module):
            def __init__(self, logits: torch.Tensor):
                super().__init__()
                self.logits = logits

            def forward(self, X: torch.Tensor) -> torch.Tensor:
                return self.logits.expand(X.size(0), -1, -1)

        model.model = StaticDepthMapper(raw_logits)
        with torch.no_grad():
            model.weight_bank.copy_(
                torch.tensor(
                    [
                        [
                            [1.0, 2.0, 3.0],
                            [10.0, 20.0, 30.0],
                            [4.0, 5.0, 6.0],
                            [40.0, 50.0, 60.0],
                        ]
                    ],
                    dtype=weight_params.dtype,
                )
            )

        output = model(weight_params, input_tensor)
        bank_logits = model.model(input_tensor).view(
            batch_size,
            depth,
            input_dim,
            expanded_bank_rows,
        )
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        expected_update = torch.einsum(
            "bdim,dmo->bdio", bank_distribution, model.weight_bank
        ).sum(dim=1)
        torch.testing.assert_close(output, expected_update)
        torch.testing.assert_close(
            bank_distribution.sum(dim=-1),
            torch.ones(batch_size, depth, input_dim),
        )

    def test_layered_weighted_bank_forward_applies_depth_and_factor_reduction(self):
        input_dim = 2
        output_dim = 3
        batch_size = 2
        depth = 1
        bank_factor = BankExpansionFactorOptions.FACTOR_OF_TWO
        cfg = self.preset(
            config_cls=LayeredWeightedBankDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=6,
            generator_depth=depth,
            bank_expansion_factor=bank_factor,
        )
        model = LayeredWeightedBankDynamicWeight(cfg)
        weight_params = torch.zeros(input_dim, output_dim)
        input_tensor = torch.randn(batch_size, input_dim)
        raw_logits = torch.tensor([[[2.0, -1.0, -0.5, 1.5]]])

        class StaticDepthMapper(torch.nn.Module):
            def __init__(self, logits: torch.Tensor):
                super().__init__()
                self.logits = logits

            def forward(self, X: torch.Tensor) -> torch.Tensor:
                return self.logits.expand(X.size(0), -1, -1)

        model.model = StaticDepthMapper(raw_logits)
        model.weight_bank.data = torch.tensor(
            [
                [
                    [
                        [1.0, 2.0, 3.0],
                        [10.0, 20.0, 30.0],
                        [4.0, 5.0, 6.0],
                        [40.0, 50.0, 60.0],
                    ]
                ]
            ]
        )

        output = model(weight_params, input_tensor)
        bank_distribution = torch.softmax(model.model(input_tensor), dim=-1)
        weighted_bank = model.weight_bank * bank_distribution.unsqueeze(-1)
        expected = weighted_bank.view(
            batch_size,
            depth,
            input_dim,
            bank_factor.value,
            output_dim,
        ).sum(dim=(1, 3))

        torch.testing.assert_close(output, expected)

    def test_decay_schedule_mathematical_correctness(self):
        input_dim = 12
        output_dim = 24
        decay_rate = 0.3
        num_steps = 5
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))

        schedule_expected_factor = {
            WeightDecayScheduleOptions.EXPONENTIAL: lambda step: torch.exp(
                torch.tensor(-decay_rate * step)
            ),
            WeightDecayScheduleOptions.LINEAR: lambda step: torch.clamp(
                torch.tensor(1.0 - decay_rate * step), min=0.0
            ),
            WeightDecayScheduleOptions.MULTIPLICATIVE: lambda step: torch.pow(
                torch.tensor(1.0 - decay_rate), torch.tensor(float(step))
            ),
        }

        for schedule, expected_factor_fn in schedule_expected_factor.items():
            message = f"schedule={schedule}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    decay_schedule=schedule,
                    decay_rate=decay_rate,
                )
                model = cfg.build()
                for step in range(num_steps):
                    result = model._maybe_apply_weight_decay(weight_params)
                    expected_factor = expected_factor_fn(step)
                    expected = weight_params * expected_factor
                    self.assertTrue(
                        torch.allclose(result, expected, atol=1e-6),
                        f"{message}, step={step}: expected factor="
                        f"{expected_factor.item():.6f}",
                    )

    def test_exponential_weight_decay_saturates_huge_rate_to_active_dtype(self):
        weight_params = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        cfg = self.preset(
            input_dim=2,
            output_dim=2,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=1.0e100,
        )
        model = cfg.build()

        initial = model._maybe_apply_weight_decay(weight_params)
        decayed = model._maybe_apply_weight_decay(weight_params)

        torch.testing.assert_close(initial, weight_params)
        torch.testing.assert_close(decayed, torch.zeros_like(weight_params))
        self.assertEqual(initial.dtype, weight_params.dtype)
        self.assertEqual(decayed.dtype, weight_params.dtype)
        self.assertTrue(torch.isfinite(initial).all())
        self.assertTrue(torch.isfinite(decayed).all())

    def test_decay_schedule_disabled_leaves_weights_unchanged(self):
        input_dim = 12
        output_dim = 24
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.3,
        )
        model = cfg.build()
        baseline_decay_step = model._decay_policy.decay_step.clone()
        baseline_warmup_step = model._decay_policy.warmup_step.clone()

        for _ in range(3):
            result = model._maybe_apply_weight_decay(weight_params)
            self.assertTrue(torch.equal(result, weight_params))

        self.assertTrue(
            torch.equal(model._decay_policy.decay_step, baseline_decay_step)
        )
        self.assertTrue(
            torch.equal(model._decay_policy.warmup_step, baseline_warmup_step)
        )

    def test_invalid_decay_parameters_raise(self):
        invalid_cases = [
            ("missing_rate", WeightDecayScheduleOptions.EXPONENTIAL, None, 0),
            ("zero_rate", WeightDecayScheduleOptions.EXPONENTIAL, 0.0, 0),
            ("negative_rate", WeightDecayScheduleOptions.EXPONENTIAL, -0.1, 0),
            ("linear_rate_too_large", WeightDecayScheduleOptions.LINEAR, 1.0, 0),
            (
                "multiplicative_rate_too_large",
                WeightDecayScheduleOptions.MULTIPLICATIVE,
                1.0,
                0,
            ),
            ("negative_warmup", WeightDecayScheduleOptions.EXPONENTIAL, 0.1, -1),
        ]

        for name, schedule, rate, warmup_batches in invalid_cases:
            with self.subTest(case=name):
                cfg = self.preset(
                    decay_schedule=schedule,
                    decay_rate=rate,
                    decay_warmup_batches=warmup_batches,
                )
                with self.assertRaises(ValueError):
                    cfg.build()

    def test_weight_decay_warmup_delays_decay(self):
        input_dim = 12
        output_dim = 24
        warmup_batches = 3
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=0.1,
            decay_warmup_batches=warmup_batches,
        )
        model = cfg.build()

        for step in range(warmup_batches):
            message = f"warmup_step={step}"
            with self.subTest(msg=message):
                result = model._maybe_apply_weight_decay(weight_params)
                self.assertTrue(torch.equal(result, weight_params))

        # first call after warmup: step=0, factor=1.0, no decay yet
        result = model._maybe_apply_weight_decay(weight_params)
        self.assertTrue(torch.equal(result, weight_params))
        # second call: step=1, factor < 1.0, decay applied
        result = model._maybe_apply_weight_decay(weight_params)
        self.assertFalse(torch.equal(result, weight_params))

    def test_decay_counters_frozen_in_eval_mode(self):
        input_dim = 12
        output_dim = 24
        warmup_batches = 2
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=0.1,
            decay_warmup_batches=warmup_batches,
        )
        model = cfg.build()

        model.eval()
        for _ in range(3):
            model._maybe_apply_weight_decay(weight_params)
        self.assertEqual(model._decay_policy.warmup_step.item(), 0)
        self.assertEqual(model._decay_policy.decay_step.item(), 0)

        model.train()
        for _ in range(warmup_batches):
            model._maybe_apply_weight_decay(weight_params)
        model._maybe_apply_weight_decay(weight_params)
        frozen_decay_step = model._decay_policy.decay_step.clone()
        frozen_warmup_step = model._decay_policy.warmup_step.clone()

        model.eval()
        baseline = model._maybe_apply_weight_decay(weight_params)
        for _ in range(3):
            result = model._maybe_apply_weight_decay(weight_params)
            self.assertTrue(torch.equal(result, baseline))
        self.assertTrue(torch.equal(model._decay_policy.decay_step, frozen_decay_step))
        self.assertTrue(
            torch.equal(model._decay_policy.warmup_step, frozen_warmup_step)
        )

    def test_weight_decay_schedule_raises_on_unknown_schedule(self):
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=0.1,
        )
        model = cfg.build()
        model._decay_policy.decay_schedule_option = "invalid_schedule"
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        with self.assertRaises(ValueError):
            model._maybe_apply_weight_decay(weight_params)

    def test_apply_normalization_transform_all_options(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        vectors = torch.randn(batch_size, generator_depth, input_dim)

        for normalization in WeightNormalizationOptions:
            message = f"normalization={normalization}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    generator_depth=generator_depth,
                    normalization_option=normalization,
                )
                model = DualModelDynamicWeight(cfg)
                result = model._apply_normalization_transform(vectors)
                self.assertEqual(result.shape, vectors.shape)
                if normalization == WeightNormalizationOptions.DISABLED:
                    self.assertTrue(torch.equal(result, vectors))

    def test_l2_and_rms_normalization_remain_finite_at_zero(self):
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            for normalization in (
                WeightNormalizationOptions.L2_SCALE,
                WeightNormalizationOptions.RMS,
            ):
                message = f"dtype={dtype}, normalization={normalization}"
                with self.subTest(msg=message):
                    cfg = self.preset(
                        generator_depth=1, normalization_option=normalization
                    )
                    model = DualModelDynamicWeight(cfg).to(dtype=dtype)
                    vectors = torch.zeros(
                        1,
                        1,
                        2,
                        dtype=dtype,
                        requires_grad=True,
                    )

                    result = model._apply_normalization_transform(vectors)
                    (vector_gradient,) = torch.autograd.grad(result.sum(), vectors)

                    self.assertTrue(torch.isfinite(result).all().item())
                    self.assertTrue(torch.isfinite(vector_gradient).all().item())
                    self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_l2_and_rms_normalization_preserve_extreme_vector_direction(self):
        expected_value_by_normalization = {
            WeightNormalizationOptions.L2_SCALE: 2.0**-0.5,
            WeightNormalizationOptions.RMS: 1.0,
        }

        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            maximum_finite_value = torch.finfo(dtype).max
            for (
                normalization,
                expected_value,
            ) in expected_value_by_normalization.items():
                message = f"dtype={dtype}, normalization={normalization}"
                with self.subTest(msg=message):
                    cfg = self.preset(
                        generator_depth=1, normalization_option=normalization
                    )
                    model = DualModelDynamicWeight(cfg).to(dtype=dtype)
                    vectors = torch.full(
                        (1, 1, 2),
                        maximum_finite_value / 2,
                        dtype=dtype,
                        requires_grad=True,
                    )

                    result = model._apply_normalization_transform(vectors)
                    (vector_gradient,) = torch.autograd.grad(result.sum(), vectors)

                    expected = torch.full_like(result, expected_value)
                    torch.testing.assert_close(result, expected)
                    self.assertTrue(torch.isfinite(vector_gradient).all().item())

    def test_soft_clamp_remains_finite_at_degenerate_limits(self):
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            for clamp_limit in (0.0, torch.finfo(dtype).tiny / 2):
                message = f"dtype={dtype}, clamp_limit={clamp_limit}"
                with self.subTest(msg=message):
                    cfg = self.preset(
                        normalization_option=WeightNormalizationOptions.SOFT_CLAMP,
                    )
                    model = DualModelDynamicWeight(cfg).to(dtype=dtype)
                    model.normalization_policy.clamp_limit.data.fill_(clamp_limit)
                    vectors = torch.tensor(
                        [[[0.0, 1.0, -1.0]]],
                        dtype=dtype,
                        requires_grad=True,
                    )

                    result = model._apply_normalization_transform(vectors)
                    vector_gradient, limit_gradient = torch.autograd.grad(
                        result.sum(),
                        (vectors, model.normalization_policy.clamp_limit),
                    )

                    self.assertTrue(torch.isfinite(result).all().item())
                    self.assertTrue(torch.isfinite(vector_gradient).all().item())
                    self.assertTrue(torch.isfinite(limit_gradient).all().item())
                    self.assertEqual(result[..., 0].item(), 0.0)

    def test_clamp_uses_the_learned_limit_as_a_magnitude(self):
        cfg = self.preset(
            normalization_option=WeightNormalizationOptions.CLAMP,
        )
        model = DualModelDynamicWeight(cfg)
        model.normalization_policy.clamp_limit.data.fill_(-1.0)
        vectors = torch.tensor([[[-3.0, 0.0, 3.0]]])

        result = model._apply_normalization_transform(vectors)

        expected = torch.tensor([[[-1.0, 0.0, 1.0]]])
        torch.testing.assert_close(result, expected)

    def test_apply_normalization_transform_exact_math(self):
        cfg = self.preset(
            generator_depth=1,
            normalization_option=WeightNormalizationOptions.SOFTSIGN,
        )
        model = DualModelDynamicWeight(cfg)
        vectors = torch.tensor([[[-2.0, -0.5, 0.5, 2.0]]])
        model.normalization_policy.raw_scale.data.fill_(log(expm1(1.9)))
        model.normalization_policy.clamp_limit.data.fill_(1.0)

        expected_by_option = {
            WeightNormalizationOptions.CLAMP: torch.clamp(vectors, -1.0, 1.0),
            WeightNormalizationOptions.L2_SCALE: F.normalize(vectors, dim=-1) * 2.0,
            WeightNormalizationOptions.SOFT_CLAMP: torch.tanh(vectors),
            WeightNormalizationOptions.RMS: vectors
            / (vectors.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8)
            * 2.0,
            WeightNormalizationOptions.SIGMOID_SCALE: (
                torch.sigmoid(vectors) * 2.0 - 1.0
            )
            * 2.0,
            WeightNormalizationOptions.DISABLED: vectors,
            WeightNormalizationOptions.SOFTSIGN: vectors / (1 + vectors.abs()) * 2.0,
        }

        for option, expected in expected_by_option.items():
            with self.subTest(option=option):
                model.normalization_policy.normalization_option = option
                result = model._apply_normalization_transform(vectors)
                torch.testing.assert_close(result, expected)

    def test_softsign_scale_learns_to_expand_the_output_range(self):
        policy = (
            self.preset(
                generator_depth=1,
                normalization_option=WeightNormalizationOptions.SOFTSIGN,
            )
            .build()
            .normalization_policy.double()
        )
        optimizer = torch.optim.SGD(policy.parameters(), lr=2.0)
        vectors = torch.tensor([[[-3.0, 0.0, 3.0]]], dtype=torch.float64)
        target = torch.tensor([[[-1.5, 0.0, 1.5]]], dtype=torch.float64)
        torch.testing.assert_close(
            policy(vectors),
            torch.tensor([[[-0.75, 0.0, 0.75]]], dtype=torch.float64),
        )

        F.mse_loss(policy(vectors), target).backward()
        torch.testing.assert_close(
            policy.raw_scale.grad,
            torch.tensor([-0.75 * (1 - exp(-0.9))], dtype=torch.float64),
        )
        optimizer.step()

        self.assertGreater(policy(vectors)[0, 0, 2].item(), 1.0)
        self.assertLess(policy(vectors)[0, 0, 0].item(), -1.0)

    def test_normalization_scale_has_a_positive_floor_for_every_scaled_method(self):
        for option in (
            WeightNormalizationOptions.L2_SCALE,
            WeightNormalizationOptions.RMS,
            WeightNormalizationOptions.SIGMOID_SCALE,
            WeightNormalizationOptions.SOFTSIGN,
        ):
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                config = WeightNormalizationConfig(
                    generator_depth=1,
                    normalization_option=option,
                    normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
                    normalization_learnable_scale_flag=True,
                )
                policy = config.build().to(dtype=dtype)
                vectors = torch.tensor([[-2.0, 1.0]], dtype=dtype)
                initial_output = policy(vectors).detach()
                for raw, effective in (
                    (-1000.0, 0.1),
                    (-3.0, 0.14858735157374206),
                    (0.0, 0.7931471805599453),
                    (1000.0, 1000.1),
                ):
                    with self.subTest(option=option, dtype=dtype, raw=raw):
                        with torch.no_grad():
                            policy.raw_scale.fill_(raw)
                        actual = policy(vectors)
                        torch.testing.assert_close(
                            policy.scale, torch.tensor([effective], dtype=dtype)
                        )
                        self.assertTrue(torch.isfinite(actual).all())
                        self.assertTrue((actual * vectors > 0).all())
                        torch.testing.assert_close(actual, initial_output * effective)
                        if raw == 0:
                            actual.sum().backward()
                            torch.testing.assert_close(
                                policy.raw_scale.grad,
                                (0.5 * initial_output.sum()).reshape(1),
                            )

    def test_learnable_normalization_flag_controls_every_method(self):
        for option in WeightNormalizationOptions:
            for flag in (False, True):
                with self.subTest(option=option, flag=flag):
                    config = WeightNormalizationConfig(
                        generator_depth=1,
                        normalization_option=option,
                        normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
                        normalization_learnable_scale_flag=flag,
                    )
                    if option is WeightNormalizationOptions.DISABLED:
                        with self.assertRaisesRegex(
                            ValueError, "normalization_config=None"
                        ):
                            config.build()
                        continue
                    policy = WeightNormalizationPolicy(config).double()
                    expected_learnable = flag is True
                    self.assertIs(policy.raw_scale.requires_grad, expected_learnable)
                    self.assertIs(policy.clamp_limit.requires_grad, expected_learnable)
                    vectors = torch.tensor(
                        [[-3.0, 0.25, 2.0]], dtype=torch.float64, requires_grad=True
                    )
                    policy(vectors).square().sum().backward()
                    active = (
                        policy.clamp_limit
                        if option
                        in (
                            WeightNormalizationOptions.CLAMP,
                            WeightNormalizationOptions.SOFT_CLAMP,
                        )
                        else policy.raw_scale
                    )
                    if (
                        expected_learnable
                        and option is not WeightNormalizationOptions.DISABLED
                    ):
                        self.assertIsNotNone(active.grad)
                        self.assertGreater(active.grad.abs().item(), 0)
                    else:
                        self.assertIsNone(policy.raw_scale.grad)
                        self.assertIsNone(policy.clamp_limit.grad)
                    self.assertTrue(torch.isfinite(vectors.grad).all())

    def test_learnable_normalization_flag_rejects_non_booleans(self):
        for flag in (None, 0, 1, "true", [], object()):
            with self.subTest(flag=flag):
                config = self.preset(normalization_learnable_scale_flag=flag)
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "normalization_learnable_scale_flag"
                ):
                    config.build()

    def test_dimension_overrides_preserve_the_learnable_normalization_flag(self):
        for flag in (False, True):
            config = self.preset(normalization_learnable_scale_flag=flag)
            model = config.build(
                DualModelDynamicWeightConfig(input_dim=3, output_dim=4)
            )
            self.assertIs(model.normalization_policy.raw_scale.requires_grad, flag)
            changed = config.build(
                DualModelDynamicWeightConfig(
                    normalization_config=replace(
                        config.normalization_config,
                        normalization_learnable_scale_flag=not flag,
                    ),
                )
            )
            self.assertIs(
                changed.normalization_policy.raw_scale.requires_grad, not flag
            )
            self.assertIs(
                config.normalization_config.normalization_learnable_scale_flag, flag
            )

    def test_apply_normalization_transform_raises_on_unknown_option(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
        model = DualModelDynamicWeight(cfg)
        model.normalization_policy.normalization_option = "invalid_normalization"
        vectors = torch.randn(batch_size, generator_depth, input_dim)
        with self.assertRaises(ValueError):
            model._apply_normalization_transform(vectors)

    def test_normalization_submodule_owns_config_parameters_and_module_lifecycle(self):
        config = self.preset(normalization_option=WeightNormalizationOptions.DISABLED)
        overrides = DualModelDynamicWeightConfig(
            normalization_config=WeightNormalizationConfig(
                normalization_option=WeightNormalizationOptions.SIGMOID_SCALE,
                normalization_position_option=WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
                normalization_learnable_scale_flag=True,
            ),
        )
        model = DualModelDynamicWeight(config, overrides).double()
        policy = model.normalization_policy
        self.assertIsInstance(policy, WeightNormalizationPolicy)
        self.assertIs(model.get_submodule("normalization_policy"), policy)
        self.assertIsInstance(policy.cfg, WeightNormalizationConfig)
        self.assertEqual(policy.cfg.generator_depth, model.cfg.generator_depth)
        self.assertEqual(policy.raw_scale.shape, (model.cfg.generator_depth,))
        self.assertIs(
            policy.cfg.normalization_option,
            model.normalization_config.normalization_option,
        )
        self.assertIs(
            policy.cfg.normalization_position_option,
            model.normalization_config.normalization_position_option,
        )
        self.assertIs(
            policy.cfg.normalization_learnable_scale_flag,
            model.normalization_config.normalization_learnable_scale_flag,
        )
        self.assertIs(
            policy.normalization_option, WeightNormalizationOptions.SIGMOID_SCALE
        )
        self.assertIsNone(config.normalization_config)
        self.assertIs(
            policy.normalization_position_option,
            WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
        )
        self.assertEqual(
            set(dict(policy.named_parameters())), {"raw_scale", "clamp_limit"}
        )
        for name, parameter in policy.named_parameters():
            self.assertFalse(hasattr(model, name))
            self.assertEqual(parameter.dtype, torch.float64)
            self.assertIs(
                dict(model.named_parameters())[f"normalization_policy.{name}"],
                parameter,
            )
            self.assertEqual(
                sum(candidate is parameter for candidate in model.parameters()), 1
            )

        model.eval()
        self.assertFalse(policy.training)
        model.train()
        self.assertTrue(policy.training)
        visited_modules = []
        model.apply(visited_modules.append)
        self.assertIn(policy, visited_modules)

        observed_outputs = []
        handle = policy.register_forward_hook(
            lambda _module, _inputs, output: observed_outputs.append(output)
        )
        try:
            vectors = torch.tensor(
                [[-2.0, 0.25, 3.0], [0.5, -1.0, 2.0]], dtype=torch.float64
            )
            output = model._apply_normalization_transform(vectors)
            torch.testing.assert_close(output, 2 * vectors.sigmoid() - 1)
            self.assertEqual(len(observed_outputs), 1)
            self.assertIs(observed_outputs[0], output)
        finally:
            handle.remove()

    def test_normalization_uses_current_parameters_after_state_assignment(self):
        model = DualModelDynamicWeight(self.preset()).double()
        state = copy.deepcopy(model.state_dict())
        state["normalization_policy.raw_scale"].copy_(torch.tensor([2.5, 0.5]))
        state["normalization_policy.clamp_limit"].fill_(1.5)
        original_scale = model.normalization_policy.raw_scale
        original_clamp_limit = model.normalization_policy.clamp_limit
        model.load_state_dict(state, strict=True, assign=True)
        self.assertIsNot(model.normalization_policy.raw_scale, original_scale)
        self.assertIsNot(model.normalization_policy.clamp_limit, original_clamp_limit)
        self.assertEqual(tuple(model.state_dict()), tuple(state))

        for option in (
            WeightNormalizationOptions.SIGMOID_SCALE,
            WeightNormalizationOptions.SOFT_CLAMP,
            WeightNormalizationOptions.SOFTSIGN,
        ):
            with self.subTest(option=option):
                model.normalization_policy.normalization_option = option
                vectors = torch.tensor(
                    [[[0.25, 1.5, 3.0], [0.5, -1.0, 2.0]]],
                    dtype=torch.float64,
                    requires_grad=True,
                )
                if option is WeightNormalizationOptions.SIGMOID_SCALE:
                    parameter = model.normalization_policy.raw_scale
                    scale = 0.1 + torch.logaddexp(parameter, parameter.new_zeros(()))
                    expected = (2 * vectors.sigmoid() - 1) * scale.reshape(1, 2, 1)
                elif option is WeightNormalizationOptions.SOFTSIGN:
                    parameter = model.normalization_policy.raw_scale
                    scale = 0.1 + torch.logaddexp(parameter, parameter.new_zeros(()))
                    expected = vectors / (1 + vectors.abs()) * scale.reshape(1, 2, 1)
                else:
                    parameter = model.normalization_policy.clamp_limit
                    expected = parameter * torch.tanh(vectors / parameter)
                actual = model._apply_normalization_transform(vectors)
                torch.testing.assert_close(actual, expected)
                actual_gradients = torch.autograd.grad(
                    actual.sum(), (vectors, parameter)
                )
                expected_gradients = torch.autograd.grad(
                    expected.sum(), (vectors, parameter)
                )
                for actual_gradient, expected_gradient in zip(
                    actual_gradients, expected_gradients, strict=True
                ):
                    torch.testing.assert_close(actual_gradient, expected_gradient)
                    self.assertGreater(actual_gradient.abs().sum().item(), 0)

    def test_init_model_accepts_depth_mapping_handler_override(self):
        input_dim = 12
        output_dim = 24
        cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
        model = DualModelDynamicWeight(cfg)

        overrides = DepthMappingHandlerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        generator = model._init_model(overrides)

        self.assertIsInstance(generator, DepthMappingLayerStack)
        self.assertEqual(generator.input_dim, input_dim)
        self.assertEqual(generator.output_dim, output_dim)

    def test_weighted_bank_requires_bank_expansion_factor_enum(self):
        bank_cases = (
            (
                LayeredWeightedBankDynamicWeightConfig,
                "LayeredWeightedBankDynamicWeight",
            ),
            (
                SoftWeightedBankDynamicWeightConfig,
                "SoftWeightedBankDynamicWeight",
            ),
        )
        for config_type, model_name in bank_cases:
            for factor in (None, 0, -1):
                with self.subTest(config_type=config_type.__name__, factor=factor):
                    config = self.preset(
                        config_cls=config_type,
                        bank_expansion_factor=factor,
                    )
                    with self.assertRaises(ValueError) as error:
                        config.build()
                    self.assertEqual(
                        str(error.exception),
                        f"{model_name} requires bank_expansion_factor to be a "
                        "BankExpansionFactorOptions value, "
                        f"received {factor!r}.",
                    )

            with self.subTest(
                config_type=config_type.__name__,
                factor=BankExpansionFactorOptions.DISABLED,
            ):
                config = self.preset(
                    config_cls=config_type,
                    bank_expansion_factor=BankExpansionFactorOptions.DISABLED,
                )
                with self.assertRaises(ValueError) as error:
                    config.build()
                self.assertEqual(
                    str(error.exception),
                    f"{model_name} requires bank_expansion_factor > 0, received "
                    "BankExpansionFactorOptions.DISABLED. Use FACTOR_OF_ONE, "
                    "FACTOR_OF_TWO, FACTOR_OF_THREE, or FACTOR_OF_FOUR.",
                )

    def test_single_model_raises_when_dims_not_square(self):
        cfg = self.preset(input_dim=12, output_dim=24)
        with self.assertRaises(ValueError):
            SingleModelDynamicWeight(cfg)
