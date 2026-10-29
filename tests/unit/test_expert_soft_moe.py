"""Mathematical and integration contracts for optional Soft MoE execution."""

import copy
import io
import pickle
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from emperor.augmentations.adaptive_parameters import DepthMappingHandlerConfig
from emperor.experts import (
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsLayerState,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
    SoftMoEConfig,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
)
from emperor.linears import LinearLayerConfig
from emperor.transformer import FeedForwardConfig


def scoring_config(layers=1):
    return DepthMappingHandlerConfig(
        model_config=LayerStackConfig(
            hidden_dim=5,
            num_layers=layers,
            apply_output_postprocessing_flag=False,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            layer_config=layer_config(),
        )
    )


def layer_config(**changes):
    return LayerConfig(
        activation=ActivationOptions.GELU,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        layer_model_config=LinearLayerConfig(bias_flag=True),
        **changes,
    )


def expert_config(*, dimension=3, output_dim=3, experts=2, slots=2, scoring_layers=1):
    stack = LayerStackConfig(
        input_dim=dimension,
        hidden_dim=5,
        output_dim=output_dim,
        num_layers=2,
        apply_output_postprocessing_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        layer_config=layer_config(),
    )
    return MixtureOfExpertsConfig(
        input_dim=dimension,
        output_dim=output_dim,
        top_k=experts,
        num_experts=experts,
        capacity_factor=0.0,
        compute_expert_mixture_flag=True,
        weighted_parameters_flag=True,
        weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
        routing_initialization_mode=RoutingInitializationMode.LAYER,
        expert_model_config=stack,
        soft_moe_config=SoftMoEConfig(
            slots_per_expert=slots, model_config=scoring_config(scoring_layers)
        ),
    )


def model_config(*, dimension=3, layers=2):
    mixture = expert_config(dimension=dimension, output_dim=dimension)
    outer_layer = MixtureOfExpertsLayerConfig(
        activation=ActivationOptions.DISABLED,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        layer_model_config=mixture,
    )
    stack = LayerStackConfig(
        input_dim=dimension,
        hidden_dim=dimension,
        output_dim=dimension,
        num_layers=layers,
        apply_output_postprocessing_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        layer_config=outer_layer,
    )
    return MixtureOfExpertsModelConfig(
        input_dim=dimension,
        output_dim=dimension,
        top_k=mixture.num_experts,
        routing_initialization_mode=RoutingInitializationMode.LAYER,
        stack_config=stack,
    )


def feed_forward_config(dimension=3):
    return FeedForwardConfig(
        input_dim=dimension,
        output_dim=dimension,
        stack_config=model_config(dimension=dimension, layers=1),
    )


def explicit_reference(model, inputs):
    """Small loop oracle: no production routing helpers or expert forwards."""
    batches, tokens, _ = inputs.shape
    experts = model.num_experts
    slots = model.soft_router.slots_per_expert
    scoring_layers = tuple(model.soft_router.slot_logit_model.model.layers)
    logits = inputs.new_empty(batches, tokens, experts, slots)
    for batch in range(batches):
        for token in range(tokens):
            value = inputs[batch, token]
            unit_value = value / (value.square().sum() + 1e-6).sqrt()
            for expert in range(experts):
                hidden = unit_value
                for index, layer in enumerate(scoring_layers):
                    weights = layer.model.weight_params[expert]
                    hidden = torch.stack(
                        [
                            sum(
                                hidden[feature] * weights[feature, output]
                                for feature in range(weights.shape[0])
                            )
                            for output in range(weights.shape[1])
                        ]
                    )
                    if layer.model.bias_params is not None:
                        hidden = hidden + layer.model.bias_params[expert]
                    if index < len(scoring_layers) - 1:
                        hidden = F.gelu(hidden)
                logits[batch, token, expert] = model.soft_router.scale * hidden
    scores = logits.exp()
    dispatch = scores / scores.sum(dim=1, keepdim=True)
    combine = scores / scores.sum(dim=(2, 3), keepdim=True)
    input_slots = inputs.new_zeros(batches, experts, slots, model.input_dim)
    output_slots = inputs.new_empty(batches, experts, slots, model.output_dim)
    for batch in range(batches):
        for expert in range(experts):
            first, second = model.expert_modules[expert].layers
            for slot in range(slots):
                slot_value = inputs.new_zeros(model.input_dim)
                for token in range(tokens):
                    slot_value = (
                        slot_value
                        + dispatch[batch, token, expert, slot] * inputs[batch, token]
                    )
                input_slots[batch, expert, slot] = slot_value
                expanded = (
                    slot_value @ first.model.weight_params + first.model.bias_params
                )
                output_slots[batch, expert, slot] = (
                    F.gelu(expanded) @ second.model.weight_params
                    + second.model.bias_params
                )
    outputs = inputs.new_zeros(batches, tokens, model.output_dim)
    for batch in range(batches):
        for token in range(tokens):
            for expert in range(experts):
                for slot in range(slots):
                    outputs[batch, token] += (
                        combine[batch, token, expert, slot]
                        * output_slots[batch, expert, slot]
                    )
    return dispatch, combine, input_slots, outputs


@pytest.mark.parametrize("scoring_layers", [1, 3])
def test_soft_moe_matches_independent_full_mlp_oracle_and_input_slots(scoring_layers):
    torch.manual_seed(7)
    model = expert_config(output_dim=2, scoring_layers=scoring_layers).build().double()
    inputs = torch.tensor(
        [
            [[2.0, -1.0, 0.0], [0.0, 0.5, 4.0], [3.0, 2.0, -2.0]],
            [[-1.0, 0.0, 7.0], [0.1, 2.0, 0.0], [1.0, -3.0, 2.0]],
        ],
        dtype=torch.float64,
    )
    expected_dispatch, expected_combine, expected_slots, expected_output = (
        explicit_reference(model, inputs)
    )
    observed_slots = []
    handles = [
        expert.register_forward_pre_hook(
            lambda module, args: observed_slots.append(args[0].hidden.detach().clone())
        )
        for expert in model.expert_modules
    ]
    output, mask, loss = model(inputs)
    for handle in handles:
        handle.remove()
    dispatch, combine = model.soft_router(inputs)
    torch.testing.assert_close(dispatch, expected_dispatch)
    torch.testing.assert_close(combine, expected_combine)
    torch.testing.assert_close(dispatch.sum(1), torch.ones_like(dispatch.sum(1)))
    torch.testing.assert_close(
        combine.sum((2, 3)), torch.ones_like(combine.sum((2, 3)))
    )
    for expert, observed in enumerate(observed_slots):
        torch.testing.assert_close(observed.reshape(2, 2, 3), expected_slots[:, expert])
    torch.testing.assert_close(output, expected_output)
    assert mask is None and loss.item() == 0
    assert model.sampler is None


def test_uniform_scores_pool_original_values_before_nonlinearity():
    model = expert_config(experts=1, slots=1).build().double()
    with torch.no_grad():
        for parameter in model.soft_router.slot_logit_model.parameters():
            parameter.zero_()
    inputs = torch.tensor([[[1.0, 2.0, -4.0], [8.0, -3.0, 2.0]]], dtype=torch.float64)
    pooled = inputs.mean(1)
    expected = model.expert_modules[0](LayerState(hidden=pooled)).hidden
    output, _, _ = model(inputs)
    torch.testing.assert_close(output, expected[:, None].expand_as(output))
    assert not torch.allclose(output, inputs)


@pytest.mark.parametrize("scoring_layers", [1, 3])
def test_gradients_match_oracle_and_reach_every_parameter(scoring_layers):
    torch.manual_seed(11)
    model = expert_config(scoring_layers=scoring_layers).build().double()
    inputs = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    coefficients = torch.randn_like(inputs)
    output = model(inputs)[0]
    expected = explicit_reference(model, inputs)[-1]
    parameters = (inputs, *tuple(model.parameters()))
    actual_gradients = torch.autograd.grad(
        (output * coefficients).sum(), parameters, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        (expected * coefficients).sum(), parameters
    )
    for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference)
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    assert torch.autograd.gradcheck(
        lambda values: model(values)[0], (inputs,), fast_mode=True
    )
    assert len(
        {
            parameter.data_ptr()
            for expert in model.expert_modules
            for parameter in expert.parameters()
        }
    ) == sum(1 for expert in model.expert_modules for _ in expert.parameters())


def test_batch_independence_permutation_noncontiguity_and_variable_length():
    model = expert_config().build().double().eval()
    inputs = torch.randn(2, 3, 5, dtype=torch.float64).transpose(1, 2)
    assert not inputs.is_contiguous()
    output = model(inputs)[0]
    torch.testing.assert_close(
        output, torch.cat([model(sample[None])[0] for sample in inputs])
    )
    permutation = torch.tensor([4, 1, 3, 0, 2])
    torch.testing.assert_close(model(inputs[:, permutation])[0], output[:, permutation])
    shapes = {name: value.shape for name, value in model.state_dict().items()}
    for length in (1, 2, 7):
        assert model(torch.zeros(2, length, 3, dtype=torch.float64))[0].shape == (
            2,
            length,
            3,
        )
    assert shapes == {name: value.shape for name, value in model.state_dict().items()}


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
def test_precision_masks_empty_sequences_and_large_logits(dtype):
    model = expert_config(scoring_layers=3).build().to(dtype)
    inputs = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [float("nan"), float("nan"), float("nan")],
                [2.0, -1.0, 4.0],
            ],
            [[float("nan")] * 3] * 3,
        ],
        dtype=dtype,
        requires_grad=True,
    )
    active = torch.tensor([[[True], [False], [True]], [[False], [False], [False]]])
    output, returned_mask, loss = model(inputs, skip_mask=active)
    assert returned_mask is active and loss.item() == 0
    assert output.dtype == dtype and torch.isfinite(output).all()
    assert torch.count_nonzero(output[~active.expand_as(output)]) == 0
    unpadded = model(inputs[:1, [0, 2]])[0]
    torch.testing.assert_close(output[:1, [0, 2]], unpadded)
    output.float().square().sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.count_nonzero(inputs.grad[~active.expand_as(inputs)]) == 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    with torch.no_grad():
        model.soft_router.scale.fill_(10000)
    assert torch.isfinite(model(torch.zeros(1, 1, 3, dtype=dtype))[0]).all()


@pytest.mark.parametrize("scoring_bias", [False, True])
def test_router_follows_cpu_autocast_and_propagates_gradients(scoring_bias):
    cfg = expert_config(scoring_layers=3)
    slot_logit_stack_config = cfg.soft_moe_config.model_config.model_config
    slot_logit_stack_config.layer_config.layer_model_config.bias_flag = scoring_bias
    model = cfg.build()
    inputs = torch.randn(2, 4, 3, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        dispatch, combine = model.soft_router(inputs)
        outputs = model(inputs)[0]
    expected_dtype = torch.float32 if scoring_bias else torch.bfloat16
    assert dispatch.dtype == combine.dtype == expected_dtype
    assert outputs.dtype == torch.bfloat16
    outputs.float().sum().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_outer_model_state_checkpoint_and_feed_forward_layout():
    config = model_config()
    original = copy.deepcopy(config)
    model = config.build().eval()
    inputs = torch.randn(2, 5, 3)
    state = model(MixtureOfExpertsLayerState(hidden=inputs, loss=torch.tensor(2.0)))
    assert state.hidden.shape == inputs.shape and state.loss.item() == 2
    assert config == original
    archive = io.BytesIO()
    torch.save(model.state_dict(), archive)
    archive.seek(0)
    restored = config.build().eval()
    restored.load_state_dict(torch.load(archive, weights_only=True), strict=True)
    torch.testing.assert_close(restored(LayerState(hidden=inputs)).hidden, state.hidden)
    feed_forward = feed_forward_config().build().eval()
    padding = torch.tensor([[False, True, False, False, False], [True] * 5])
    actual, loss = feed_forward(inputs, padding_mask=padding)
    transposed, _ = feed_forward(
        inputs.transpose(0, 1), padding_mask=padding, batch_first=False
    )
    torch.testing.assert_close(actual, transposed.transpose(0, 1))
    torch.testing.assert_close(
        actual[:1, ~padding[0]], feed_forward(inputs[:1, ~padding[0]])[0]
    )
    assert torch.count_nonzero(actual[padding]) == 0 and loss.item() == 0
    additive_mask = torch.zeros_like(padding, dtype=torch.float32).masked_fill(
        padding, -torch.inf
    )
    torch.testing.assert_close(
        actual, feed_forward(inputs, padding_mask=additive_mask)[0]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"routing_initialization_mode": RoutingInitializationMode.SHARED},
        {"top_k": 1},
        {"capacity_factor": 1.0},
        {"weighted_parameters_flag": False},
        {"compute_expert_mixture_flag": False},
        {"weighting_position_option": ExpertWeightingPositionOptions.BEFORE_EXPERTS},
        {"soft_moe_config": SoftMoEConfig(slots_per_expert=0)},
        {"soft_moe_config": SoftMoEConfig(slots_per_expert=True)},
        {"soft_moe_config": SoftMoEConfig(slots_per_expert=1.5)},
        {"soft_moe_config": SoftMoEConfig(slots_per_expert=1, num_experts=9)},
    ],
)
def test_incompatible_configurations_fail_before_execution(changes):
    with pytest.raises((ValueError, TypeError), match="Soft MoE"):
        replace(expert_config(), **changes).build()


def test_flat_inputs_external_routes_and_invalid_masks_are_rejected():
    model = expert_config().build()
    inputs = torch.randn(2, 4, 3)
    for invalid in (
        inputs.flatten(0, 1),
        inputs[:, :, :2],
        inputs[:0],
        inputs[:, :0],
        inputs.long(),
    ):
        with pytest.raises((ValueError, TypeError), match="Soft MoE"):
            model(invalid)
    with pytest.raises(ValueError, match="external routing"):
        model(inputs, probabilities=torch.ones(2, 2))
    for invalid_mask in (torch.ones(2, 4, 1), torch.ones(2, 4, dtype=torch.bool)):
        with pytest.raises((ValueError, TypeError), match="Soft MoE"):
            model(inputs, skip_mask=invalid_mask)


@pytest.mark.parametrize("decoder", [False, True])
def test_transformer_integration_padding_layout_and_causality(decoder):
    from emperor.layers import AdditiveResidualConfig
    from emperor.transformer import (
        TransformerDecoderLayerConfig,
        TransformerEncoderLayerConfig,
    )
    from unit.test_transformer import self_attention

    attention = self_attention(4, causal=False)
    fields = dict(
        embedding_dim=4,
        layer_norm_position=LayerNormPositionOptions.BEFORE,
        dropout_probability=0.0,
        residual_config=AdditiveResidualConfig(),
        feed_forward_config=feed_forward_config(4),
    )
    if decoder:
        cfg = TransformerDecoderLayerConfig(
            self_attention_config=attention, cross_attention_config=None, **fields
        )
        mask_name = "key_padding_mask"
    else:
        cfg = TransformerEncoderLayerConfig(attention_config=attention, **fields)
        mask_name = "source_key_padding_mask"
    model = cfg.build().eval()
    tokens = torch.randn(2, 4, 4)
    padding = torch.tensor([[False, True, False, False], [False, False, True, False]])
    output, loss = model(tokens, **{mask_name: padding})
    changed = tokens.clone()
    changed[padding] = 1000
    torch.testing.assert_close(
        output[~padding], model(changed, **{mask_name: padding})[0][~padding]
    )
    for batch in range(2):
        expected = model(tokens[batch : batch + 1, ~padding[batch]])[0]
        torch.testing.assert_close(output[batch, ~padding[batch]], expected[0])
    assert loss.item() == 0

    attention.batch_first_flag = False
    transposed_model = cfg.build().eval()
    transposed_model.load_state_dict(model.state_dict(), strict=True)
    transposed_output, _ = transposed_model(
        tokens.transpose(0, 1), **{mask_name: padding}
    )
    torch.testing.assert_close(output, transposed_output.transpose(0, 1))
    with pytest.raises(ValueError, match="Soft MoE.*noncausal"):
        model(tokens, attention_mask=torch.zeros(4, 4))
    attention.causal_attention_mask_flag = True
    with pytest.raises(ValueError, match="Soft MoE.*noncausal"):
        cfg.build()
    model.self_attention_model.causal_attention_mask_flag = True
    with pytest.raises(ValueError, match="Soft MoE.*noncausal"):
        model(tokens)


def test_expert_auxiliary_loss_and_row_validation_use_existing_execution():
    from emperor.nn import Module

    class LossExpert(Module):
        def forward(self, state):
            state.hidden = state.hidden * 2
            state.loss = state.hidden.square().mean()
            return state

    model = expert_config().build()
    model.expert_modules[0] = LossExpert()
    inputs = torch.randn(2, 4, 3, requires_grad=True)
    _, _, loss = model(inputs)
    assert loss.item() > 0
    loss.backward()
    assert inputs.grad is not None and inputs.grad.abs().sum() > 0

    class WrongRows(Module):
        def forward(self, state):
            state.hidden = state.hidden[:1]
            return state

    model.expert_modules[0] = WrongRows()
    with pytest.raises(ValueError, match="one row"):
        model(inputs)


def test_sampler_map_reduce_and_context_controllers_are_rejected():
    from emperor.experts._config import (
        _MixtureOfExpertsMapConfig,
        _MixtureOfExpertsReduceConfig,
    )
    from emperor.layers import GateConfig
    from emperor.sampler import SamplerConfig

    cfg = expert_config()
    with pytest.raises(ValueError, match="sampler_config=None"):
        replace(cfg, sampler_config=SamplerConfig()).build()
    for variant in (_MixtureOfExpertsMapConfig, _MixtureOfExpertsReduceConfig):
        with pytest.raises(ValueError, match="Soft MoE"):
            variant(cfg).build()
    cfg.expert_model_config.layer_config.gate_config = GateConfig()
    with pytest.raises(ValueError, match="context controllers"):
        cfg.build()


def test_outer_normalization_excludes_padding_before_arithmetic():
    from emperor.layers import NormalizationOptions

    cfg = model_config()
    cfg.stack_config.layer_config.layer_norm_position = LayerNormPositionOptions.BEFORE
    cfg.stack_config.layer_config.normalization = NormalizationOptions.LAYER_NORM
    cfg.stack_config.apply_output_postprocessing_flag = True
    model = cfg.build()
    tokens = torch.randn(2, 4, 3)
    active = torch.tensor([[[True], [False], [True], [False]], [[False]] * 4])
    tokens[~active.expand_as(tokens)] = float("nan")
    tokens.requires_grad_()
    output = model(MixtureOfExpertsLayerState(hidden=tokens, skip_mask=active)).hidden
    assert (
        torch.isfinite(output).all()
        and torch.count_nonzero(output[~active.expand_as(output)]) == 0
    )
    output.square().sum().backward()
    assert torch.isfinite(tokens.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())






def test_per_row_adaptive_experts_work_but_grouped_experts_are_rejected():
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterGroupingScopeOptions,
    )
    from support.adaptive_grouping import bias_linear, grouping_value

    cfg = expert_config(dimension=2, output_dim=2)
    cfg.expert_model_config.layer_config.layer_model_config = bias_linear().cfg
    model = cfg.build().eval()
    inputs = torch.randn(2, 3, 2)
    torch.testing.assert_close(
        model(inputs)[0], torch.cat([model(sample[None])[0] for sample in inputs])
    )
    cfg.expert_model_config.layer_config.layer_model_config.adaptive_augmentation_config.grouping_config = grouping_value(
        AdaptiveParameterGroupingScopeOptions.ROWS, 1
    )
    with pytest.raises(ValueError, match="grouping context"):
        cfg.build()


def test_router_configuration_overrides_serialization_and_validation():
    from emperor.experts._routing.soft import SoftMoERouter

    cfg = SoftMoEConfig(
        input_dim=2, num_experts=3, slots_per_expert=1, model_config=scoring_config(3)
    )
    restored = pickle.loads(pickle.dumps(cfg))
    model = restored.build(SoftMoEConfig(slots_per_expert=2))
    assert cfg == restored and cfg.slots_per_expert == 1
    assert model.slot_logit_model.model.layers[-1].model.weight_params.shape == (
        3,
        5,
        2,
    )
    assert model.scale.item() == 1
    with pytest.raises(TypeError, match="SoftMoEConfig"):
        SoftMoERouter(object())
    with pytest.raises(TypeError, match="overrides"):
        cfg.build(object())
    invalid = expert_config(dimension=1)
    invalid.soft_moe_config.input_dim = True
    with pytest.raises(ValueError, match="positive integer"):
        invalid.build()


def test_feed_forward_and_outer_model_reject_ambiguous_boundaries():
    from emperor.layers import RecurrentLayerConfig

    cfg = feed_forward_config()
    feed_forward = cfg.build()
    with pytest.raises(ValueError, match="rank-three"):
        feed_forward(torch.zeros(4, 3))
    for padding in (
        torch.zeros(2, 4, dtype=torch.int64),
        torch.zeros(4, 2, dtype=torch.bool),
        torch.full((2, 4), float("nan")),
    ):
        with pytest.raises((TypeError, ValueError), match="Soft MoE"):
            feed_forward(torch.zeros(2, 4, 3), padding_mask=padding)
    with pytest.raises(TypeError, match="batch_first"):
        feed_forward(torch.zeros(2, 4, 3), batch_first=None)
    with pytest.raises(ValueError, match="external routing"):
        model_config().build()(
            MixtureOfExpertsLayerState(
                hidden=torch.zeros(2, 4, 3), indices=torch.zeros(2, 1, dtype=torch.long)
            )
        )
    cfg.stack_config = RecurrentLayerConfig(block_config=cfg.stack_config)
    with pytest.raises(ValueError, match="direct MixtureOfExpertsModelConfig"):
        cfg.build()


@pytest.mark.parametrize("experts,slots", [(1, 1), (11, 2), (32, 1)])
def test_depth_mapped_scoring_has_independent_expert_parameters_at_every_layer(
    experts, slots
):
    from emperor.augmentations.adaptive_parameters import DepthMappingLayerStack

    torch.manual_seed(23)
    model = expert_config(experts=experts, slots=slots, scoring_layers=3).build()
    slot_logit_model = model.soft_router.slot_logit_model
    assert isinstance(slot_logit_model, DepthMappingLayerStack)
    assert len(slot_logit_model.model.layers) == 3
    selected_expert = experts - 1
    weights = tuple(
        layer.model.weight_params for layer in slot_logit_model.model.layers
    )
    assert [tuple(weight.shape) for weight in weights] == [
        (experts, 3, 5),
        (experts, 5, 5),
        (experts, 5, slots),
    ]
    scores = slot_logit_model(torch.randn(4, 3))
    assert scores.shape == (4, experts, slots)
    gradients = torch.autograd.grad(scores[:, selected_expert].sum(), weights)
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient[selected_expert].abs().sum() > 0
        assert torch.count_nonzero(gradient[:selected_expert]) == 0
    for batch_size, tokens in ((1, 1), (2, 3), (1, 7)):
        inputs = torch.randn(batch_size, tokens, 3)
        output, _, _ = model(inputs)
        dispatch, combine = model.soft_router(inputs)
        assert output.shape == inputs.shape
        assert dispatch.shape == combine.shape == (batch_size, tokens, experts, slots)
        torch.testing.assert_close(dispatch.sum(1), torch.ones_like(dispatch.sum(1)))
        torch.testing.assert_close(
            combine.sum((2, 3)), torch.ones_like(combine.sum((2, 3)))
        )


@pytest.mark.parametrize(
    "parameter_dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_scoring_precision_matches_promoted_stack_without_changing_stored_parameters(
    parameter_dtype, input_dtype, autocast_enabled
):
    router = expert_config(scoring_layers=3).build().soft_router.to(parameter_dtype)
    original_parameters = dict(router.named_parameters())
    original_values = {
        name: value.detach().clone() for name, value in original_parameters.items()
    }
    reference = copy.deepcopy(router).to(input_dtype)
    inputs = torch.randn(2, 3, 3, dtype=input_dtype, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        expected = reference(inputs)
        actual = router(inputs)
    for weights, expected_weights in zip(actual, expected, strict=True):
        assert weights.dtype == input_dtype
        torch.testing.assert_close(weights, expected_weights)
    objective = sum((weights * torch.randn_like(weights)).sum() for weights in actual)
    objective.backward()
    assert torch.isfinite(inputs.grad).all()
    for name, parameter in router.named_parameters():
        assert parameter is original_parameters[name]
        assert parameter.dtype == parameter_dtype
        torch.testing.assert_close(parameter, original_values[name], rtol=0, atol=0)
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_multilayer_scorer_configuration_and_checkpoint_round_trip():
    cfg = expert_config(experts=32, slots=1, scoring_layers=3)
    cfg.soft_moe_config.model_config.input_dim = 3
    cfg.soft_moe_config.model_config.output_dim = 1
    cfg.soft_moe_config.model_config.generator_depth = 32
    cfg.soft_moe_config.model_config.model_config.input_dim = 3
    cfg.soft_moe_config.model_config.model_config.output_dim = 1
    original = copy.deepcopy(cfg)
    source = cfg.build().double().eval()
    inputs = torch.randn(2, 4, 3, dtype=torch.float64)
    restored_config = pickle.loads(pickle.dumps(cfg))
    restored = restored_config.build().double().eval()
    archive = io.BytesIO()
    torch.save(source.state_dict(), archive)
    archive.seek(0)
    restored.load_state_dict(torch.load(archive, weights_only=True), strict=True)
    torch.testing.assert_close(restored(inputs)[0], source(inputs)[0])
    assert cfg == restored_config == original
    assert any(
        key.startswith("soft_router.slot_logit_model.") for key in source.state_dict()
    )
    assert "soft_router.phi" not in source.state_dict()
    old_state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith("soft_router.slot_logit_model.")
    }
    old_state["soft_router.phi"] = torch.zeros(3, 32, 1)
    with pytest.raises(RuntimeError, match="soft_router.phi"):
        restored.load_state_dict(old_state, strict=True)


@pytest.mark.parametrize(
    "target,changes",
    [
        ("mapping", {"input_dim": 4}),
        ("mapping", {"input_dim": True}),
        ("mapping", {"input_dim": 3.0}),
        ("mapping", {"output_dim": 3}),
        ("mapping", {"output_dim": True}),
        ("mapping", {"output_dim": 2.0}),
        ("mapping", {"generator_depth": 3}),
        ("mapping", {"generator_depth": True}),
        ("mapping", {"generator_depth": 2.0}),
        ("stack", {"input_dim": 4}),
        ("stack", {"output_dim": 3}),
        ("stack", {"apply_output_postprocessing_flag": True}),
        ("stack", {"apply_output_postprocessing_flag": None}),
        ("stack", {"layer_config": None}),
    ],
)
def test_conflicting_scorer_configuration_is_rejected_without_mutation(target, changes):
    cfg = expert_config(scoring_layers=3)
    if target == "mapping":
        cfg.soft_moe_config.model_config = replace(
            cfg.soft_moe_config.model_config, **changes
        )
    else:
        cfg.soft_moe_config.model_config.model_config = replace(
            cfg.soft_moe_config.model_config.model_config, **changes
        )
    original = copy.deepcopy(cfg)
    with pytest.raises((ValueError, TypeError), match="Soft MoE"):
        cfg.build()
    assert cfg == original


def test_scorer_requires_explicit_ordinary_controller_free_stack():
    from emperor.augmentations.adaptive_parameters import AdaptiveLinearLayerConfig
    from emperor.layers import GateConfig, RecurrentLayerConfig

    for invalid in (None, RecurrentLayerConfig(), LayerStackConfig()):
        cfg = expert_config()
        cfg.soft_moe_config.model_config = invalid
        with pytest.raises(TypeError, match="Soft MoE requires model_config"):
            cfg.build()
    cfg = expert_config()
    cfg.soft_moe_config.model_config.model_config.layer_config.gate_config = (
        GateConfig()
    )
    with pytest.raises(ValueError, match="context controllers"):
        cfg.build()
    cfg = expert_config()
    cfg.soft_moe_config.model_config.model_config.layer_config.layer_model_config = (
        AdaptiveLinearLayerConfig()
    )
    with pytest.raises(TypeError, match="ordinary LayerConfig and LinearLayerConfig"):
        cfg.build()
