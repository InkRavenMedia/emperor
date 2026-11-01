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
    LayerStack,
    LayerStackConfig,
    LayerState,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, RouterModel
from emperor.transformer import FeedForwardConfig


def scoring_config(layers=1):
    return LayerStackConfig(
        hidden_dim=5,
        num_layers=layers,
        apply_output_postprocessing_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        layer_config=layer_config(),
    )


def adaptive_scoring_config():
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveLinearLayerConfig,
        AdaptiveParameterAugmentationConfig,
        AdditiveDynamicBiasConfig,
        WeightDecayScheduleOptions,
    )

    config = scoring_config()
    config.layer_config.layer_model_config = AdaptiveLinearLayerConfig(
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            bias_config=AdditiveDynamicBiasConfig(
                model_config=scoring_config(),
                decay_schedule=WeightDecayScheduleOptions.DISABLED,
                decay_rate=0.0,
                decay_warmup_batches=0,
            ),
        ),
    )
    return config


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
            slots_per_expert=slots,
            router_config=RouterConfig(
                noisy_topk_flag=False, model_config=scoring_config(scoring_layers)
            ),
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
    scoring_layers = tuple(model.soft_router.router.model.layers)
    logits = inputs.new_empty(batches, tokens, experts, slots)
    for batch in range(batches):
        for token in range(tokens):
            value = inputs[batch, token]
            unit_value = value / (value.square().sum() + 1e-6).sqrt()
            hidden = unit_value
            for index, layer in enumerate(scoring_layers):
                weights = layer.model.weight_params
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
                    hidden = hidden + layer.model.bias_params
                if index < len(scoring_layers) - 1:
                    hidden = F.gelu(hidden)
            logits[batch, token] = model.soft_router.scale * hidden.reshape(
                experts, slots
            )
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
        for parameter in model.soft_router.router.model.parameters():
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
def test_router_preserves_float32_scores_under_cpu_autocast_and_propagates_gradients(
    scoring_bias,
):
    cfg = expert_config(scoring_layers=3)
    slot_logit_stack_config = cfg.soft_moe_config.router_config.model_config
    slot_logit_stack_config.layer_config.layer_model_config.bias_flag = scoring_bias
    cfg.expert_model_config.layer_config.layer_model_config.bias_flag = scoring_bias
    model = cfg.build()
    inputs = torch.randn(2, 4, 3, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        dispatch, combine = model.soft_router(inputs)
        outputs = model(inputs)[0]
    assert dispatch.dtype == combine.dtype == torch.float32
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
    with pytest.raises(ValueError, match="Soft MoE.*explicit"):
        model(tokens, attention_mask=torch.zeros(4, 4))
    attention.causal_attention_mask_flag = True
    causal_model = cfg.build().eval()
    causal_input = tokens.transpose(0, 1).clone().requires_grad_()
    causal_output, _ = causal_model(causal_input, **{mask_name: padding})
    changed = causal_input.detach().clone()
    changed[3] += 100
    torch.testing.assert_close(
        causal_output[:3], causal_model(changed, **{mask_name: padding})[0][:3]
    )
    causal_output[:3].square().sum().backward()
    assert torch.count_nonzero(causal_input.grad[3]) == 0


def test_causal_soft_mixture_matches_independent_prefix_evaluation():
    torch.manual_seed(53)
    model = expert_config().build().double().eval()
    tokens = torch.randn(2, 5, 3, dtype=torch.double, requires_grad=True)
    padding = torch.tensor([[True, False, True, False, False], [True] * 5])
    expected = torch.cat(
        [
            model(tokens[:, :end], skip_mask=~padding[:, :end, None])[0][:, -1:]
            for end in range(1, tokens.size(1) + 1)
        ],
        dim=1,
    )
    actual, _, loss = model(
        tokens, skip_mask=~padding[..., None], causal_prefix_flag=True
    )
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[padding]) == 0
    assert torch.isfinite(actual).all() and torch.isfinite(loss)
    actual[:, :4].square().sum().backward()
    assert torch.count_nonzero(tokens.grad[:, 4]) == 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    # Causality is carried by this call; it must not change subsequent bidirectional calls.
    torch.testing.assert_close(
        model(tokens, skip_mask=~padding[..., None])[0][:, -1:], expected[:, -1:]
    )


@pytest.mark.parametrize("batch_first", [True, False])
def test_causal_soft_feed_forward_preserves_prefixes_through_all_layers(batch_first):
    model = feed_forward_config().build().eval()
    tokens = torch.randn(2, 5, 3, requires_grad=True)
    inputs = tokens if batch_first else tokens.transpose(0, 1)
    output, _ = model(inputs, batch_first=batch_first, causal_prefix_flag=True)
    prefix = inputs[:, :3] if batch_first else inputs[:3]
    prefix_output, _ = model(prefix, batch_first=batch_first, causal_prefix_flag=True)
    expected = output[:, :3] if batch_first else output[:3]
    torch.testing.assert_close(prefix_output, expected)
    expected.square().sum().backward()
    assert torch.count_nonzero(tokens.grad[:, 3:]) == 0


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


@pytest.mark.parametrize("padding_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("normalization_name", ["RMS_NORM", "LAYER_NORM"])
def test_standalone_layer_masks_padding_before_normalization_and_backward(
    padding_value, normalization_name
):
    from emperor.layers import NormalizationOptions

    torch.manual_seed(17)
    mixture_config = expert_config()
    normalization_option = NormalizationOptions[normalization_name]
    standalone_config = MixtureOfExpertsLayerConfig(
        input_dim=3,
        output_dim=3,
        activation=ActivationOptions.DISABLED,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.BEFORE,
        normalization=normalization_option,
        layer_model_config=mixture_config,
    )
    layer = standalone_config.build().double().eval()
    reference_layer = copy.deepcopy(layer)
    active_tokens = torch.tensor([[[True], [False], [True], [False]], [[False]] * 4])
    original_tokens = torch.randn(2, 4, 3, dtype=torch.float64)
    padded_tokens = original_tokens.masked_fill(~active_tokens, padding_value)
    padded_tokens.requires_grad_()
    clean_tokens = original_tokens.masked_fill(~active_tokens, 0)
    clean_tokens.requires_grad_()
    padded_state = MixtureOfExpertsLayerState(
        hidden=padded_tokens, skip_mask=active_tokens
    )
    clean_state = MixtureOfExpertsLayerState(
        hidden=clean_tokens, skip_mask=active_tokens
    )

    actual_state = layer(padded_state)
    reference_state = reference_layer(clean_state)
    actual_outputs = actual_state.hidden
    reference_outputs = reference_state.hidden
    torch.testing.assert_close(actual_outputs, reference_outputs)
    assert actual_state.skip_mask is active_tokens
    finite_outputs = torch.isfinite(actual_outputs)
    assert finite_outputs.all()

    loss_coefficients = torch.randn_like(actual_outputs)
    weighted_actual_outputs = actual_outputs * loss_coefficients
    actual_objective = weighted_actual_outputs.sum()
    actual_objective.backward()
    weighted_reference_outputs = reference_outputs * loss_coefficients
    reference_objective = weighted_reference_outputs.sum()
    reference_objective.backward()

    finite_input_gradients = torch.isfinite(padded_tokens.grad)
    assert finite_input_gradients.all()
    torch.testing.assert_close(padded_tokens.grad, clean_tokens.grad)
    masked_input_gradients = padded_tokens.grad.masked_select(~active_tokens)
    assert masked_input_gradients.count_nonzero() == 0
    for actual_parameter, reference_parameter in zip(
        layer.parameters(), reference_layer.parameters(), strict=True
    ):
        assert actual_parameter.grad is not None
        finite_parameter_gradients = torch.isfinite(actual_parameter.grad)
        assert finite_parameter_gradients.all()
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad)
    preserved_padding = padded_tokens.detach().masked_select(~active_tokens)
    expected_padding = torch.full_like(preserved_padding, padding_value)
    torch.testing.assert_close(preserved_padding, expected_padding, equal_nan=True)


@pytest.mark.parametrize("invalid_mask_kind", ["floating", "wrong_shape"])
def test_standalone_layer_validates_mask_before_clearing_padding(invalid_mask_kind):
    outer_config = model_config(layers=1)
    standalone_config = replace(
        outer_config.stack_config.layer_config,
        input_dim=3,
        output_dim=3,
        layer_norm_position=LayerNormPositionOptions.BEFORE,
    )
    layer = standalone_config.build()
    token_features = torch.ones(1, 3, 3)
    if invalid_mask_kind == "floating":
        invalid_mask = torch.ones(1, 3, 1)
        expected_error = TypeError
        expected_message = "Boolean"
    else:
        invalid_mask = torch.ones(1, 3, dtype=torch.bool)
        expected_error = ValueError
        expected_message = "shape"
    input_state = MixtureOfExpertsLayerState(
        hidden=token_features, skip_mask=invalid_mask
    )
    with pytest.raises(expected_error, match=expected_message):
        layer(input_state)


def test_per_row_adaptive_experts_work_but_cross_sample_grouping_is_rejected():
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
    with pytest.raises(ValueError, match="grouping requires"):
        cfg.build()


def test_router_configuration_overrides_serialization_and_validation():
    from emperor.experts._routing.soft import SoftMoERouter

    cfg = SoftMoEConfig(
        input_dim=2,
        num_experts=3,
        slots_per_expert=1,
        router_config=RouterConfig(
            noisy_topk_flag=False, model_config=scoring_config(3)
        ),
    )
    restored = pickle.loads(pickle.dumps(cfg))
    model = restored.build(SoftMoEConfig(slots_per_expert=2))
    assert cfg == restored and cfg.slots_per_expert == 1
    assert isinstance(model.router, RouterModel)
    assert model.router.num_experts == 3
    assert model.router.router_output_dim == 6
    assert cfg.router_config.input_dim is None
    assert cfg.router_config.output_dim is None
    assert model.router.model.layers[-1].model.weight_params.shape == (5, 6)
    assert model.scale.item() == 1
    with pytest.raises(TypeError, match="SoftMoEConfig"):
        SoftMoERouter(object())
    with pytest.raises(TypeError, match="overrides"):
        cfg.build(object())
    invalid = expert_config(dimension=1)
    invalid.soft_moe_config.input_dim = True
    with pytest.raises(ValueError, match="positive integer"):
        invalid.build()


@pytest.mark.parametrize(
    "updates",
    [
        {"input_dim": 4},
        {"input_dim": True},
        {"num_experts": 4},
        {"num_experts": 2.0},
        {"output_dim": 2},
        {"output_dim": 0},
        {"output_dim": 4.0},
        {"noisy_topk_flag": True},
        {"noisy_topk_flag": None},
    ],
)
def test_soft_router_rejects_incompatible_router_config_before_initialization(updates):
    cfg = expert_config()
    cfg.soft_moe_config.router_config = replace(
        cfg.soft_moe_config.router_config, **updates
    )
    rng_before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="Soft MoE router|noisy_topk_flag"):
        cfg.build()
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize("router_config", [None, LayerStackConfig()])
def test_soft_router_requires_an_explicit_router_config(router_config):
    cfg = expert_config()
    cfg.soft_moe_config.router_config = router_config
    with pytest.raises(TypeError, match="router_config=RouterConfig"):
        cfg.build()


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
def test_one_shared_scoring_stack_produces_logits_for_all_expert_slots(experts, slots):
    torch.manual_seed(23)
    model = expert_config(experts=experts, slots=slots, scoring_layers=3).build()
    slot_logit_model = model.soft_router.router.model
    assert isinstance(slot_logit_model, LayerStack)
    assert len(slot_logit_model.layers) == 3
    selected_expert = experts - 1
    weights = tuple(layer.model.weight_params for layer in slot_logit_model.layers)
    assert [tuple(weight.shape) for weight in weights] == [
        (3, 5),
        (5, 5),
        (5, experts * slots),
    ]
    scores = slot_logit_model(LayerState(hidden=torch.randn(4, 3))).hidden
    assert scores.shape == (4, experts * slots)
    selected_slot_start = selected_expert * slots
    selected_scores = scores[:, selected_slot_start : selected_slot_start + slots]
    gradients = torch.autograd.grad(selected_scores.sum(), weights)
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert torch.count_nonzero(gradients[-1][:, :selected_slot_start]) == 0
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
    cfg.soft_moe_config.router_config.model_config.input_dim = 3
    cfg.soft_moe_config.router_config.model_config.output_dim = 32
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
        key.startswith("soft_router.router.model.") for key in source.state_dict()
    )
    assert "soft_router.phi" not in source.state_dict()
    old_state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith("soft_router.router.model.")
    }
    old_state["soft_router.phi"] = torch.zeros(3, 32, 1)
    with pytest.raises(RuntimeError, match="soft_router.phi"):
        restored.load_state_dict(old_state, strict=True)

    depth_mapped_state = {}
    scoring_prefix = "soft_router.router.model."
    for key, value in source.state_dict().items():
        if key.startswith(scoring_prefix):
            layer_key = key.removeprefix(scoring_prefix)
            depth_mapped_state[f"{scoring_prefix}model.{layer_key}"] = value
        else:
            depth_mapped_state[key] = value
    with pytest.raises(RuntimeError, match="soft_router.router.model.model"):
        restored.load_state_dict(depth_mapped_state, strict=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"input_dim": 4},
        {"input_dim": True},
        {"input_dim": 3.0},
        {"output_dim": 3},
        {"output_dim": 2},
        {"output_dim": True},
        {"output_dim": 4.0},
        {"apply_output_postprocessing_flag": True},
        {"apply_output_postprocessing_flag": None},
        {"layer_config": None},
    ],
)
def test_conflicting_scorer_configuration_is_rejected_without_mutation(changes):
    cfg = expert_config(scoring_layers=3)
    cfg.soft_moe_config.router_config.model_config = replace(
        cfg.soft_moe_config.router_config.model_config, **changes
    )
    original = copy.deepcopy(cfg)
    with pytest.raises((ValueError, TypeError), match="Soft MoE"):
        cfg.build()
    assert cfg == original


def test_scorer_requires_explicit_ordinary_controller_free_stack():
    from emperor.layers import GateConfig, RecurrentLayerConfig

    for invalid in (
        None,
        RecurrentLayerConfig(),
        DepthMappingHandlerConfig(model_config=scoring_config()),
    ):
        cfg = expert_config()
        cfg.soft_moe_config.router_config.model_config = invalid
        with pytest.raises(
            TypeError, match="Soft MoE requires router_config.model_config"
        ):
            cfg.build()
    cfg = expert_config()
    cfg.soft_moe_config.router_config.model_config.layer_config.gate_config = (
        GateConfig()
    )
    with pytest.raises(ValueError, match="context controllers"):
        cfg.build()
    cfg = expert_config()
    cfg.soft_moe_config.router_config.model_config.layer_config.layer_model_config = (
        LayerConfig()
    )
    with pytest.raises(TypeError, match="ordinary LayerConfig and LinearLayerConfig"):
        cfg.soft_moe_config.build(SoftMoEConfig(input_dim=3, num_experts=2))
    cfg.soft_moe_config.router_config.model_config = LayerStackConfig()
    with pytest.raises(TypeError, match="ordinary LayerConfig and LinearLayerConfig"):
        cfg.build()


def test_adaptive_router_scores_and_gradients_match_generated_bias_oracle():
    torch.manual_seed(57)
    cfg = SoftMoEConfig(
        input_dim=3,
        num_experts=2,
        slots_per_expert=2,
        router_config=RouterConfig(
            noisy_topk_flag=False, model_config=adaptive_scoring_config()
        ),
    )
    router = cfg.build().double().eval()
    scoring_layer = router.router.model.layers[0].model
    bias_generator = scoring_layer.adaptive_behaviour.bias_model.model[0].model
    inputs = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    active = torch.tensor(
        [[[True], [True], [False], [True]], [[False], [False], [False], [False]]]
    )
    masked_inputs = inputs.masked_fill(~active, 0)
    normalized_inputs = (
        masked_inputs / (masked_inputs.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    )
    generated_bias = (
        normalized_inputs @ bias_generator.weight_params + bias_generator.bias_params
    )
    logits = (
        normalized_inputs @ scoring_layer.weight_params
        + scoring_layer.bias_params
        + generated_bias
    ) * router.scale
    logits = logits.reshape(2, 4, 2, 2)
    dispatch_logits = logits.masked_fill(~active.unsqueeze(-1), -torch.inf)
    dispatch_logits = torch.where(
        active.unsqueeze(-1).any(1, keepdim=True), dispatch_logits, 0
    )
    expected_dispatch = dispatch_logits.softmax(1).masked_fill(~active.unsqueeze(-1), 0)
    expected_combine = (
        logits.flatten(2)
        .softmax(-1)
        .reshape_as(logits)
        .masked_fill(~active.unsqueeze(-1), 0)
    )
    actual_dispatch, actual_combine = router(inputs, active)
    torch.testing.assert_close(actual_dispatch, expected_dispatch)
    torch.testing.assert_close(actual_combine, expected_combine)
    coefficients = torch.randn_like(logits)
    parameters = (inputs, *router.parameters())
    actual_gradients = torch.autograd.grad(
        ((actual_dispatch + actual_combine) * coefficients).sum(), parameters
    )
    expected_gradients = torch.autograd.grad(
        ((expected_dispatch + expected_combine) * coefficients).sum(), parameters
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "parameter_dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_adaptive_router_preserves_decay_counters_and_checkpoint_under_autocast(
    parameter_dtype,
):
    from emperor.augmentations.adaptive_parameters import WeightDecayScheduleOptions

    cfg = SoftMoEConfig(
        input_dim=3,
        num_experts=2,
        slots_per_expert=2,
        router_config=RouterConfig(
            noisy_topk_flag=False, model_config=adaptive_scoring_config()
        ),
    )
    bias_config = cfg.router_config.model_config.layer_config.layer_model_config.adaptive_augmentation_config.bias_config
    bias_config.decay_schedule = WeightDecayScheduleOptions.EXPONENTIAL
    bias_config.decay_rate = 0.01
    bias_config.decay_warmup_batches = 1
    router = cfg.build().to(parameter_dtype).train()
    inputs = torch.randn(2, 4, 3, requires_grad=True)
    for _ in range(3):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            dispatch, combine = router(inputs)
        assert dispatch.dtype == combine.dtype == torch.float32
        ((dispatch + combine) * torch.randn_like(dispatch)).sum().backward()
    buffers = dict(router.named_buffers())
    warmup = [value for name, value in buffers.items() if name.endswith("warmup_step")]
    decay = [value for name, value in buffers.items() if name.endswith("decay_step")]
    assert warmup and all(value.item() == 1 for value in warmup)
    assert decay and all(value.item() == 2 for value in decay)
    assert all(
        parameter.dtype == parameter_dtype
        and parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in router.parameters()
    )
    restored = cfg.build().to(parameter_dtype)
    restored.load_state_dict(router.state_dict(), strict=True)
    router.eval()
    restored.eval()
    for actual, expected in zip(restored(inputs), router(inputs), strict=True):
        torch.testing.assert_close(actual, expected)
    assert all(value.item() == 2 for value in decay)


@pytest.mark.parametrize("scope", ["ROWS", "SEQUENCE"])
def test_adaptive_router_rejects_row_and_fixed_count_grouping(scope):
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterGroupingScopeOptions,
    )
    from support.adaptive_grouping import grouping_value

    cfg = SoftMoEConfig(
        input_dim=3,
        num_experts=2,
        slots_per_expert=2,
        router_config=RouterConfig(
            noisy_topk_flag=False, model_config=adaptive_scoring_config()
        ),
    )
    augmentation = cfg.router_config.model_config.layer_config.layer_model_config.adaptive_augmentation_config
    augmentation.grouping_config = grouping_value(
        AdaptiveParameterGroupingScopeOptions[scope], 2
    )
    with pytest.raises(ValueError, match="grouping requires"):
        cfg.build()


@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_adaptive_router_auxiliary_loss_reaches_owner_once_per_forward(
    attention, causal
):
    from support.adaptive_generation import bias_mixture_config, matrix_sampler_config
    from tests.unit.test_attention_soft_moe import attention_config

    if attention:
        cfg = attention_config(
            scoring_layers=1,
            causal_attention_mask_flag=causal,
            query_key_projection_dim=8,
            value_projection_dim=8,
        )
        routing_config = cfg.experts_config.soft_moe_config
        expert_stack = cfg.experts_config.expert_model_config
        dimension = cfg.embedding_dim
    else:
        cfg = expert_config()
        routing_config = cfg.soft_moe_config
        expert_stack = cfg.expert_model_config
        dimension = cfg.input_dim
    scoring_stack = adaptive_scoring_config()
    augmentation = (
        scoring_stack.layer_config.layer_model_config.adaptive_augmentation_config
    )
    augmentation.bias_config = bias_mixture_config(
        input_dim=dimension,
        output_dim=None,
        sampler_config=matrix_sampler_config(
            input_dim=dimension, switch_loss_weight=0.2, zero_centred_loss_weight=0.1
        ),
    )
    routing_config.router_config.model_config = scoring_stack
    expert_stack.num_layers = 1
    expert_stack.layer_config.layer_model_config = copy.deepcopy(
        scoring_stack.layer_config.layer_model_config
    )
    model = cfg.build().train()
    router = model.projector.soft_router if attention else model.soft_router
    if attention:
        experts = [*model.projector.query_model, *model.projector.output_model]
    else:
        experts = list(model.expert_modules)
    for _ in range(2):
        features = torch.randn(2, 4, dimension)
        if attention:
            _, _, owner_loss = model(features, features, features)
        else:
            _, _, owner_loss = model(features, causal_prefix_flag=causal)
        sampler = router.router.model.layers[
            0
        ].model.adaptive_behaviour.bias_model.sampler
        expected_loss = sampler.get_auxiliary_loss().mean()
        assert expected_loss > 0
        torch.testing.assert_close(router.get_auxiliary_loss(), expected_loss)
        expert_loss = sum(
            layer.model.adaptive_behaviour.bias_model.sampler.get_auxiliary_loss().mean()
            for expert in experts
            for layer in expert.layers
        )
        torch.testing.assert_close(owner_loss, expected_loss + expert_loss)
        model.zero_grad(set_to_none=True)
        owner_loss.backward()
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in sampler.router.parameters()
        )


@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_soft_moe_router_always_uses_its_expert_linear_model_type(attention, adaptive):
    from tests.unit.test_attention_soft_moe import attention_config

    if attention:
        cfg = attention_config(scoring_layers=2)
        mixture_config = cfg.experts_config
    else:
        cfg = expert_config(scoring_layers=2)
        mixture_config = cfg
    adaptive_linear = adaptive_scoring_config().layer_config.layer_model_config
    if adaptive:
        mixture_config.expert_model_config.layer_config.layer_model_config = (
            adaptive_linear
        )
    else:
        mixture_config.soft_moe_config.router_config.model_config.layer_config.layer_model_config = adaptive_linear
    original_config = copy.deepcopy(cfg)
    model = cfg.build().eval()
    if attention:
        router = model.projector.soft_router
        expert = model.projector.query_model[0]
        dimension = cfg.embedding_dim
    else:
        router = model.soft_router
        expert = model.expert_modules[0]
        dimension = cfg.input_dim
    assert (
        router.cfg.router_config.model_config.layer_config.layer_model_config
        == expert.cfg.layer_config.layer_model_config
    )
    assert all(
        type(layer.model) is type(expert.layers[0].model)
        for layer in router.router.model.layers
    )
    assert {id(parameter) for parameter in router.parameters()}.isdisjoint(
        id(parameter) for parameter in expert.parameters()
    )
    features = torch.randn(2, 4, dimension)
    outputs = model(features, features, features) if attention else model(features)
    outputs[0].square().sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in router.parameters()
    )
    assert cfg == original_config


def test_adaptive_router_padding_does_not_contribute_to_auxiliary_loss():
    from support.adaptive_generation import bias_mixture_config, matrix_sampler_config

    scoring_stack = adaptive_scoring_config()
    augmentation = (
        scoring_stack.layer_config.layer_model_config.adaptive_augmentation_config
    )
    augmentation.bias_config = bias_mixture_config(
        input_dim=3,
        output_dim=None,
        sampler_config=matrix_sampler_config(input_dim=3, zero_centred_loss_weight=0.1),
    )
    router = (
        SoftMoEConfig(
            input_dim=3,
            num_experts=2,
            slots_per_expert=2,
            router_config=RouterConfig(
                noisy_topk_flag=False, model_config=scoring_stack
            ),
        )
        .build()
        .eval()
    )
    inputs = torch.randn(1, 2, 3)
    expected_weights = router(inputs)
    expected_loss = router.get_auxiliary_loss()
    padded_inputs = torch.cat((inputs, torch.full_like(inputs, float("nan"))), dim=1)
    active_mask = torch.tensor([[[True], [True], [False], [False]]])
    actual_weights = router(padded_inputs, active_mask)
    for actual, expected in zip(actual_weights, expected_weights, strict=True):
        torch.testing.assert_close(actual[:, :2], expected)
        assert torch.count_nonzero(actual[:, 2:]) == 0
    torch.testing.assert_close(router.get_auxiliary_loss(), expected_loss)
    empty_weights = router(padded_inputs, torch.zeros_like(active_mask))
    assert all(torch.count_nonzero(weights) == 0 for weights in empty_weights)
    assert router.get_auxiliary_loss().item() == 0
