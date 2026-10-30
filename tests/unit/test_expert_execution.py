"""The expert container dispatches batches using supplied routing assignments."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from emperor.experts import (
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    RoutingInitializationMode,
)
from emperor.experts._layers.experts import Experts
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig


def expert_config():
    linear_config = LinearLayerConfig(bias_flag=True)
    layer_config = LayerConfig(
        activation=ActivationOptions.DISABLED,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        layer_model_config=linear_config,
    )
    stack_config = LayerStackConfig(
        hidden_dim=4,
        num_layers=1,
        apply_output_postprocessing_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        layer_config=layer_config,
    )
    return MixtureOfExpertsConfig(
        input_dim=3,
        output_dim=3,
        num_experts=3,
        top_k=3,
        capacity_factor=0.0,
        compute_expert_mixture_flag=True,
        weighted_parameters_flag=True,
        weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
        routing_initialization_mode=RoutingInitializationMode.DISABLED,
        expert_model_config=stack_config,
    )


class ScalingExpert(nn.Module):
    def __init__(self, scale: float, *, return_loss: bool):
        super().__init__()
        scale_value = torch.tensor(scale)
        self.scale = nn.Parameter(scale_value)
        feature_offsets = torch.ones(3)
        self.register_buffer("feature_offsets", feature_offsets)
        self.return_loss = return_loss

    def forward(self, state):
        state.hidden = state.hidden * self.scale + self.feature_offsets
        if self.return_loss:
            squared_outputs = state.hidden.square()
            state.loss = squared_outputs.mean()
        return state


def test_experts_construct_independent_networks_with_legacy_state_keys():
    cfg = expert_config()
    original_config = deepcopy(cfg)
    experts = Experts(cfg)

    assert cfg == original_config
    assert tuple(experts._modules) == ("0", "1", "2")
    expected_keys = tuple(
        f"{index}.layers.0.model.{parameter_name}"
        for index in range(3)
        for parameter_name in ("weight_params", "bias_params")
    )
    expert_state = experts.state_dict()
    assert tuple(expert_state) == expected_keys
    parameter_ids = [
        id(parameter) for expert in experts for parameter in expert.parameters()
    ]
    unique_parameter_ids = set(parameter_ids)
    assert len(parameter_ids) == len(unique_parameter_ids)
    for expert in experts:
        assert expert.cfg.input_dim == expert.cfg.output_dim == 3

    legacy_container = nn.ModuleList()
    for _ in range(3):
        overrides = LayerStackConfig(input_dim=3, output_dim=3)
        legacy_expert = cfg.expert_model_config.build(overrides)
        legacy_container.append(legacy_expert)
    legacy_state = legacy_container.state_dict()
    experts.load_state_dict(legacy_state, strict=True)
    restored_state = experts.state_dict()
    legacy_container.load_state_dict(restored_state, strict=True)
    for key, expected_parameter in legacy_state.items():
        torch.testing.assert_close(restored_state[key], expected_parameter)


def test_experts_preserve_indexing_slicing_replacement_and_registered_state():
    cfg = expert_config()
    mixture = cfg.build()
    experts = mixture.expert_modules
    assert isinstance(experts, Experts)
    assert tuple(mixture._modules) == ("expert_modules",)
    assert experts[-1] is experts[2]

    rng_state = torch.get_rng_state()
    for selection in (slice(0, 2), slice(None, None, -1), slice(0, 0)):
        selected_experts = experts[selection]
        original_experts = list(experts)
        expected_experts = original_experts[selection]
        assert type(selected_experts) is nn.ModuleList
        assert list(selected_experts) == expected_experts
    rng_state_after_slicing = torch.get_rng_state()
    torch.testing.assert_close(rng_state_after_slicing, rng_state, rtol=0, atol=0)

    replacement = ScalingExpert(2.0, return_loss=False)
    experts[1] = replacement
    assert experts[1] is replacement
    named_parameters = mixture.named_parameters()
    registered_parameters = dict(named_parameters)
    assert registered_parameters["expert_modules.1.scale"] is replacement.scale
    named_buffers = mixture.named_buffers()
    registered_buffers = dict(named_buffers)
    assert (
        registered_buffers["expert_modules.1.feature_offsets"]
        is replacement.feature_offsets
    )
    mixture.to(dtype=torch.float64)
    mixture.eval()
    assert not experts.training and all(not expert.training for expert in experts)
    assert replacement.scale.dtype == replacement.feature_offsets.dtype == torch.float64
    mixture.train()
    assert all(expert.training for expert in experts)


@pytest.mark.parametrize(
    "slot_shape", [(2, 3, 3), (0, 3, 2, 3), (2, 2, 2, 3), (2, 3, 0, 3), (2, 3, 2, 4)]
)
def test_preallocated_slots_require_configured_experts_and_features(slot_shape):
    configuration = expert_config()
    experts = Experts(configuration)
    invalid_slots = torch.randn(slot_shape)
    with pytest.raises(ValueError, match="Expert slots require"):
        experts.forward_slots(invalid_slots)


def test_batch_forward_dispatches_samples_and_preserves_selected_gradients():
    cfg = expert_config()
    cfg.top_k = 1
    experts = Experts(cfg)
    for expert_index in range(3):
        expert_scale = float(expert_index + 1)
        replacement = ScalingExpert(expert_scale, return_loss=False)
        experts[expert_index] = replacement
    experts.double()
    samples = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    probabilities = torch.ones(5, dtype=torch.float64)
    indices = torch.ones(5, dtype=torch.long)
    outputs, routing_positions, returned_probabilities, loss = experts(
        samples, probabilities, indices
    )
    expected_outputs = samples * 2 + 1
    expected_positions = torch.arange(5)
    torch.testing.assert_close(outputs, expected_outputs)
    torch.testing.assert_close(routing_positions, expected_positions)
    assert returned_probabilities is probabilities
    assert loss.shape == () and loss.dtype == samples.dtype and loss.item() == 0
    output_sum = outputs.sum()
    output_sum.backward()
    expected_input_gradients = torch.full_like(samples, 2)
    torch.testing.assert_close(samples.grad, expected_input_gradients)
    assert experts[1].scale.grad is not None
    assert experts[0].scale.grad is None and experts[2].scale.grad is None


def test_batch_forward_groups_multiple_assignments_and_accumulates_losses():
    cfg = expert_config()
    cfg.top_k = 2
    experts = Experts(cfg)
    for expert_index in range(3):
        expert_scale = float(expert_index + 1)
        replacement = ScalingExpert(expert_scale, return_loss=True)
        experts[expert_index] = replacement
    samples = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    probabilities = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
    indices = torch.tensor([[0, 2], [0, 2]])

    outputs, routing_positions, returned_probabilities, loss = experts(
        samples, probabilities, indices
    )

    expected_outputs = torch.tensor(
        [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [4.0, 7.0, 10.0], [13.0, 16.0, 19.0]]
    )
    expected_positions = torch.tensor([0, 2, 1, 3])
    squared_first_outputs = expected_outputs[:2].square()
    squared_last_outputs = expected_outputs[2:].square()
    first_loss = squared_first_outputs.mean()
    last_loss = squared_last_outputs.mean()
    expected_loss = first_loss + last_loss
    torch.testing.assert_close(outputs, expected_outputs)
    torch.testing.assert_close(routing_positions, expected_positions)
    torch.testing.assert_close(loss, expected_loss)
    assert returned_probabilities is probabilities

    output_sum = outputs.sum()
    output_sum.backward()
    expected_input_gradients = torch.full_like(samples, 4)
    torch.testing.assert_close(samples.grad, expected_input_gradients)
    assert experts[0].scale.grad is not None and experts[2].scale.grad is not None
    assert experts[1].scale.grad is None


def test_batch_forward_runs_every_expert_for_dense_inputs():
    cfg = expert_config()
    experts = Experts(cfg)
    for expert_index in range(3):
        expert_scale = float(expert_index + 1)
        replacement = ScalingExpert(expert_scale, return_loss=False)
        experts[expert_index] = replacement
    samples = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    probabilities = torch.full((2, 3), 1 / 3)
    outputs, routing_positions, returned_probabilities, loss = experts(
        samples, probabilities
    )
    expected_outputs = torch.tensor(
        [
            [2.0, 3.0, 4.0],
            [3.0, 5.0, 7.0],
            [4.0, 7.0, 10.0],
            [5.0, 6.0, 7.0],
            [9.0, 11.0, 13.0],
            [13.0, 16.0, 19.0],
        ]
    )
    torch.testing.assert_close(outputs, expected_outputs)
    assert routing_positions is None
    assert returned_probabilities is probabilities
    assert loss.item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_soft_tokens_are_pooled_and_executed_inside_experts(dtype):
    cfg = expert_config()
    experts = Experts(cfg)
    for expert_index in range(3):
        expert_scale = float(expert_index + 1)
        return_loss = expert_index != 1
        replacement = ScalingExpert(expert_scale, return_loss=return_loss)
        experts[expert_index] = replacement
    experts.to(dtype=dtype)
    features = torch.randn(2, 3, 4, dtype=dtype)
    token_features = features.transpose(1, 2)
    token_features.requires_grad_()
    assert not token_features.is_contiguous()
    dispatch_scores = torch.arange(1, 49, dtype=dtype)
    dispatch_scores = dispatch_scores.reshape(2, 4, 3, 2)
    token_score_totals = dispatch_scores.sum(dim=1, keepdim=True)
    dispatch_weights = dispatch_scores / token_score_totals
    dispatch_weights.requires_grad_()

    output_slots, total_loss = experts.forward_soft_tokens(
        token_features, dispatch_weights
    )
    expected_expert_outputs = []
    for expert_index in range(3):
        expected_slot_outputs = []
        for slot_index in range(2):
            token_weights = dispatch_weights[:, :, expert_index, slot_index]
            feature_weights = token_weights.unsqueeze(-1)
            weighted_tokens = token_features * feature_weights
            pooled_slot = weighted_tokens.sum(dim=1)
            expected_output = pooled_slot * (expert_index + 1) + 1
            expected_slot_outputs.append(expected_output)
        expert_outputs = torch.stack(expected_slot_outputs, dim=1)
        expected_expert_outputs.append(expert_outputs)
    expected_slots = torch.stack(expected_expert_outputs, dim=1)
    first_squared_outputs = expected_expert_outputs[0].square()
    last_squared_outputs = expected_expert_outputs[2].square()
    first_loss = first_squared_outputs.mean()
    last_loss = last_squared_outputs.mean()
    expected_loss = first_loss + last_loss
    torch.testing.assert_close(output_slots, expected_slots)
    torch.testing.assert_close(total_loss, expected_loss)

    output_sum = output_slots.sum()
    objective = output_sum + total_loss
    objective.backward()
    assert token_features.grad is not None
    assert dispatch_weights.grad is not None
    finite_input_gradients = torch.isfinite(token_features.grad)
    assert finite_input_gradients.all()
    finite_dispatch_gradients = torch.isfinite(dispatch_weights.grad)
    assert finite_dispatch_gradients.all()
    expert_parameters = experts.parameters()
    assert all(parameter.grad is not None for parameter in expert_parameters)
