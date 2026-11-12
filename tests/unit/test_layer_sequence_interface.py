"""Sequence entry points retain each model's established execution semantics."""

from copy import deepcopy

import pytest
import torch

from emperor.experts import RoutingInitializationMode
from emperor.layers import HierarchicalReasoningModelRecurrentConfig, LayerState
from tests.unit import test_recurrent_layer as recurrent_fixtures
from tests.unit.test_attention_model_mixer import (
    _linear_stack,
    _mixture_mixing_config,
    _recurrent_stack,
)
from tests.unit.test_expert_soft_moe import (
    adaptive_scoring_config,
)
from tests.unit.test_expert_soft_moe import (
    model_config as soft_model_config,
)
from tests.unit.test_recurrent_composition_inner_thinking import (
    config as inner_thinking_config,
)
from tests.unit.test_recurrent_composition_tiny_recursive_model import (
    _config as tiny_recursive_config,
)


def sequence_model_config(kind):
    if kind == "dense":
        config = _linear_stack(3, 5, num_layers=2)
        config.layer_config.dropout_probability = 0.2
        return config
    if kind == "adaptive":
        config = adaptive_scoring_config()
        config.input_dim = 3
        config.output_dim = 5
        return config
    if kind in {"hard", "hard_shared"}:
        config = _mixture_mixing_config(configured_dim=3, auxiliary_loss_weight=0.1)
        if kind == "hard_shared":
            config.routing_initialization_mode = RoutingInitializationMode.SHARED
            expert_config = config.stack_config.layer_config.layer_model_config
            config.sampler_config = deepcopy(expert_config.sampler_config)
            expert_config.routing_initialization_mode = RoutingInitializationMode.SHARED
        return config
    if kind == "soft":
        return soft_model_config(dimension=3)
    if kind == "inner_thinking":
        return inner_thinking_config()
    if kind == "tiny_recursive":
        return tiny_recursive_config(model_dim=3, auxiliary_loss=0.1)
    if kind == "hierarchical":
        return HierarchicalReasoningModelRecurrentConfig(
            input_dim=3,
            output_dim=3,
            high_block_config=_linear_stack(3, 3),
            low_block_config=_linear_stack(3, 3),
            high_cycles=2,
            low_cycles=2,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        )
    config = _recurrent_stack(3)
    if kind == "recurrent_hard":
        config.block_config = _mixture_mixing_config(
            configured_dim=3, auxiliary_loss_weight=0.1
        )
    if kind == "recurrent_adaptive":
        config.block_config = (
            recurrent_fixtures.TestRecurrentLayer().adaptive_parameter_block_config(3)
        )
    if kind == "recurrent_halting":
        config.halting_config = recurrent_fixtures.TestRecurrentLayer().halting_config(
            3, gate_threshold=0.5
        )
    return config


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "dense",
        "adaptive",
        "hard",
        "hard_shared",
        "soft",
        "recurrent",
        "recurrent_hard",
        "recurrent_adaptive",
        "recurrent_halting",
        "inner_thinking",
        "tiny_recursive",
        "hierarchical",
    ],
)
def test_sequence_interface_preserves_outputs_gradients_state_and_rng(kind, training):
    torch.manual_seed(271)
    config = sequence_model_config(kind)
    model = config.build().double().train(training)
    reference = deepcopy(model)
    for _ in range(2):
        values = torch.randn(2, 3, config.input_dim * 2, dtype=torch.float64)
        actual_input = values.clone()[..., ::2].requires_grad_()
        reference_input = values.clone()[..., ::2].requires_grad_()
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        rng_before = torch.get_rng_state()
        actual = model.forward_sequence(actual_input)
        rng_after = torch.get_rng_state()
        torch.set_rng_state(rng_before)
        reference_vectors = reference_input
        if kind != "soft":
            reference_vectors = reference_input.reshape(-1, config.input_dim)
        expected = reference(LayerState(hidden=reference_vectors))
        assert torch.equal(torch.get_rng_state(), rng_after)
        expected_output = expected.hidden.reshape(2, 3, config.output_dim)
        torch.testing.assert_close(actual.hidden, expected_output, rtol=0, atol=0)
        if expected.loss is None:
            assert actual.loss is None
        else:
            torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
        actual_loss = actual.hidden.square().sum()
        expected_loss = expected_output.square().sum()
        if actual.loss is not None:
            actual_loss = actual_loss + actual.loss
            expected_loss = expected_loss + expected.loss
        actual_loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(
            actual_input.grad, reference_input.grad, rtol=0, atol=0
        )
        actual_parameters = dict(model.named_parameters())
        reference_parameters = dict(reference.named_parameters())
        assert actual_parameters.keys() == reference_parameters.keys()
        for name, parameter in actual_parameters.items():
            expected_gradient = reference_parameters[name].grad
            if expected_gradient is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(
                    parameter.grad, expected_gradient, rtol=0, atol=0
                )
        actual_state = model.state_dict()
        expected_state = reference.state_dict()
        assert actual_state.keys() == expected_state.keys()
        for name, value in actual_state.items():
            torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(6, 3), (0, 2, 3), (2, 0, 3), (2, 3, 0)])
def test_sequence_interface_rejects_ambiguous_or_empty_layouts(shape):
    model = _linear_stack(3, 3).build()
    with pytest.raises(ValueError, match="Sequence input"):
        model.forward_sequence(torch.zeros(shape))


def test_soft_sequence_interface_keeps_sample_routing_independent():
    model = soft_model_config(dimension=3).build().eval()
    inputs = torch.randn(2, 4, 3)
    original = model.forward_sequence(inputs).hidden
    changed = inputs.clone()
    changed[0] += 100
    changed_output = model.forward_sequence(changed).hidden
    torch.testing.assert_close(changed_output[1], original[1], rtol=0, atol=0)
    assert not torch.equal(changed_output[0], original[0])
