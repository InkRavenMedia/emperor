from dataclasses import replace

import pytest
import torch

from emperor.attention import AttentionLayerState
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    AttentionResidualConfig,
    LayerNormPositionOptions,
    LayerState,
    RecurrentCompressionConfig,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import MemoryPositionOptions
from tests.unit.test_recurrent_composition_tiny_recursive_model import (
    _config,
    _RecordingHaltingConfig,
    _RecordingMemory,
    _RecordingTransform,
)


def answer_cycle_config(**options):
    return replace(
        _config(model_dim=2, **options),
        residual_config=AttentionResidualConfig(block_size=1, rms_norm_epsilon=1e-6),
    )


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("generated_query", [False, True])
def test_pipeline_applies_standard_attention_to_each_full_length_answer_candidate(
    monkeypatch, compressed, generated_query
):
    torch.manual_seed(37)
    cfg = answer_cycle_config(latent_updates_per_answer_update=4, auxiliary_loss=0.25)
    if generated_query:
        cfg.residual_config.model_config = LinearLayerConfig(bias_flag=False)
    if compressed:
        cfg.compression_config = RecurrentCompressionConfig(max_slots=2)
    model = cfg.build()
    inputs = torch.randn(2, 5, 2)
    with torch.no_grad():
        model.answer_initial.copy_(torch.tensor([-0.5, 1.0]))
        second_connection = model.pipeline.residual_schedule.subsequent_connections[0]
        if generated_query:
            second_connection.query_model.weight_params.copy_(torch.eye(2))
        else:
            second_connection.query.copy_(torch.tensor([0.8, -0.4]))

    residual_calls, transitions = [], []
    call_block = model.step._call_transition_block

    def observe_residual(connection, args, kwargs, mixed):
        candidate, previous = args
        sources = kwargs["residual_state"].sources
        residual_calls.append(
            (connection, len(transitions), candidate, previous, sources, mixed)
        )

    def observe_transition(block, state):
        before = state.hidden
        result = call_block(block, state)
        transitions.append((before, result.hidden))
        return result

    for connection in (model.pipeline.residual.connection, second_connection):
        connection.register_forward_hook(observe_residual, with_kwargs=True)
    monkeypatch.setattr(model.step, "_call_transition_block", observe_transition)
    pipeline_calls = []
    model.pipeline.postprocessing.gate = torch.nn.Identity()
    model.pipeline.postprocessing.gate.register_forward_hook(
        lambda *_: pipeline_calls.append(len(transitions))
    )
    result = model(LayerState(inputs, inputs.new_tensor(1.0)))
    assert pipeline_calls == [5, 10]
    assert len(transitions) == 10
    assert [x.shape[-2] for x in model.block_model.inputs] == [
        2 if compressed else 5
    ] * 10
    assert all(
        before.shape == after.shape == inputs.shape for before, after in transitions
    )
    assert [call[1] for call in residual_calls] == [5, 10]
    answer = model.answer_initial.expand_as(inputs)
    expected_sources = [answer]
    latent = model.latent_initial.expand_as(inputs)
    for cycle in range(2):
        for before, after in transitions[cycle * 5 : cycle * 5 + 4]:
            torch.testing.assert_close(before, inputs + answer + latent)
            latent = after
        before, after = transitions[cycle * 5 + 4]
        connection, _, candidate, previous, sources, mixed = residual_calls[cycle]
        torch.testing.assert_close(before, answer + latent)
        torch.testing.assert_close(previous, answer)
        torch.testing.assert_close(candidate, after)
        expected_sources.append(after)
        stacked = torch.stack(expected_sources)
        torch.testing.assert_close(torch.stack(sources), stacked)
        query = (
            connection.query_model.forward_hidden(candidate)
            if generated_query
            else connection.query
        )
        weights = (connection.key_norm(stacked) * query).sum(-1).softmax(0)
        answer = (weights.unsqueeze(-1) * stacked).sum(0)
        torch.testing.assert_close(mixed, answer)
    torch.testing.assert_close(result.hidden, answer)
    assert result.loss.item() == 3.5
    assert result.residual_state is None
    result.hidden.sum().backward()
    parameter = (
        second_connection.query_model.weight_params
        if generated_query
        else second_connection.query
    )
    assert parameter.grad is not None and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("residual", "first_answer", "second_answer"),
    [
        (None, 3.0, 11.0),
        (AdditiveResidualConfig(), 3.0, 14.0),
        (
            WeightedResidualConfig(coefficient_activation=ActivationOptions.SIGMOID),
            1.5,
            5.5,
        ),
        (
            WeightedBlendResidualConfig(
                coefficient_activation=ActivationOptions.SIGMOID
            ),
            2.7,
            9.63,
        ),
    ],
)
def test_answer_pipeline_without_attention(residual, first_answer, second_answer):
    cfg = answer_cycle_config(
        latent_updates_per_answer_update=1, initialization_standard_deviation=0.0
    )
    cfg.residual_config = residual
    model = cfg.build()
    value = torch.ones(1, 2)
    with torch.no_grad():
        result = model(LayerState(value))
    expected = [1, 2, 3 + first_answer, 4 + 2 * first_answer]
    for received, scalar in zip(model.block_model.inputs, expected, strict=True):
        torch.testing.assert_close(received, value * scalar)
    torch.testing.assert_close(result.hidden, value * second_answer)


def test_normalization_memory_and_gate_wrap_only_the_answer_call():
    cfg = answer_cycle_config(latent_updates_per_answer_update=1, answer_update_count=1)
    model = cfg.build()
    model.answer_initial.fill_(2)
    model.latent_initial.fill_(3)
    normalization = _RecordingTransform(10.0)
    memory = _RecordingMemory(20.0, MemoryPositionOptions.BEFORE_AFFINE)
    gate = _RecordingTransform(30.0)
    model.pipeline.normalization.position = LayerNormPositionOptions.BEFORE
    model.pipeline.normalization.module = normalization
    model.pipeline.memory.model = memory
    model.pipeline.postprocessing.gate = gate
    value = torch.ones(1, 2)
    result = model(LayerState(value))
    assert len(normalization.inputs) == len(memory.inputs) == len(gate.inputs) == 1
    torch.testing.assert_close(model.block_model.inputs[0], value * 6)
    torch.testing.assert_close(normalization.inputs[0], value * 9)
    torch.testing.assert_close(memory.inputs[0], value * 19)
    torch.testing.assert_close(model.block_model.inputs[1], value * 39)
    torch.testing.assert_close(gate.inputs[0], value * 40)
    torch.testing.assert_close(result.hidden, value * 36)


def test_depth_growth_moves_gradient_window_and_restores_checkpoint():
    torch.manual_seed(8)
    cfg = answer_cycle_config(
        latent_updates_per_answer_update=4, answer_update_count=3, initial_iterations=2
    )
    model = cfg.build()
    value = torch.randn(2, 2)

    def run(current_model):
        result = current_model(LayerState(value))
        result.hidden.sum().backward()
        return result.hidden

    run(model)
    queries = [
        conn.query for conn in model.pipeline.residual_schedule.subsequent_connections
    ]
    assert queries[0].grad is not None and queries[0].grad.abs().sum() > 0
    assert queries[1].grad is None
    assert len(model.block_model.inputs) == 10
    restored = cfg.build()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.zero_grad(set_to_none=True)
    torch.testing.assert_close(run(model), run(restored))
    assert queries[0].grad is None and queries[1].grad.abs().sum() > 0
    assert model.block_model.grad_modes[10:] == [False] * 10 + [True] * 5


@pytest.mark.parametrize("block_size", [1, 2])
@pytest.mark.parametrize(
    "window", [{"gradient_transition_count": 4}, {"no_gradient_transition_count": 0}]
)
def test_attention_history_is_branch_local_during_smooth_growth(block_size, window):
    torch.manual_seed(19)
    cfg = answer_cycle_config(
        answer_update_count=3,
        initial_iterations=2,
        **window,
        forward_calls_before_iteration_increment=4,
        smooth_iteration_growth_flag=True,
    )
    cfg.residual_config.block_size = block_size
    model = cfg.build()
    stable = replace(
        cfg,
        smooth_iteration_growth_flag=False,
        forward_calls_before_iteration_increment=1000,
    )
    source = stable.build()
    target = replace(stable, initial_iterations=3).build()
    source.load_state_dict(model.state_dict())
    target.load_state_dict(model.state_dict())
    value = torch.randn(3, 2)
    for _ in range(4):
        model(LayerState(value))
    weight = model.recurrent_iteration_schedule.snapshot().transition_weight
    assert 0 < weight < 1
    actual = model(LayerState(value)).hidden
    left = source(LayerState(value)).hidden
    right = target(LayerState(value)).hidden
    torch.testing.assert_close(actual, torch.lerp(left, right, weight))
    actual.sum().backward()
    (left.sum() * (1 - weight) + right.sum() * weight).backward()
    for (name, param), (_, src), (_, dst) in zip(
        model.named_parameters(),
        source.named_parameters(),
        target.named_parameters(),
        strict=True,
    ):
        if src.grad is None and dst.grad is None:
            assert param.grad is None, name
            continue
        expected_grad = sum(grad for grad in (src.grad, dst.grad) if grad is not None)
        torch.testing.assert_close(param.grad, expected_grad, msg=name)


def test_halting_receives_the_mixed_answer_once_per_cycle(monkeypatch):
    cfg = answer_cycle_config(answer_update_count=4, no_gradient_transition_count=0)
    cfg.halting_config = _RecordingHaltingConfig(halt_after_updates=2)
    model = cfg.build()
    mixed_answers = []
    halting_model = model.pipeline.halting.model
    update = halting_model.update_halting_state

    def halt_first_item(previous, candidate):
        state, hidden = update(previous, candidate)
        state.halt_mask[0] = True
        return state, torch.where(state.halt_mask[..., None], -7.0, hidden)

    monkeypatch.setattr(halting_model, "update_halting_state", halt_first_item)

    for connection in (
        model.pipeline.residual.connection,
        *model.pipeline.residual_schedule.subsequent_connections,
    ):
        connection.register_forward_hook(
            lambda module, args, output: mixed_answers.append(output)
        )
    result = model(LayerState(torch.randn(3, 2)))
    assert len(mixed_answers) == len(halting_model.update_inputs) == 2
    assert len(model.block_model.inputs) == 6
    torch.testing.assert_close(
        torch.stack(mixed_answers), torch.stack(halting_model.update_inputs)
    )
    torch.testing.assert_close(result.hidden, torch.full_like(result.hidden, -7.0))


def test_compressed_answer_cycles_exclude_padded_tokens_and_preserve_gradients():
    torch.manual_seed(11)
    cfg = answer_cycle_config(no_gradient_transition_count=0)
    cfg.compression_config = RecurrentCompressionConfig(max_slots=2)
    model = cfg.build()
    valid = torch.randn(1, 4, 2)
    padding = torch.tensor([[False, True, False, False, False], [True] * 5])
    hidden = torch.full((2, 5, 2), torch.nan)
    hidden[~padding] = valid[0]
    hidden.requires_grad_()
    reference = model(LayerState(valid)).hidden
    result = model(AttentionLayerState(hidden, key_padding_mask=padding))
    torch.testing.assert_close(result.hidden[~padding], reference[0])
    assert result.key_padding_mask is padding
    assert (result.hidden[padding] == 0).all()
    result.hidden.sum().backward()
    assert torch.isfinite(hidden.grad).all() and (hidden.grad[padding] == 0).all()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
