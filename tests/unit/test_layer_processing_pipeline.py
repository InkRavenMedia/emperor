from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from emperor.halting import HaltingHiddenStateModeOptions, SoftHaltingConfig
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    AttentionResidualConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
    NormalizationOptions,
)
from emperor.layers._pipeline import LayerProcessingPipeline
from emperor.layers._pipeline.delegates import (
    LayerHaltingDelegate,
    LayerMemoryDelegate,
    LayerNormalizationDelegate,
    LayerPostprocessingDelegate,
    LayerResidualDelegate,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import MemoryPositionOptions, WeightedDynamicMemoryConfig
from support.layers import recurrent_config
from support.recurrent import inner_thinking_config


@pytest.fixture
def layer_config():
    return LayerConfig(
        input_dim=2,
        output_dim=2,
        activation=ActivationOptions.DISABLED,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
    )


def test_pipeline_registers_all_five_delegates_with_the_same_config(layer_config):
    ordinary = LayerProcessingPipeline(layer_config)
    recurrent = LayerProcessingPipeline(
        recurrent_config(recurrent_layer_norm_position=LayerNormPositionOptions.AFTER)
    )
    assert ordinary.cfg is layer_config
    assert recurrent.cfg == replace(
        layer_config,
        layer_norm_position=LayerNormPositionOptions.AFTER,
        normalization=NormalizationOptions.LAYER_NORM,
    )
    for pipeline in (ordinary, recurrent):
        delegates = list(pipeline.named_children())
        assert [(name, type(delegate)) for name, delegate in delegates] == [
            ("postprocessing", LayerPostprocessingDelegate),
            ("residual", LayerResidualDelegate),
            ("halting", LayerHaltingDelegate),
            ("memory", LayerMemoryDelegate),
            ("normalization", LayerNormalizationDelegate),
        ]
        assert all(delegate.cfg is pipeline.cfg for _, delegate in delegates)


@pytest.mark.parametrize("config_factory", [recurrent_config, inner_thinking_config])
def test_recurrent_pipeline_uses_both_steps_weights_and_advances_detached_history(
    config_factory,
):
    model = replace(
        config_factory(),
        max_steps=2,
        initial_iterations=2,
        residual_config=AttentionResidualConfig(block_size=1, rms_norm_epsilon=1e-6),
    ).build()
    pipeline = model.pipeline
    state = LayerState(torch.tensor([[1.0, -1.0]], requires_grad=True))
    recurrent_state = model.step.initialize_recurrent_state(state)

    def increment(state):
        return replace(state, hidden=state.hidden + 1)

    state = pipeline.process_recurrent_transition(
        state, increment, recurrent_state=recurrent_state
    )
    recurrent_state = model.step.update_state(
        recurrent_state, state, tracks_gradients=True
    )
    target_state = model.step.fork(recurrent_state)
    result = pipeline.process_recurrent_transition(
        state, increment, recurrent_state=recurrent_state, target_state=target_state
    )
    result.hidden.sum().backward()

    assert result.residual_state is None
    history = recurrent_state.residual_state
    target_history = target_state.residual_state
    assert len(history.sources) == len(target_history.sources) == 3
    torch.testing.assert_close(history.sources[-1], target_history.sources[-1])
    assert history.sources[-1].requires_grad
    assert not target_history.sources[-1].requires_grad
    for name in (
        "residual.connection.query",
        "residual_schedule.subsequent_connections.0.query",
    ):
        assert pipeline.get_parameter(name).grad.abs().sum() > 0


@pytest.mark.parametrize("use_bound_method", [False, True], ids=["model", "method"])
def test_process_normal_layer_applies_configured_delegates_to_model_output(
    layer_config, use_bound_method
):
    cfg = replace(
        layer_config,
        activation=ActivationOptions.RELU,
        dropout_probability=1.0,
        residual_config=AdditiveResidualConfig(),
        layer_norm_position=LayerNormPositionOptions.BEFORE,
        normalization=NormalizationOptions.LAYER_NORM,
    )
    pipeline = LayerProcessingPipeline(cfg).eval()
    input_hidden = torch.tensor([[-2.0, 3.0]])

    class DoubleHidden(nn.Module):
        def forward(self, state):
            return replace(
                state, hidden=state.hidden * 2, loss=state.hidden.square().sum()
            )

    model = DoubleHidden()
    run_model = model.forward if use_bound_method else model
    input_state = LayerState(hidden=input_hidden)
    result = pipeline.process_normal_layer(input_state, run_model)

    normalized = nn.functional.layer_norm(input_hidden, (2,))
    assert result is not input_state
    torch.testing.assert_close(result.hidden, input_hidden + (normalized * 2).relu())
    torch.testing.assert_close(result.loss, normalized.square().sum())

    pipeline.train()
    dropped = pipeline.process_normal_layer(LayerState(hidden=input_hidden), run_model)
    torch.testing.assert_close(dropped.hidden, input_hidden)


@pytest.mark.parametrize("position", list(MemoryPositionOptions))
def test_process_normal_layer_runs_configured_memory_at_its_position(
    layer_config, position
):
    memory_generator = LayerStackConfig(
        hidden_dim=2,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=False,
        layer_config=replace(
            layer_config, layer_model_config=LinearLayerConfig(bias_flag=True)
        ),
    )
    memory_config = WeightedDynamicMemoryConfig(
        memory_position_option=position, model_config=memory_generator
    )
    pipeline = LayerProcessingPipeline(
        replace(layer_config, memory_config=memory_config)
    ).eval()
    hidden = torch.tensor([[1.0, 2.0]])

    def increment(state):
        return replace(state, hidden=state.hidden + 1)

    result = pipeline.process_normal_layer(LayerState(hidden=hidden), increment)
    memory = pipeline.memory.model
    if position == MemoryPositionOptions.BEFORE_AFFINE:
        expected = memory(hidden) + 1
    else:
        expected = memory(hidden + 1)
    torch.testing.assert_close(result.hidden, expected)


def test_process_normal_layer_updates_halting_and_skips_a_completed_state(layer_config):
    halting_config = SoftHaltingConfig(
        threshold=0.25,
        ponder_cost_weight=1.0,
        min_steps=1,
        dropout_probability=0.0,
        hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
    )
    pipeline = LayerProcessingPipeline(
        replace(layer_config, halting_config=halting_config)
    ).eval()
    for parameter in pipeline.halting.parameters():
        nn.init.zeros_(parameter)
    run_model = Mock(side_effect=nn.Identity())
    state = LayerState(hidden=torch.ones(1, 2))

    state = pipeline.process_normal_layer(state, run_model)
    state = pipeline.process_normal_layer(state, run_model)
    assert state.halting_state.halt_mask.all()
    assert state.loss is not None
    assert pipeline.process_normal_layer(state, run_model) is state
    assert run_model.call_count == 2


def test_layer_checks_halting_once_before_entering_the_pipeline(layer_config):
    events = []

    class HookedLayer(Layer):
        def _handle_layer_input(self, state):
            events.append("input")
            return state

        def _handle_model_processing(self, state):
            events.append("model")
            return replace(state, hidden=state.hidden * 2)

        def _handle_layer_output(self, state):
            events.append("output")
            return state

    layer = HookedLayer(
        replace(layer_config, layer_model_config=LinearLayerConfig(bias_flag=False))
    )
    for name, delegate in layer.pipeline.named_children():
        assert layer.get_submodule(f"pipeline.{name}") is delegate
        assert not hasattr(layer, name)
    state = LayerState(hidden=torch.ones(1, 2))
    with (
        patch.object(
            layer.pipeline.halting, "should_skip", side_effect=[False, True]
        ) as should_skip,
        patch.object(
            layer.pipeline,
            "process_normal_layer",
            wraps=layer.pipeline.process_normal_layer,
        ) as process_normal_layer,
    ):
        result = layer(state)
        assert layer(result) is result

    assert should_skip.call_count == 2
    process_normal_layer.assert_called_once_with(
        state, run_model=layer._handle_model_processing, halting_already_checked=True
    )
    assert events == ["input", "model", "output"]
    torch.testing.assert_close(result.hidden, state.hidden * 2)


def test_layer_pipeline_registers_and_loads_only_current_state_paths(layer_config):
    cfg = replace(
        layer_config,
        layer_norm_position=LayerNormPositionOptions.AFTER,
        residual_config=AttentionResidualConfig(block_size=1, rms_norm_epsilon=1e-6),
        layer_model_config=LinearLayerConfig(input_dim=2, output_dim=2, bias_flag=True),
    )
    source = nn.Sequential(Layer(cfg))
    destination = nn.Sequential(Layer(cfg))
    checkpoint = source.state_dict()

    parameters = dict(source.named_parameters())
    buffer_names = {name for name, _ in source.named_buffers()}
    assert set(checkpoint) == parameters.keys() | buffer_names
    assert any(name.startswith("0.pipeline.normalization.") for name in checkpoint)
    assert any(name.startswith("0.pipeline.residual.") for name in checkpoint)
    for name, parameter in parameters.items():
        assert source.get_parameter(name) is parameter
    registered_modules = dict(source.named_modules(remove_duplicate=False))
    assert set(checkpoint._metadata) == set(registered_modules)
    for name, module in registered_modules.items():
        assert source.get_submodule(name) is module
    destination.load_state_dict(checkpoint, strict=True)
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, checkpoint[name])

    retired = {
        key.replace("0.pipeline.", "0."): value for key, value in checkpoint.items()
    }
    assert retired.keys() != checkpoint.keys()
    with pytest.raises(RuntimeError, match="Missing key"):
        destination.load_state_dict(retired, strict=True)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        destination.load_state_dict(checkpoint | retired, strict=True)
