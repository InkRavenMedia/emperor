"""Temporary token memory in MixerAttention, including retained-output gradients."""

from dataclasses import replace

import pytest
import torch

from emperor.memory import (
    AttentionDynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,
)
from emperor.memory._base import DynamicMemoryAbstract
from tests.unit.test_attention_model_mixer import (
    _linear_stack,
    _mixer_config,
    _mixture_mixing_config,
    _recurrent_stack,
)


def memory_config(variant=GatedResidualDynamicMemoryConfig, **options):
    return variant(
        memory_position_option=MemoryPositionOptions.AFTER_AFFINE,
        model_config=_linear_stack(4, 4),
        **options,
    )




@pytest.mark.parametrize("batch_first", [True, False])
def test_mixer_appends_memory_mixes_all_positions_and_keeps_original_outputs(
    batch_first,
):
    torch.manual_seed(163)
    config = replace(
        _mixer_config(batch_first_flag=batch_first),
        source_memory_config=memory_config(),
    )
    model = config.build().double()
    weight = torch.arange(36, dtype=torch.float64).reshape(6, 6) / 19
    bias = torch.linspace(-0.2, 0.3, 6, dtype=torch.float64)
    with torch.no_grad():
        linear = model.mixing_model.layers[0].model
        linear.weight_params.copy_(weight)
        linear.bias_params.copy_(bias)
    memory = next(m for m in model.modules() if isinstance(m, DynamicMemoryAbstract))
    calls = []
    handle = memory.register_forward_hook(
        lambda _module, args, out: calls.append((args[0], out))
    )
    inputs = torch.randn(2, 3, 4, dtype=torch.float64)
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    inputs.requires_grad_()
    output, attention_weights, loss = model(inputs, inputs, inputs)
    handle.remove()
    assert len(calls) == 1
    rows, generated = calls[0]
    torch.testing.assert_close(rows, inputs.reshape(-1, 4))
    sequence_axis = 1 if batch_first else 0
    combined = torch.cat((inputs, generated.reshape_as(inputs)), dim=sequence_axis)
    expanded = combined.movedim(sequence_axis, -1)
    expected = (expanded @ weight + bias)[..., :3].movedim(-1, sequence_axis)
    torch.testing.assert_close(output, expected)
    assert output.shape == inputs.shape
    assert attention_weights is None and loss is None
    # Only retained positions enter this loss; memory must still receive gradients.
    output.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in memory.parameters()
    )
    assert config.mixing_model_config.input_dim == 3
    assert config.source_memory_config.input_dim is None


@pytest.mark.parametrize(
    "variant",
    [
        GatedResidualDynamicMemoryConfig,
        WeightedDynamicMemoryConfig,
        ElementWiseWeightedDynamicMemoryConfig,
        AttentionDynamicMemoryConfig,
    ],
)
@pytest.mark.parametrize("steps", [None, 1])
def test_memory_variants_keep_adaptation_local_and_round_trip_checkpoints(
    variant, steps
):
    options = {"num_memory_slots": 2} if variant is AttentionDynamicMemoryConfig else {}
    config = replace(
        _mixer_config(),
        source_memory_config=memory_config(
            variant,
            test_time_training_learning_rate=0.01 if steps else None,
            test_time_training_num_inner_steps=steps,
            **options,
        ),
    )
    model = config.build()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    output = model(inputs, inputs, inputs)[0]
    assert output.shape == inputs.shape
    output.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.source_memory.parameters()
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    restored = config.build().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.inference_mode():
        expected = model(inputs, inputs, inputs)[0]
        changed = inputs + 8
        model(changed, changed, changed)
        torch.testing.assert_close(model(inputs, inputs, inputs)[0], expected)
        torch.testing.assert_close(restored(inputs, inputs, inputs)[0], expected)
        changed = inputs.clone()
        changed[1] += 10
        torch.testing.assert_close(model(changed, changed, changed)[0][0], expected[0])


@pytest.mark.parametrize("backend", ["experts", "recurrent", "recurrent_experts"])
def test_memory_extends_nested_mixer_dimensions_and_preserves_loss(backend):
    nested = (
        _mixture_mixing_config(auxiliary_loss_weight=0.1)
        if backend == "experts"
        else _recurrent_stack(7)
    )
    if backend == "recurrent_experts":
        nested.block_config = _mixture_mixing_config(auxiliary_loss_weight=0.1)
    config = replace(
        _mixer_config(mixing_model_config=nested), source_memory_config=memory_config()
    )
    model = config.build()
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    output, _, loss = model(inputs, inputs, inputs)
    assert output.shape == inputs.shape
    objective = output.square().sum()
    if "experts" in backend:
        assert loss is not None and torch.isfinite(loss) and loss > 0
        objective = objective + loss
    objective.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.source_memory.parameters()
    )
    assert torch.isfinite(inputs.grad).all()
    assert nested.input_dim == 7


def test_missing_memory_config_creates_no_memory_and_loads_existing_state_strictly():
    config = _mixer_config()
    assert config.source_memory_config is None
    torch.manual_seed(136)
    original = config.build()
    torch.manual_seed(136)
    disabled = replace(config, source_memory_config=None).build()
    assert disabled.source_memory is None
    assert not any(isinstance(m, DynamicMemoryAbstract) for m in disabled.modules())
    disabled.load_state_dict(original.state_dict(), strict=True)
    inputs = torch.randn(2, 3, 4)
    torch.testing.assert_close(
        original(inputs, inputs, inputs)[0], disabled(inputs, inputs, inputs)[0]
    )


def test_source_memory_rejects_grouped_generation():
    from tests.unit.test_attention_adaptive_grouping import (
        grouped_projection_model_config,
    )

    memory = memory_config()
    memory.model_config = grouped_projection_model_config()
    with pytest.raises(ValueError, match="source memory.*independent token"):
        replace(_mixer_config(), source_memory_config=memory).build()
