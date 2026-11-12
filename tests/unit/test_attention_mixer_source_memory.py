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
from emperor.sampler import RouterConfig
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
@pytest.mark.parametrize("chunked_memory", [True, False])
@pytest.mark.parametrize("additive_mask", [True, False])
def test_memory_uses_masked_original_tokens_with_soft_mixing(
    batch_first, chunked_memory, additive_mask
):
    from emperor.attention import AttentionSourceMemoryConfig
    from emperor.augmentations.adaptive_parameters import SumGroupingConfig
    from emperor.linears import LinearLayerConfig
    from tests.unit.test_expert_soft_moe import model_config as soft_moe_model_config

    torch.manual_seed(241)
    config = replace(
        _mixer_config(sequence_length=8, batch_first_flag=batch_first),
        mixing_model_config=soft_moe_model_config(dimension=8, layers=1),
        source_memory_config=(
            AttentionSourceMemoryConfig(
                grouping_config=SumGroupingConfig(chunk_size=5),
                model_config=LinearLayerConfig(bias_flag=True),
            )
            if chunked_memory
            else memory_config()
        ),
    )
    model = config.build().double().eval()
    batch_tokens = torch.randn(2, 7, 4, dtype=torch.float64)
    excluded_tokens = torch.tensor([[False] * 6 + [True], [True] * 7])
    poisoned_tokens = batch_tokens.masked_fill(
        excluded_tokens.unsqueeze(-1), float("nan")
    )
    inputs = poisoned_tokens if batch_first else poisoned_tokens.transpose(0, 1)
    inputs.requires_grad_()
    padding_mask = excluded_tokens
    if additive_mask:
        padding_mask = torch.zeros_like(
            excluded_tokens, dtype=torch.float64
        ).masked_fill(excluded_tokens, -torch.inf)

    memory_inputs = []
    mixing_inputs = []
    memory_hook = model.source_memory.register_forward_pre_hook(
        lambda _model, args: memory_inputs.append(args[0].detach().clone())
    )
    mixing_hook = model.mixing_model.register_forward_pre_hook(
        lambda _model, args: mixing_inputs.append(args[0].hidden.detach().clone())
    )
    output, attention_weights, auxiliary_loss = model(
        inputs, inputs, inputs, padding_mask
    )
    memory_hook.remove()
    mixing_hook.remove()

    masked_tokens = batch_tokens.masked_fill(excluded_tokens.unsqueeze(-1), 0)
    if chunked_memory:
        expected_memory_input = masked_tokens.transpose(0, 1)
    else:
        layout_tokens = masked_tokens if batch_first else masked_tokens.transpose(0, 1)
        expected_memory_input = layout_tokens.reshape(-1, 4)
    assert len(memory_inputs) == len(mixing_inputs) == 1
    torch.testing.assert_close(memory_inputs[0], expected_memory_input)
    torch.testing.assert_close(mixing_inputs[0][..., :7], masked_tokens.transpose(1, 2))
    assert output.shape == inputs.shape and attention_weights is None
    output_exclusions = (
        excluded_tokens if batch_first else excluded_tokens.transpose(0, 1)
    )
    assert torch.isfinite(output).all()
    assert (output[output_exclusions] == 0).all()

    objective = output.square().sum()
    if auxiliary_loss is not None:
        objective = objective + auxiliary_loss
    objective.backward()
    assert torch.isfinite(inputs.grad).all()
    assert (inputs.grad[output_exclusions] == 0).all()
    for component in (model.mixing_model, model.source_memory):
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in component.parameters()
        )


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("with_memory", [True, False])
def test_soft_mixer_routes_channel_vectors_then_restores_token_layout(
    batch_first, with_memory
):
    from emperor.experts import MixtureOfExpertsLayerState, SoftMoEConfig
    from emperor.transformer._soft_moe_config import build_soft_moe_stack
    from tests.unit.test_expert_soft_moe import scoring_config

    config = _mixer_config(batch_first_flag=batch_first)
    config.mixing_model_config = build_soft_moe_stack(
        config.mixing_model_config,
        SoftMoEConfig(
            num_experts=2,
            slots_per_expert=2,
            router_config=RouterConfig(
                noisy_topk_flag=False, model_config=scoring_config()
            ),
        ),
        3,
    )
    config.source_memory_config = memory_config() if with_memory else None
    model = config.build().double().eval()
    inputs = torch.randn(2, 3, 4, dtype=torch.double)
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    sequence_axis = 1 if batch_first else 0
    extended = inputs
    if with_memory:
        memory = model.source_memory(inputs.reshape(-1, 4)).reshape_as(inputs)
        extended = torch.cat((inputs, memory), dim=sequence_axis)
    vectors = extended.movedim(sequence_axis, -1)
    assert vectors.shape == (2, 4, 6 if with_memory else 3)
    expected = model.mixing_model(MixtureOfExpertsLayerState(hidden=vectors)).hidden
    expected = expected[..., :3].movedim(-1, sequence_axis)
    actual, weights, _ = model(inputs, inputs, inputs)
    torch.testing.assert_close(actual, expected)
    assert weights is None and actual.shape == inputs.shape
    actual.square().sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.mixing_model.parameters()
    )
    if with_memory:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.source_memory.parameters()
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
