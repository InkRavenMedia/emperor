"""Optional temporary source memory through each public MLP-Mixer package."""

import pytest
import torch

from emperor.attention._variants.mixer.layer import MixerAttention
from emperor.memory import AttentionDynamicMemoryConfig
from models.catalog import model_package

PACKAGES = tuple(
    f"mlp_mixer/{variant}"
    for variant in (
        "linear",
        "linear_adaptive",
        "expert_linear",
        "expert_linear_adaptive",
    )
)


def configuration(name, **updates):
    options = dict(
        batch_size=2,
        input_dim=64,
        hidden_dim=8,
        output_dim=3,
        image_patch_size=4,
        input_channels=1,
        image_height=8,
        stack_num_layers=1,
        stack_dropout_probability=0.0,
        token_mixer_stack_dropout_probability=0.0,
        channel_mixer_stack_dropout_probability=0.0,
        token_mixer_stack_hidden_dim=6,
        channel_mixer_stack_hidden_dim=12,
        submodule_stack_hidden_dim=4,
        submodule_stack_num_layers=1,
    )
    if "adaptive" in name:
        options.update(
            adaptive_generator_stack_hidden_dim=4, adaptive_generator_stack_num_layers=1
        )
    if "expert" in name:
        options.update(
            num_experts=3,
            top_k=2,
            capacity_factor=0.0,
            expert_stack_hidden_dim=8,
            router_stack_hidden_dim=4,
        )
    options.update(updates)
    return model_package(name).build_configuration(config_overrides=options)


@pytest.mark.parametrize("name", PACKAGES)
def test_mlp_mixer_source_memory_is_optional_and_differentiable(name):
    package = model_package(name)
    disabled = package.build_model(configuration(name))
    mixers = [m for m in disabled.modules() if isinstance(m, MixerAttention)]
    assert mixers and all(m.source_memory is None for m in mixers)
    assert all(m.mixing_sequence_length == m.sequence_length == 4 for m in mixers)
    enabled_config = configuration(
        name,
        token_mixer_source_memory_flag=True,
        token_mixer_source_memory_option=AttentionDynamicMemoryConfig,
        token_mixer_source_memory_num_memory_slots=3,
    )
    model = package.build_model(enabled_config).eval()
    mixers = [m for m in model.modules() if isinstance(m, MixerAttention)]
    calls = []
    handles = []
    for mixer in mixers:
        assert isinstance(mixer.source_memory.cfg, AttentionDynamicMemoryConfig)
        assert mixer.source_memory.num_memory_slots == 3
        assert mixer.mixing_sequence_length == 8 and mixer.sequence_length == 4
        handles.append(
            mixer.register_forward_hook(lambda _m, _a, result: calls.append(result[0]))
        )
    inputs = torch.randn(2, 1, 8, 8)
    result = model(inputs)
    logits = result[0] if isinstance(result, tuple) else result
    assert logits.shape == (2, 3)
    assert len(calls) == len(mixers) and all(out.shape == (2, 4, 8) for out in calls)
    logits.square().mean().backward()
    for mixer in mixers:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in mixer.source_memory.parameters()
        )
    for handle in handles:
        handle.remove()
    restored = package.build_model(enabled_config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    result = restored(inputs)
    restored_logits = result[0] if isinstance(result, tuple) else result
    torch.testing.assert_close(restored_logits, logits)


@pytest.mark.parametrize("name", PACKAGES)
def test_source_memory_public_options_parse_and_coexist_with_existing_controls(name):
    from emperor.memory import (
        ElementWiseWeightedDynamicMemoryConfig,
        GatedResidualDynamicMemoryConfig,
        WeightedDynamicMemoryConfig,
    )
    from model_runtime.inspection.schema import configuration_schema

    package = model_package(name)
    spec = package.runtime_defaults_spec
    raw = {
        "TOKEN_MIXER_SOURCE_MEMORY_FLAG": "true",
        "TOKEN_MIXER_SOURCE_MEMORY_NUM_MEMORY_SLOTS": "3",
        "TOKEN_MIXER_SOURCE_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE": "0.01",
        "TOKEN_MIXER_SOURCE_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS": "1",
    }
    fields = {field.key for field in configuration_schema(package).fields}
    assert fields.issuperset(raw)
    assert "TOKEN_MIXER_SOURCE_MEMORY_OPTION" in fields
    options = {key.lower(): spec.parse_value(key, value) for key, value in raw.items()}
    for variant in (
        GatedResidualDynamicMemoryConfig,
        WeightedDynamicMemoryConfig,
        ElementWiseWeightedDynamicMemoryConfig,
        AttentionDynamicMemoryConfig,
    ):
        options["token_mixer_source_memory_option"] = spec.parse_value(
            "TOKEN_MIXER_SOURCE_MEMORY_OPTION", variant.__name__
        )
        config = configuration(
            name,
            **options,
            token_mixer_memory_flag=True,
            token_mixer_recurrent_flag=True,
            token_mixer_stack_hidden_dim=8,
        )
        model = package.build_model(config).eval()
        for mixer in (m for m in model.modules() if isinstance(m, MixerAttention)):
            assert type(mixer.source_memory.cfg) is variant
            assert mixer.source_memory.test_time_training_learning_rate == 0.01
            assert mixer.source_memory.test_time_training_num_inner_steps == 1
        inputs = torch.randn(2, 1, 8, 8)
        result = model(inputs)
        logits = result[0] if isinstance(result, tuple) else result
        assert logits.shape == (2, 3) and torch.isfinite(logits).all()
    options["token_mixer_source_memory_flag"] = False
    disabled = package.build_model(configuration(name, **options))
    assert all(
        m.source_memory is None
        for m in disabled.modules()
        if isinstance(m, MixerAttention)
    )
