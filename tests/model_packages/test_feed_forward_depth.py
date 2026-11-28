import pytest

from emperor.experts import MixtureOfExpertsModelConfig
from emperor.layers import MirroredLayerStack
from emperor.transformer import FeedForward
from model_runtime.inspection.materialization import materialize_inspection
from model_runtime.inspection.records import InspectionRequest
from models.catalog import discover_model_packages


def _inspect_feed_forward_stacks(package, **overrides):
    spec = package.runtime_defaults_spec
    small_model = {
        "hidden_dim": 16,
        "model_dim": 16,
        "stack_num_layers": 1,
        "encoder_num_layers": 1,
        "decoder_num_layers": 1,
        "attn_num_heads": 2,
        "num_experts": 4,
        "expert_stack_hidden_dim": 8,
        "adaptive_generator_stack_hidden_dim": 8,
    }
    options = {
        key: value
        for key, value in small_model.items()
        if spec.resolve_key(key.upper()) is not None
    }
    model = materialize_inspection(
        package,
        InspectionRequest(
            preset=package.preset_name(package.default_preset),
            overrides=options | overrides,
        ),
    ).model
    stacks = {}
    for name, module in model.named_modules():
        if isinstance(module, FeedForward):
            stack = module.model
            if isinstance(module.stack_config, MixtureOfExpertsModelConfig):
                stack = stack.expert_stack
            assert isinstance(stack, MirroredLayerStack), name
            stacks[name] = (len(stack.layers), stack.hidden_dim)
    assert stacks
    return stacks


@pytest.mark.parametrize(
    "package",
    [
        package
        for package in discover_model_packages()
        if "FF_NUM_LAYERS" in package.runtime_defaults_spec.supported_keys
    ],
    ids=lambda package: package.catalog_key,
)
def test_feed_forward_depth_and_width_reach_built_model(package):
    stacks = _inspect_feed_forward_stacks(
        package, ff_num_layers=3, ff_stack_hidden_dim=24
    )
    assert set(stacks.values()) == {(6, 24)}, stacks


@pytest.mark.parametrize(
    "package",
    [
        package
        for package in discover_model_packages()
        if package.identity.model_type == "transformer"
    ],
    ids=lambda package: package.catalog_key,
)
def test_transformer_scoped_feed_forward_depth_and_width_reach_built_model(package):
    stacks = _inspect_feed_forward_stacks(
        package,
        ff_num_layers=1,
        ff_stack_hidden_dim=16,
        encoder_ff_num_layers=2,
        encoder_ff_stack_hidden_dim=24,
        decoder_ff_num_layers=3,
        decoder_ff_stack_hidden_dim=32,
    )
    assert {
        "encoder": {shape for name, shape in stacks.items() if "encoder" in name},
        "decoder": {shape for name, shape in stacks.items() if "decoder" in name},
    } == {"encoder": {(4, 24)}, "decoder": {(6, 32)}}
