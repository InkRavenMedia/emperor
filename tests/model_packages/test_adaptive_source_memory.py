from dataclasses import replace

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdditiveDynamicBiasConfig,
    DualModelDynamicWeightConfig,
    SingleModelDynamicWeightConfig,
    SumGroupingConfig,
)
from emperor.layers import LayerConfig, LayerStackConfig
from models._projection_model import (
    ProjectionModelOptions,
    resolve_projection_model,
)
from models.catalog import discover_model_packages, model_package
from tests.model_packages.test_attention_source_memory import run_model
from tests.model_packages.test_grouped_attention_memory import (
    build_config,
    parse_overrides,
)
from tests.model_packages.test_memory_expert_parity import inputs_for
from tests.model_packages.test_mlp_mixer_source_memory import (
    configuration as mixer_config,
)

MEMORY_FLAGS = {"ATTN_SOURCE_MEMORY_FLAG", "TOKEN_MIXER_SOURCE_MEMORY_FLAG"}
PACKAGES = [
    package.catalog_key
    for package in discover_model_packages()
    if MEMORY_FLAGS.intersection(package.runtime_defaults_spec.supported_keys)
]


@pytest.fixture(autouse=True)
def fixed_random_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(705)
        yield


@pytest.mark.parametrize("name", PACKAGES)
def test_named_adaptive_memory_uses_separate_trainable_generator(name):
    family = name.split("/")[0]
    prefix = "token-mixer" if family == "mlp_mixer" else "attn"
    overrides = parse_overrides(
        name,
        f"--{prefix}-source-memory-flag",
        "true",
        f"--{prefix}-source-memory-chunk-size",
        "2",
        f"--{prefix}-source-memory-model-option",
        "ADAPTIVE_LINEAR",
    )
    if "adaptive" in name:
        scope = ""
        if family == "transformer":
            scope = "attention_expert_adaptive_" if "expert" in name else "attn_"
        overrides.update(
            {
                scope + "weight_option_flag": True,
                scope + "weight_option": DualModelDynamicWeightConfig,
                scope + "generator_depth": 3,
            }
        )
    if name.endswith("_cluster"):
        overrides["cluster_x_axis_total_neurons"] = 1
    if name == "transformer/expert_linear_adaptive":
        overrides["encoder_attn_expert_adaptive_generator_depth"] = 4
        overrides["decoder_self_attn_expert_adaptive_generator_depth"] = 5
    config = (
        mixer_config(name, **overrides)
        if family == "mlp_mixer"
        else build_config(name, overrides)
    )
    model = model_package(name).build_model(config).eval()
    memories = [
        module
        for path, module in model.named_modules()
        if path.endswith(".source_memory")
    ]
    assert len(memories) >= (2 if family == "transformer" else 1)
    assert all(memory.context.chunk_size == 2 for memory in memories)
    external_parameters = {
        p.data_ptr()
        for path, p in model.named_parameters(remove_duplicate=False)
        if ".source_memory." not in path
    }
    depths = []
    for memory in memories:
        assert isinstance(memory.model.cfg, AdaptiveLinearLayerConfig)
        augmentation = memory.model.cfg.adaptive_augmentation_config
        if "adaptive" in name:
            assert isinstance(augmentation.weight_config, DualModelDynamicWeightConfig)
            depths.append(augmentation.weight_config.generator_depth)
        else:
            assert isinstance(augmentation.bias_config, AdditiveDynamicBiasConfig)
        assert external_parameters.isdisjoint(
            p.data_ptr() for p in memory.model.parameters()
        )
    if depths:
        assert set(depths) == (
            {4, 5} if name == "transformer/expert_linear_adaptive" else {3}
        )
    inputs = inputs_for(name)
    if family == "transformer":
        result = model(inputs, inputs)
        logits = result[0] if isinstance(result, tuple) else result
    else:
        logits = run_model(model, name, inputs)[0]
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    for memory in memories:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in memory.model.adaptive_behaviour.parameters()
        )


def test_grouped_expert_keeps_its_grouping_and_memory_keeps_its_chunk_boundaries():
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterAugmentationConfig,
    )
    from emperor.experts import MixtureOfExpertsConfig

    original = AdaptiveParameterAugmentationConfig(
        weight_config=DualModelDynamicWeightConfig(generator_depth=4),
        grouping_config=SumGroupingConfig(chunk_size=7),
    )
    expert = AdaptiveLinearLayerConfig(
        input_dim=3,
        output_dim=5,
        bias_flag=False,
        adaptive_augmentation_config=original,
    )
    mixture = MixtureOfExpertsConfig(
        expert_model_config=LayerStackConfig(
            layer_config=LayerConfig(layer_model_config=expert)
        )
    )
    memory = resolve_projection_model(
        None, ProjectionModelOptions.ADAPTIVE_LINEAR, None, template_source=mixture
    )
    assert memory.input_dim is None and memory.output_dim is None
    assert memory.bias_flag is False
    assert memory.adaptive_augmentation_config == replace(
        original, grouping_config=None
    )
    assert (
        memory.adaptive_augmentation_config.weight_config is not original.weight_config
    )
    assert original.grouping_config.chunk_size == 7


def test_memory_preserves_disabled_expert_generators():
    name = "vit/expert_linear_adaptive"
    config = build_config(
        name,
        {
            "attn_source_memory_flag": True,
            "attn_source_memory_chunk_size": 2,
            "attn_source_memory_model_option": ProjectionModelOptions.ADAPTIVE_LINEAR,
            "weight_option_flag": False,
            "bias_option_flag": False,
            "diagonal_option_flag": False,
            "mask_option_flag": False,
        },
    )
    model = model_package(name).build_model(config)
    memory = next(
        m for path, m in model.named_modules() if path.endswith(".source_memory.model")
    )
    assert isinstance(memory.cfg, AdaptiveLinearLayerConfig)
    assert memory.adaptive_behaviour is None


@pytest.mark.parametrize(
    "variant", ["linear", "linear_adaptive", "expert_linear", "expert_linear_adaptive"]
)
def test_mixer_accepts_one_token_per_adaptive_source_memory_chunk(variant):
    name = f"mlp_mixer/{variant}"
    options = dict(
        token_mixer_source_memory_flag=True,
        token_mixer_source_memory_model_option=ProjectionModelOptions.ADAPTIVE_LINEAR,
    )
    with pytest.raises(ValueError, match="--token-mixer-source-memory-chunk-size"):
        model_package(name).build_model(mixer_config(name, **options))
    config = mixer_config(name, **options, token_mixer_source_memory_chunk_size=1)
    model = model_package(name).build_model(config).eval()
    memory = next(
        m for path, m in model.named_modules() if path.endswith(".source_memory")
    )
    assert memory.context.chunk_size == 1
    assert isinstance(memory.model.cfg, AdaptiveLinearLayerConfig)
    result = run_model(model, name, inputs_for(name))[0]
    assert torch.isfinite(result).all()


def test_plain_vit_can_use_bias_free_adaptive_memory():
    name = "vit/linear"
    config = build_config(
        name,
        {
            "attn_source_memory_flag": True,
            "attn_source_memory_chunk_size": 2,
            "attn_source_memory_model_option": ProjectionModelOptions.ADAPTIVE_LINEAR,
            "attn_source_memory_model_bias_flag": False,
        },
    )
    model = model_package(name).build_model(config).eval()
    memory = next(
        m for path, m in model.named_modules() if path.endswith(".source_memory.model")
    )
    assert memory.cfg.bias_flag is False
    assert isinstance(
        memory.cfg.adaptive_augmentation_config.weight_config,
        SingleModelDynamicWeightConfig,
    )
    run_model(model, name, inputs_for(name))[0].square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in memory.adaptive_behaviour.parameters()
    )
