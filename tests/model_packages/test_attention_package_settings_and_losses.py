from dataclasses import fields, replace
from importlib import import_module
from unittest.mock import patch

import pytest
import torch
from torch import nn

from emperor.datasets.image.classification import Mnist
from emperor.layers import GateConfig, LayerConfig, LayerStackConfig
from emperor.linears import LinearLayer, LinearLayerConfig
from emperor.patch import LinearPatchEmbeddingConfig, PatchEmbeddingLinear
from emperor.sampler import SamplerConfig
from models.catalog import MODEL_CATALOG, model_package

PACKAGES = [
    package
    for name, package in MODEL_CATALOG.items()
    if name.split("/")[0] in {"bert", "gpt", "vit", "transformer"}
]


@pytest.mark.parametrize("package", PACKAGES, ids=lambda package: package.catalog_key)
def test_replacing_runtime_refreshes_builder_dimensions_atomically(package):
    family, name = package.catalog_key.split("/")
    module = import_module(f"models.{family}.{name}.config_builder")
    class_name = "".join(part.capitalize() for part in (family, *name.split("_")))
    builder = getattr(module, f"{class_name}ConfigBuilder")()
    replacement = package.bind_runtime_defaults({"batch_size": 7})
    builder.runtime = replacement
    with pytest.raises(TypeError, match="runtime"):
        builder.runtime = object()
    assert builder.runtime is replacement
    assert builder.build().batch_size == 7


@pytest.mark.parametrize("package", PACKAGES, ids=lambda package: package.catalog_key)
def test_missing_encoder_or_decoder_names_the_required_field(package):
    family = package.identity.model_type
    field = "decoder_config" if family == "gpt" else "encoder_config"
    if package.catalog_key.endswith("_cluster"):
        field = "neuron_cluster_config"
    config = package.build_configuration()
    stages = replace(config.experiment_config, **{field: None})
    with pytest.raises(ValueError, match=rf"experiment_config\.{field} is required"):
        package.build_model(replace(config, experiment_config=stages))


@pytest.mark.parametrize("package", PACKAGES, ids=lambda package: package.catalog_key)
def test_attention_projection_controllers_admit_only_valid_settings(package):
    runtime = package.bind_runtime_defaults({"attn_halting_threshold": 0.9})
    if package.identity.model_type == "transformer":
        controls = runtime.encoder_attention_options.layer_controller_options
    elif hasattr(runtime, "attention_projection_layer_controller_options"):
        controls = runtime.attention_projection_layer_controller_options
    else:
        controls = runtime._values["attention_projection_layer_controller_options"]
    with pytest.raises(ValueError, match="gate_option"):
        replace(controls, stack_gate_flag=True, gate_option=None)
    with pytest.raises(ValueError, match="shared_gate_config"):
        replace(controls, stack_gate_flag=True, shared_gate_config=GateConfig())
    with pytest.raises(ValueError, match="halting_threshold"):
        replace(controls, halting_threshold=0.0)
    assert replace(controls, halting_threshold=1.0).halting_threshold == 1.0


class _PenalizedLinear(LinearLayer):
    def __init__(self, cfg, overrides=None):
        super().__init__(cfg, overrides)
        self.penalty = nn.Parameter(torch.zeros(()))

    def get_auxiliary_loss(self):
        return self.penalty


class _PenalizedLinearConfig(LinearLayerConfig):
    def _registry_owner(self):
        return _PenalizedLinear


class _PenalizedPatch(PatchEmbeddingLinear):
    def __init__(self, cfg, overrides=None):
        super().__init__(cfg, overrides)
        self.penalty = nn.Parameter(torch.zeros(()))

    def get_auxiliary_loss(self):
        return self.penalty


class _PenalizedPatchConfig(LinearPatchEmbeddingConfig):
    def _registry_owner(self):
        return _PenalizedPatch


def _field_values(config):
    return {
        field.name: getattr(config, field.name)
        for field in fields(config)
        if field.init
    }


@pytest.mark.parametrize(
    "package",
    [p for p in PACKAGES if p.identity.model_type == "vit"],
    ids=lambda package: package.catalog_key,
)
def test_vit_keeps_zero_valued_penalties_from_patch_projection_patch_and_head(package):
    config = package.build_configuration(
        dataset=Mnist,
        config_overrides={"image_patch_size": 14, "stack_num_layers": 1},
    )
    stages = config.experiment_config
    stages.patch_config = _PenalizedPatchConfig(**_field_values(stages.patch_config))
    projection = stages.patch_config.embedding_stack_config.layer_config
    projection.layer_model_config = _PenalizedLinearConfig(
        **_field_values(projection.layer_model_config)
    )
    if isinstance(stages.output_config, LayerConfig):
        stages.output_config.layer_model_config = _PenalizedLinearConfig(
            **_field_values(stages.output_config.layer_model_config)
        )
    else:
        stages.output_config = _PenalizedLinearConfig(
            **_field_values(stages.output_config)
        )
    model = package.build_model(config).eval()
    logits, loss = model(torch.zeros(2, 1, 28, 28))
    assert logits.shape == (2, config.output_dim)
    loss.backward()
    with torch.no_grad():
        _, evaluation_loss = model(torch.zeros(2, 1, 28, 28))
    assert evaluation_loss.shape == ()
    penalties = [
        module.penalty
        for module in model.modules()
        if isinstance(module, (_PenalizedLinear, _PenalizedPatch))
    ]
    assert [penalty.grad.item() for penalty in penalties] == [1.0, 1.0, 1.0]


def test_transformer_sampler_only_built_for_matrix_mixtures():
    from emperor.augmentations.adaptive_parameters import MatrixWeightsMixtureConfig
    from models import _adaptive_parameters

    package = model_package("transformer/linear_adaptive")
    with patch.object(
        _adaptive_parameters, "SamplerConfig", wraps=SamplerConfig
    ) as sampler:
        package.build_configuration()
        sampler.assert_not_called()
        package.build_configuration(
            config_overrides={
                "encoder_attn_weight_option": MatrixWeightsMixtureConfig,
                "encoder_attn_weight_mixture_num_experts": 3,
            }
        )
    assert [call.kwargs["num_experts"] for call in sampler.call_args_list] == [3]


@pytest.mark.parametrize(
    "package",
    [p for p in PACKAGES if p.identity.model_type == "transformer"],
    ids=lambda package: package.catalog_key,
)
def test_scoped_gate_choices_validate_after_broadcast_overrides(package):
    from emperor.layers import LayerGateOptions

    runtime = package.bind_runtime_defaults(
        {
            "attn_gate_option": None,
            "encoder_attn_stack_gate_flag": True,
            "encoder_attn_gate_option": LayerGateOptions.MULTIPLIER,
        }
    )
    encoder = runtime.encoder_attention_options.layer_controller_options
    decoder = runtime.decoder_self_attention_options.layer_controller_options
    assert encoder.stack_gate_flag
    assert encoder.gate_option is LayerGateOptions.MULTIPLIER
    assert not decoder.stack_gate_flag
    assert decoder.gate_option is None


@pytest.mark.parametrize("family", ("bert", "gpt", "transformer"))
def test_language_models_include_each_positional_embedding_loss(family):
    package = model_package(f"{family}/linear")
    config = package.build_configuration()
    model = package.build_model(config).eval()
    positions = (
        (model.source_positional_embedding, model.target_positional_embedding)
        if family == "transformer"
        else (model.positional_embedding,)
    )
    for position in positions:
        position.penalty = nn.Parameter(torch.zeros(()))
        position.get_auxiliary_loss = lambda owner=position: owner.penalty
    ids = torch.tensor([[1, 2, 3]])
    output = model(ids, ids) if family == "transformer" else model(ids)
    output[-1].backward()
    assert all(position.penalty.grad.item() == 1 for position in positions)


@pytest.mark.parametrize(
    ("family", "build_method"),
    [("bert", "build_encoder_config"), ("gpt", "build_decoder_config")],
)
def test_dense_projection_uses_gate_selected_during_stack_construction(
    family, build_method
):
    module = import_module(f"models.{family}.linear._core_config_factory")
    attention_gate = GateConfig()
    feed_forward_gate = GateConfig()
    projections = []

    class ProjectionFactory:
        def build_submodule_stack_config(self, options):
            if options is core.attention_projection_stack_options:
                core.attention_projection_layer_controller_options = replace(
                    core.attention_projection_layer_controller_options,
                    shared_gate_config=attention_gate,
                )
            if options is core.feed_forward_stack_options:
                core.feed_forward_layer_controller_options = replace(
                    core.feed_forward_layer_controller_options,
                    shared_gate_config=feed_forward_gate,
                )
            stack = LayerStackConfig(
                hidden_dim=options.hidden_dim, num_layers=1, layer_config=LayerConfig()
            )
            projections.append(stack)
            return stack

    settings = dict.fromkeys(
        field.name for field in fields(module.CoreConfigDependencies)
    )
    settings.update(
        batch_size=1, sequence_length=5, linear_layer_config_factory=ProjectionFactory()
    )
    core = module.CoreConfigFactory(module.CoreConfigDependencies(**settings))
    getattr(core, build_method)()

    attention, feed_forward = projections
    assert attention.shared_gate_config is attention_gate
    assert feed_forward.shared_gate_config is feed_forward_gate
