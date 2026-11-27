"""Public compression controls, causal decoding, and adaptive expert integration."""

from dataclasses import replace

import pytest
import torch

from emperor.augmentations import adaptive_parameters
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    LowRankDynamicWeightConfig,
)
from emperor.experts import MixtureOfExpertsConfig
from emperor.layers import (
    LayerState,
    RecurrentCompositionConfig,
    RecurrentCompressionConfig,
)
from model_runtime.inspection import configuration_schema
from model_runtime.packages.configuration import serialize_config_value
from models.catalog import discover_model_packages, model_package
from tests.model_packages.test_attention_residual_options import walk
from tests.model_packages.test_bert_expert_adaptive_cli import (
    configuration_from_cli,
    readme_arguments,
)
from tests.model_packages.test_memory_expert_parity import (
    configuration,
    inputs_for,
    run_model,
)
from tests.model_packages.test_soft_moe_cli import cli_values
from tests.unit.test_recurrent_compression import adaptive_scorer


def adaptive_generator_cases():
    options = (
        ("weight", "weight_option", adaptive_parameters.DynamicWeightConfig),
        ("bias", "bias_option", adaptive_parameters.DynamicBiasConfig),
        ("diagonal", "diagonal_option", adaptive_parameters.DynamicDiagonalConfig),
        ("mask", "row_mask_option", adaptive_parameters.AxisMaskConfig),
    )
    for component, option, base in options:
        for name in adaptive_parameters.__all__:
            variant = getattr(adaptive_parameters, name)
            if (
                isinstance(variant, type)
                and issubclass(variant, base)
                and variant is not base
            ):
                yield pytest.param(component, option, variant, id=name)


@pytest.mark.parametrize(("component", "option", "variant"), adaptive_generator_cases())
def test_recurrent_scorer_inherits_each_adaptive_generator(component, option, variant):
    width = 4
    slots = (
        width if variant is adaptive_parameters.SingleModelDynamicWeightConfig else 3
    )
    options = {
        "recurrent_flag": True,
        "hidden_dim": width,
        "stack_num_layers": 1,
        "adaptive_generator_stack_hidden_dim": width,
        "adaptive_generator_stack_num_layers": 1,
        f"{component}_option_flag": True,
        option: variant,
    }
    if variant is adaptive_parameters.BroadcastSumDynamicWeightConfig:
        options["weight_normalization_option"] = (
            adaptive_parameters.WeightNormalizationOptions.SOFTSIGN
        )
    if variant in (
        adaptive_parameters.MatrixWeightsMixtureConfig,
        adaptive_parameters.MatrixBiasMixtureConfig,
    ):
        options[f"{component}_mixture_num_experts"] = 2
    if variant is adaptive_parameters.MatrixBiasMixtureConfig:
        options["bias_mixture_top_k"] = 2
    package = model_package("linears/linear_adaptive")
    owner = package.build_configuration(
        config_overrides=options
    ).experiment_config.model_config
    template = owner.block_config.layer_config.layer_model_config
    owner = replace(
        owner,
        input_dim=width,
        output_dim=width,
        compression_config=RecurrentCompressionConfig(max_slots=slots),
    )
    recurrent = owner.build()
    scorer = recurrent.step.compressor.scorer
    assert isinstance(getattr(scorer.adaptive_cfg, f"{component}_config"), variant)
    assert (
        recurrent.step.compressor.cfg.scorer_config.adaptive_augmentation_config
        == template.adaptive_augmentation_config
    )
    tokens = torch.randn(7, width, requires_grad=True)
    result = recurrent(LayerState(tokens)).hidden
    assert result.shape == tokens.shape and torch.isfinite(result).all()
    result.square().mean().backward()
    assert (
        scorer.weight_params.grad is not None
        and torch.isfinite(scorer.weight_params.grad).all()
    )


@pytest.mark.parametrize(
    "package_name",
    [
        pkg.catalog_key
        for pkg in discover_model_packages()
        if "adaptive" in pkg.catalog_key
    ],
)
def test_every_adaptive_package_inherits_its_generator_in_recurrent_compression(
    package_name,
):
    package = model_package(package_name)
    options = dict(
        recurrent_flag=True,
        weight_option_flag=True,
        weight_option=LowRankDynamicWeightConfig,
        generator_depth=2,
        adaptive_generator_stack_hidden_dim=4,
        adaptive_generator_stack_num_layers=1,
    )
    slots = 3
    if (
        "RECURRENT_COMPRESSION_SLOTS"
        in package.runtime_defaults_spec.ordered_configuration_keys()
    ):
        options["recurrent_compression_slots"] = slots
    configuration = package.build_configuration(config_overrides=options)
    owners = [
        config
        for config in walk(configuration.experiment_config)
        if isinstance(config, RecurrentCompositionConfig)
    ]
    assert owners
    for owner in owners:
        width = owner.output_dim or configuration.hidden_dim
        if "recurrent_compression_slots" in options:
            resolved = owner.resolve_compression_config()
            compressor = resolved.build(RecurrentCompressionConfig(input_dim=width))
            tokens = torch.randn(2, 5, width)
            result = compressor(torch.nn.Identity(), LayerState(tokens))
        else:
            owner = replace(
                owner, compression_config=RecurrentCompressionConfig(max_slots=slots)
            )
            recurrent = owner.build(type(owner)(input_dim=width, output_dim=width))
            compressor = recurrent.step.compressor
            tokens = torch.randn(5, width)
            result = recurrent(LayerState(tokens))
        scorer = compressor.scorer
        assert isinstance(scorer.cfg, AdaptiveLinearLayerConfig)
        assert isinstance(scorer.weight_config, LowRankDynamicWeightConfig)
        assert scorer.weight_config.generator_depth == 2
        assert (scorer.input_dim, scorer.output_dim) == (width, slots)
        assert result.hidden.shape == tokens.shape
        result.hidden.square().mean().backward()
        assert any(
            p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
            for p in scorer.adaptive_behaviour.parameters()
        )


@pytest.mark.parametrize(
    "package_name",
    ["transformer/linear_adaptive", "transformer/expert_linear_adaptive"],
)
def test_encoder_and_decoder_inherit_their_own_feed_forward_options(package_name):
    package = model_package(package_name)
    experiment = package.build_configuration(
        config_overrides=dict(
            recurrent_flag=True,
            recurrent_compression_slots=3,
            model_dim=8,
            weight_option=adaptive_parameters.HypernetworkDynamicWeightConfig,
            encoder_ff_adaptive_weight_option=adaptive_parameters.DualModelDynamicWeightConfig,
            decoder_ff_adaptive_weight_option=LowRankDynamicWeightConfig,
        )
    ).experiment_config
    for owner, expected in (
        (experiment.encoder_config, adaptive_parameters.DualModelDynamicWeightConfig),
        (experiment.decoder_config, LowRankDynamicWeightConfig),
    ):
        scorer = owner.resolve_compression_config().scorer_config
        assert isinstance(scorer.adaptive_augmentation_config.weight_config, expected)


def test_package_schema_and_roundtrip():
    keys = {
        "RECURRENT_COMPRESSION_CONFIG",
        "RECURRENT_COMPRESSION_SLOTS",
        "RECURRENT_COMPRESSION_SCORER_CONFIG",
    }
    compression = RecurrentCompressionConfig(
        max_slots=2, scorer_config=adaptive_scorer(8, 2, grouped=True)
    )
    for package in discover_model_packages():
        if package.catalog_key.split("/")[0] not in {
            "gpt",
            "bert",
            "vit",
            "mlp_mixer",
            "transformer",
        }:
            continue
        assert keys <= {field.key for field in configuration_schema(package).fields}, (
            package.catalog_key
        )
        experiment = package.build_configuration(
            config_overrides={
                "recurrent_flag": True,
                "recurrent_compression_slots": 2,
            }
        ).experiment_config
        stacks = [
            getattr(experiment, key)
            for key in ("encoder_config", "decoder_config")
            if hasattr(experiment, key)
        ]
        if hasattr(experiment, "neuron_cluster_config"):
            stacks.append(
                experiment.neuron_cluster_config.neuron_config.nucleus_config.model_config
            )
        assert stacks and all(
            stack.compression_config.max_slots == 2 for stack in stacks
        )
        spec = package.runtime_defaults_spec
        decoded = spec.parse_value(
            "RECURRENT_COMPRESSION_CONFIG", serialize_config_value(compression)
        )
        assert (
            package.bind_runtime_defaults(
                {"recurrent_compression_config": decoded}
            ).recurrent_compression_config
            == compression
        )


@pytest.mark.parametrize(
    "name", ["bert/linear", "gpt/linear", "vit/linear", "mlp_mixer/linear"]
)
def test_package_cli_forward_backward_and_checkpoint(name):
    torch.manual_seed(15)
    package = model_package(name)
    options = cli_values(
        package,
        [
            "--recurrent-flag",
            "true",
            "--recurrent-max-steps",
            "2",
            "--recurrent-initial-iterations",
            "2",
            "--recurrent-compression-slots",
            "2",
            "--recurrent-compression-scorer-config",
            serialize_config_value(adaptive_scorer(8, 2)),
        ],
    )
    config = configuration(name, **options)
    model = package.build_model(config).eval()
    inputs = inputs_for(name)
    logits = run_model(model, name, inputs)[0]
    logits.square().mean().backward()
    gradients = [
        p.grad for key, p in model.named_parameters() if "compressor.scorer" in key
    ]
    assert any(g is not None and g.abs().sum() > 0 for g in gradients)
    if name.startswith("gpt"):
        modified = inputs.clone()
        modified[:, 2:] = 6
        torch.testing.assert_close(
            run_model(model, name, modified)[0][:, :2], logits[:, :2]
        )
        torch.testing.assert_close(
            run_model(model, name, inputs[:, :2])[0], logits[:, :2]
        )
    restored = package.build_model(config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(run_model(restored, name, inputs)[0], logits)


def test_gpt_causal_compression_padding_preserves_auxiliary_loss_and_gradients():
    torch.manual_seed(31)
    name = "gpt/expert_linear"
    config = configuration(
        name,
        attn_source_memory_flag=False,
        recurrent_flag=True,
        recurrent_initial_iterations=1,
        recurrent_max_steps=1,
        recurrent_compression_slots=2,
        sampler_switch_loss_weight=0.1,
    )
    model = model_package(name).build_model(config).eval()
    tokens = torch.tensor([[1, 2, 0, 0], [2, 1, 0, 0]])
    mask = tokens.ne(0).long()
    short, short_loss = model(tokens[:, :2], mask[:, :2])
    padded, padded_loss = model(tokens, mask)
    torch.testing.assert_close(padded[:, :2], short)
    assert short_loss > 0
    torch.testing.assert_close(padded_loss, short_loss)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    expected = torch.autograd.grad(short_loss, parameters, allow_unused=True)
    actual = torch.autograd.grad(padded_loss, parameters, allow_unused=True)
    assert any(grad is not None and grad.abs().sum() > 0 for grad in expected)
    torch.testing.assert_close(actual, expected)


def test_translation_encoder_decoder_and_source_context():
    torch.manual_seed(14)
    package = model_package("transformer/linear")
    config = package.build_configuration(
        config_overrides=dict(
            batch_size=2,
            vocab_size=32,
            model_dim=8,
            source_sequence_length=4,
            target_sequence_length=4,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            ff_stack_hidden_dim=8,
            dropout_probability=0.0,
            recurrent_flag=True,
            recurrent_max_steps=2,
            recurrent_initial_iterations=2,
            recurrent_compression_slots=2,
        )
    )
    model = package.build_model(config).eval()
    stacks = config.experiment_config
    assert (
        stacks.encoder_config.compression_config.max_slots
        == stacks.decoder_config.compression_config.max_slots
        == 2
    )
    source = torch.tensor([[2, 8, 3, 0], [2, 9, 10, 3]])
    target = torch.tensor([[2, 11, 3], [2, 12, 3]])
    logits = model(source, target)
    if isinstance(logits, tuple):
        logits = logits[0]
    changed = target.clone()
    changed[:, -1] = 15
    other = model(source, changed)
    if isinstance(other, tuple):
        other = other[0]
    torch.testing.assert_close(logits[:, :2], other[:, :2])
    logits.square().mean().backward()
    assert (
        len(
            [
                p
                for name, p in model.named_parameters()
                if "compressor.scorer.weight_params" in name and p.grad is not None
            ]
        )
        == 2
    )


def test_documented_bert_with_soft_experts_grouped_weights_and_memory():
    package = model_package("bert/expert_linear_adaptive")
    config = configuration_from_cli(
        readme_arguments()
        + [
            "--recurrent-compression-slots",
            "20",
            "--input-dim",
            "16",
            "--output-dim",
            "16",
            "--batch-size",
            "2",
        ]
    )
    model = package.build_model(config).eval()
    model.set_token_vocabulary([f"token{i}" for i in range(16)])
    tokens = torch.randint(16, (2, 100))
    mask = torch.ones_like(tokens)
    mask[1, 12:] = 0
    with torch.no_grad():
        logits = model(tokens, mask, torch.zeros_like(tokens))[0]
    assert logits.shape == (2, 100, 16) and torch.isfinite(logits).all()


@pytest.mark.parametrize("soft_moe", [False, True])
def test_clustered_vit_compression_inherits_low_rank_expert_settings(soft_moe):
    from tests.model_packages.vit.expert_linear_adaptive_cluster.test_model import (
        model,
        split,
    )

    classifier = model(
        preset="low-rank-expert-weight",
        recurrent_flag=True,
        recurrent_compression_slots=2,
        cluster_x_axis_total_neurons=1,
        soft_moe_flag=soft_moe,
    )
    recurrent = classifier.transformer.cluster["neuron_1_1_1"].nucleus.model
    scorer = recurrent.step.compressor.scorer
    assert (scorer.input_dim, scorer.output_dim) == (32, 2)
    assert isinstance(scorer.weight_config, LowRankDynamicWeightConfig)
    mixture_type = MixtureOfExpertsConfig().registry_owner()
    mixtures = [
        module for module in recurrent.modules() if isinstance(module, mixture_type)
    ]
    assert mixtures and any(mix.soft_router is not None for mix in mixtures) == soft_moe
    for mixture in mixtures:
        expert = mixture.expert_modules[0].layers[0].model
        layers = [expert]
        if mixture.soft_router is not None:
            layers.extend(
                layer.model for layer in mixture.soft_router.router.model.layers
            )
        for layer in layers:
            assert type(scorer) is type(layer)
            assert scorer.adaptive_cfg == layer.adaptive_cfg
            assert set(scorer.parameters()).isdisjoint(layer.parameters())
    logits, _ = split(classifier(torch.randn(2, 1, 28, 28)))
    logits.square().sum().backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in scorer.adaptive_behaviour.parameters()
    )
