"""Sequence Expansion CLI binding, execution, and task compatibility."""

from copy import deepcopy

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdditiveDynamicBiasConfig,
    WeightDecayScheduleOptions,
)
from emperor.datasets.image.classification import Mnist
from emperor.embedding.absolute import TextSinusoidalPositionalEmbeddingConfig
from emperor.experiments import ExperimentTask
from emperor.linears import LinearLayerConfig
from emperor.transformer import (
    SequenceExpansionConfig,
    TransformerDecoderBlockLayer,
    TransformerEncoderBlockLayer,
)
from emperor.transformer.sequence_expansion import SequenceExpansion
from model_runtime.inspection.schema import configuration_schema
from model_runtime.packages.configuration import serialize_config_value
from model_runtime.task_behavior import experiment_task_behavior
from models._sequence_expansion import SEQUENCE_EXPANSION_FIELDS
from models.catalog import model_package
from models.config_overrides import print_config_options
from models.experiment_cli_parser import get_experiment_parser
from tests.model_packages.test_grouped_attention_memory import (
    PACKAGES as ATTENTION_PACKAGES,
)
from tests.model_packages.test_grouped_attention_memory import (
    build_config,
    parse_overrides,
)
from tests.model_packages.test_grouped_attention_memory import (
    execute as execute_attention,
)
from tests.model_packages.test_mlp_mixer_source_memory import (
    PACKAGES as MIXER_PACKAGES,
)
from tests.model_packages.test_mlp_mixer_source_memory import (
    configuration as mixer_configuration,
)
from unit.test_transformer import linear_stack

PACKAGES = (*ATTENTION_PACKAGES, *MIXER_PACKAGES)
LINEAR_PACKAGES = [name for name in PACKAGES if name.endswith("/linear")]
ENABLE_ARGUMENTS = (
    "--sequence-expansion-flag",
    "true",
    "--sequence-expansion-max-new-tokens",
    "2",
    "--sequence-expansion-model-config",
    '{"type":"LinearLayerConfig","bias_flag":true}',
)


@pytest.fixture(autouse=True)
def isolated_random_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(145)
        yield


def configuration(package_name, overrides):
    if package_name in MIXER_PACKAGES:
        return mixer_configuration(package_name, **overrides)
    return build_config(package_name, {"attn_source_memory_flag": False, **overrides})


def execute(model, package_name):
    if package_name in MIXER_PACKAGES:
        return model(torch.randn(2, 1, 8, 8))
    return execute_attention(model, package_name)


def expansions(model):
    return [
        module for module in model.modules() if isinstance(module, SequenceExpansion)
    ]


@pytest.mark.parametrize("package_name", PACKAGES)
def test_flags_are_discoverable_and_disabled_preserves_initialization(
    package_name, capsys
):
    package = model_package(package_name)
    prefixes = (
        ("", "ENCODER_", "DECODER_")
        if package_name.startswith("transformer/")
        else ("",)
    )
    keys = {
        prefix + name.upper()
        for prefix in prefixes
        for name in SEQUENCE_EXPANSION_FIELDS
    }
    schema = {field.key: field for field in configuration_schema(package).fields}
    assert keys <= schema.keys()
    assert all(schema[key].description for key in keys)
    for key in sorted(key for key in keys if key.endswith("_CONFIG")):
        assert schema[key].value_type == "json", key
        assert not schema[key].choices, key
    help_text = get_experiment_parser(package).format_help()
    print_config_options(package_name)
    listed = capsys.readouterr().out
    for key in keys:
        flag = "--" + key.lower().replace("_", "-")
        assert flag in help_text and flag in listed
    assert not any(
        "FIXED_NEW_TOKENS" in key or "MAX_SEQUENCE_LENGTH" in key for key in keys
    )

    baseline = configuration(package_name, {})
    disabled = configuration(
        package_name,
        parse_overrides(
            package_name,
            "--sequence-expansion-flag",
            "false",
            "--sequence-expansion-config",
            '{"type":"SequenceExpansionConfig"}',
            "--sequence-expansion-positional-embedding-config",
            "None",
        ),
    )
    torch.manual_seed(801)
    expected = package.build_model(baseline).eval()
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(801)
    actual = package.build_model(disabled).eval()
    assert not expansions(actual)
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    assert expected.state_dict().keys() == actual.state_dict().keys()
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
    torch.manual_seed(802)
    expected_outputs = execute(expected, package_name)
    torch.manual_seed(802)
    actual_outputs = execute(actual, package_name)
    torch.testing.assert_close(actual_outputs, expected_outputs, rtol=0, atol=0)


@pytest.mark.parametrize("package_name", PACKAGES)
@pytest.mark.parametrize("recurrent", [False, True])
def test_cli_reaches_blocks_and_backpropagates_through_repeated_expansion(
    package_name, recurrent
):
    overrides = parse_overrides(package_name, *ENABLE_ARGUMENTS)
    overrides.update(
        recurrent_flag=recurrent, recurrent_max_steps=3, recurrent_initial_iterations=3
    )
    if package_name.startswith("transformer/"):
        overrides.update(encoder_num_layers=2, decoder_num_layers=2)
    else:
        overrides["stack_num_layers"] = 2
    package = model_package(package_name)
    model = package.build_model(configuration(package_name, overrides)).eval()
    modules = expansions(model)
    assert modules
    events = []
    hooks = []
    for module in modules:
        assert module.embedding_dim == 8 and module.max_new_tokens == 2
        with torch.no_grad():
            module.count_predictor.weight_params.zero_()
            module.count_predictor.bias_params.copy_(torch.tensor([0.0, 2.0]))
        hooks.append(
            module.register_forward_hook(lambda _, args, result: events.append(result))
        )
    outputs = execute(model, package_name)
    for hook in hooks:
        hook.remove()
    assert len(events) == len(modules) * (3 if recurrent else 1)
    for event in events:
        assert event.expansion.selected_counts.tolist() == [2, 2]
        assert not torch.any(
            event.key_padding_mask[:, :-1] & ~event.key_padding_mask[:, 1:]
        )
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    loss = sum(output.square().mean() for output in outputs)
    assert torch.isfinite(loss)
    loss.backward()
    for module in modules:
        assert module.count_predictor.weight_params.grad is not None
        assert torch.isfinite(module.count_predictor.weight_params.grad).all()
        assert module.new_position_vector.grad is not None


@pytest.mark.parametrize("package_name", PACKAGES)
def test_enabled_expansion_requires_count_and_predictor(package_name):
    for extra, message in (
        ({}, "max_new_tokens"),
        ({"sequence_expansion_max_new_tokens": 2}, "model_config"),
    ):
        with pytest.raises(ValueError, match=message):
            configuration(package_name, {"sequence_expansion_flag": True, **extra})


@pytest.mark.parametrize("package_name", LINEAR_PACKAGES)
def test_structured_config_round_trip_overrides_and_copying(package_name):
    supplied = SequenceExpansionConfig(
        embedding_dim=999,
        max_new_tokens=1,
        model_config=LinearLayerConfig(bias_flag=False),
        positional_embedding_config=TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=1,
            auto_expand_flag=True,
        ),
    )
    original = deepcopy(supplied)
    overrides = parse_overrides(
        package_name,
        "--sequence-expansion-flag",
        "true",
        "--sequence-expansion-config",
        serialize_config_value(supplied),
        "--sequence-expansion-max-new-tokens",
        "3",
        "--sequence-expansion-model-config",
        '{"type":"LinearLayerConfig","bias_flag":true}',
    )
    parsed = overrides["sequence_expansion_config"]
    assert parsed == original
    model = model_package(package_name).build_model(
        configuration(package_name, overrides)
    )
    for module in expansions(model):
        assert module.max_new_tokens == 3 and module.embedding_dim == 8
        assert module.count_predictor.bias_params is not None
        assert module.position_embedding.cfg.embedding_dim == 8
        assert module.position_embedding.cfg.auto_expand_flag
    assert parsed == original and supplied == original


@pytest.mark.parametrize(
    "variant", ["linear", "linear_adaptive", "expert_linear", "expert_linear_adaptive"]
)
def test_transformer_path_overrides_and_explicit_false(variant):
    name = f"transformer/{variant}"
    branch_config = SequenceExpansionConfig(
        max_new_tokens=4, model_config=LinearLayerConfig(bias_flag=True)
    )
    overrides = parse_overrides(
        name,
        *ENABLE_ARGUMENTS,
        "--encoder-sequence-expansion-max-new-tokens",
        "None",
        "--decoder-sequence-expansion-config",
        serialize_config_value(branch_config),
    )
    runtime = model_package(name).bind_runtime_defaults(overrides)
    from models._sequence_expansion import build_sequence_expansion_config

    encoder = build_sequence_expansion_config(runtime.encoder_options, 8)
    decoder = build_sequence_expansion_config(runtime.decoder_options, 8)
    assert encoder.max_new_tokens == 2 and decoder.max_new_tokens == 4
    for disabled_side in ("encoder", "decoder"):
        runtime = model_package(name).bind_runtime_defaults(
            {**overrides, f"{disabled_side}_sequence_expansion_flag": False}
        )
        assert (
            build_sequence_expansion_config(
                getattr(runtime, f"{disabled_side}_options"), 8
            )
            is None
        )
    if variant == "linear":
        model = model_package(name).build_model(configuration(name, overrides))
        blocks = [
            module
            for module in model.modules()
            if isinstance(
                module, (TransformerEncoderBlockLayer, TransformerDecoderBlockLayer)
            )
        ]
        assert [block.sequence_expansion.max_new_tokens for block in blocks] == [2, 4]


@pytest.mark.parametrize(
    "name", ["vit/linear", "mlp_mixer/linear", "mlp_mixer/expert_linear_adaptive"]
)
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
def test_predictor_json_accepts_linear_and_adaptive_stacks(name, adaptive, stacked):
    predictor = LinearLayerConfig(bias_flag=True)
    if adaptive:
        predictor = AdaptiveLinearLayerConfig(
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                bias_config=AdditiveDynamicBiasConfig(
                    decay_schedule=WeightDecayScheduleOptions.DISABLED,
                    decay_rate=0.0,
                    decay_warmup_batches=0,
                    model_config=linear_stack(8, 2),
                ),
            ),
        )
    if stacked:
        stack = linear_stack(8, 2, num_layers=2)
        stack.layer_config.layer_model_config = predictor
        predictor = stack
    overrides = parse_overrides(
        name,
        *ENABLE_ARGUMENTS,
        "--sequence-expansion-model-config",
        serialize_config_value(predictor),
        "--sequence-expansion-positional-embedding-config",
        '{"type":"TextLearnedPositionalEmbeddingConfig","num_embeddings":16}',
    )
    model = model_package(name).build_model(configuration(name, overrides))
    outputs = execute(model, name)
    loss = sum(output.square().mean() for output in outputs)
    loss.backward()
    predictor_model = expansions(model)[0].count_predictor
    assert any(parameter.grad is not None for parameter in predictor_model.parameters())
    if adaptive:
        assert any(
            "bias_model" in name and parameter.grad is not None
            for name, parameter in predictor_model.named_parameters()
        )


@pytest.mark.parametrize(
    "variant", ["linear", "linear_adaptive", "expert_linear", "expert_linear_adaptive"]
)
@pytest.mark.parametrize(
    "family,task",
    [
        ("bert", ExperimentTask.BERT_PRETRAINING),
        ("gpt", ExperimentTask.CAUSAL_LANGUAGE_MODELING),
        ("transformer", ExperimentTask.TEXT_TRANSLATION),
    ],
)
def test_text_tasks_align_supervision_and_generation_after_recurrent_expansion(
    family, task, variant
):
    name = f"{family}/{variant}"
    overrides = parse_overrides(name, *ENABLE_ARGUMENTS)
    overrides.update(
        recurrent_flag=True, recurrent_max_steps=2, recurrent_initial_iterations=2
    )
    config = configuration(name, overrides)
    behavior = experiment_task_behavior(task)
    behavior.validate_configuration(config)
    behavior.validate_dataset(object, config)
    model = model_package(name).build_model(config).eval()
    tokens = torch.tensor([[2, 0, 4, 0], [2, 5, 6, 3]])
    attention_mask = tokens.ne(0).long()
    hooks = []
    for module in expansions(model):
        with torch.no_grad():
            module.count_predictor.weight_params.zero_()
            module.count_predictor.bias_params.zero_()
        hooks.append(
            module.count_predictor.register_forward_hook(
                lambda _, args, logits: logits
                + logits.new_tensor([[2.0, 0.0], [0.0, 2.0]])
            )
        )
    if family == "bert":
        labels = tokens.masked_fill(tokens.eq(0), -100)
        batch = (
            tokens,
            labels,
            attention_mask,
            torch.zeros_like(tokens),
            torch.tensor([0, 1]),
        )
        full_logits = model(tokens, attention_mask, batch[3])[0]
        aligned_logits = model.forward_for_input_tokens(
            tokens, attention_mask, batch[3]
        )[0]
    elif family == "gpt":
        batch = (tokens, tokens.roll(-1, dims=1))
        full_logits = model(tokens, attention_mask)[0]
        aligned_logits = model(tokens, attention_mask, align_to_input=True)[0]
        generated = model.generate(tokens[:, :1], max_new_tokens=2)
        assert generated.shape == (2, 3)
        first_logits = model(tokens[:, :1], align_to_input=True)[0][:, -1]
        assert torch.equal(generated[:, 1], first_logits.argmax(-1))
    else:
        batch = (tokens, torch.tensor([[2, 4, 3, 0], [2, 6, 7, 3]]))
        full_logits = model(tokens, tokens)[0]
        aligned_logits = model.forward_for_input_tokens(tokens, tokens)[0]
        generated = model.generate(tokens, max_length=3)
        assert generated.shape == (2, 3)
        expected = model(tokens, torch.full((2, 1), 2), align_to_input=True)[0][
            :, -1
        ].argmax(-1)
        assert torch.equal(generated[:, 1], expected)
    assert full_logits.shape[1] == 8
    assert aligned_logits.shape[:2] == tokens.shape
    torch.testing.assert_close(aligned_logits[0, [0, 2]], full_logits[0, :2])
    torch.testing.assert_close(aligned_logits[1], full_logits[1, :4])
    loss = model._model_step(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("source_memory", [False, True])
def test_encoder_only_translation_preserves_source_masks_and_supervised_loss(
    source_memory,
):
    name = "transformer/linear"
    overrides = parse_overrides(
        name,
        *ENABLE_ARGUMENTS,
        "--decoder-sequence-expansion-flag",
        "false",
        "--recurrent-flag",
        "true",
        "--recurrent-max-steps",
        "3",
        "--recurrent-initial-iterations",
        "3",
    )
    overrides.update(
        attn_source_memory_flag=source_memory, attn_source_memory_chunk_size=2
    )
    config = configuration(name, overrides)
    experiment_task_behavior(ExperimentTask.TEXT_TRANSLATION).validate_configuration(
        config
    )
    model = model_package(name).build_model(config).eval()
    module = expansions(model)[0]
    with torch.no_grad():
        module.count_predictor.weight_params.zero_()
        module.count_predictor.bias_params.zero_()
    hook = module.count_predictor.register_forward_hook(
        lambda _, args, logits: logits + logits.new_tensor([[10.0, 0.0], [0.0, 10.0]])
    )
    source = torch.tensor([[2, 4, 0, 0], [2, 4, 5, 3]])
    hidden, mask, _ = model._encode(source)
    assert hidden.shape[1] == 10
    assert (~mask).sum(1).tolist() == [5, 10]
    assert mask[0].tolist() == [False] * 5 + [True] * 5
    loss = model._model_step((source, torch.tensor([[2, 5, 3, 0], [2, 6, 7, 3]])))
    hook.remove()
    assert torch.isfinite(loss)
    loss.backward()
    assert module.count_predictor.weight_params.grad is not None


def test_readme_vit_command_configuration_runs_offline():
    name = "vit/linear"
    overrides = parse_overrides(
        name,
        "--sequence-expansion-flag",
        "true",
        "--sequence-expansion-max-new-tokens",
        "3",
        "--sequence-expansion-model-config",
        '{"type":"LinearLayerConfig","bias_flag":true}',
    )
    package = model_package(name)
    config = package.build_configuration(dataset=Mnist, config_overrides=overrides)
    experiment_task_behavior(
        ExperimentTask.IMAGE_CLASSIFICATION
    ).validate_configuration(config)
    model = package.build_model(config).eval()
    output = model(torch.zeros(2, 1, 28, 28))
    assert output.shape == (2, 10)
    torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()


@pytest.mark.parametrize("maximum", [0, -1])
def test_nonpositive_per_call_counts_fail_during_configuration(maximum):
    overrides = parse_overrides(
        "vit/linear",
        *ENABLE_ARGUMENTS,
        "--sequence-expansion-max-new-tokens",
        str(maximum),
    )
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        configuration("vit/linear", overrides)


@pytest.mark.parametrize(
    "flag",
    [
        "--sequence-expansion-fixed-new-tokens",
        "--sequence-expansion-max-sequence-length",
    ],
)
def test_removed_limits_are_not_exposed(flag):
    with pytest.raises(SystemExit):
        parse_overrides("vit/linear", flag, "2")


@pytest.mark.parametrize(
    "variant",
    ["TinyRecursiveModelRecurrentConfig", "HierarchicalReasoningModelRecurrentConfig"],
)
def test_nested_recurrent_variants_have_capacity_for_all_expansion_calls(variant):
    name = "transformer/linear"
    overrides = parse_overrides(
        name,
        *ENABLE_ARGUMENTS,
        "--recurrent-flag",
        "true",
        "--recurrent-composition-option",
        variant,
        "--recurrent-initial-iterations",
        "2",
        "--recurrent-answer-update-count",
        "2",
        "--recurrent-latent-updates-per-answer-update",
        "2",
        "--recurrent-high-cycles",
        "2",
        "--recurrent-low-cycles",
        "2",
    )
    model = model_package(name).build_model(configuration(name, overrides))
    for module in expansions(model):
        with torch.no_grad():
            module.count_predictor.weight_params.zero_()
            module.count_predictor.bias_params.copy_(torch.tensor([0.0, 2.0]))
    logits, auxiliary_loss = execute(model, name)
    assert logits.size(1) > 5
    loss = logits.square().mean() + auxiliary_loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        module.new_position_vector.grad is not None for module in expansions(model)
    )


@pytest.mark.parametrize("name", MIXER_PACKAGES)
@pytest.mark.parametrize("source_memory", [False, True])
def test_mixer_ragged_recurrence_masks_pooling_and_reloads(name, source_memory):
    overrides = parse_overrides(name, *ENABLE_ARGUMENTS)
    overrides.update(
        stack_num_layers=2,
        recurrent_flag=True,
        recurrent_max_steps=3,
        recurrent_initial_iterations=3,
        token_mixer_source_memory_flag=source_memory,
    )
    config = configuration(name, overrides)
    package = model_package(name)
    model = package.build_model(config).eval()
    hooks = []

    def choose_ragged_counts(_, args, logits):
        return logits + logits.new_tensor([[2.0, 0.0], [0.0, 2.0]])

    for module in expansions(model):
        with torch.no_grad():
            module.count_predictor.weight_params.zero_()
            module.count_predictor.bias_params.zero_()
        hooks.append(module.count_predictor.register_forward_hook(choose_ragged_counts))
    states = []
    hooks.append(
        model.transformer.register_forward_hook(
            lambda _, args, state: states.append(state)
        )
    )
    images = torch.randn(2, 1, 8, 8)
    outputs = model(images)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    state = states[0]
    assert state.hidden.shape == (2, 16, 8)
    assert (~state.key_padding_mask).sum(1).tolist() == [10, 16]
    assert len(state.sequence_expansions) == 6
    assert state.key_padding_mask[0].tolist() == [False] * 10 + [True] * 6
    logits.square().mean().backward()
    for module in expansions(model):
        gradient = module.count_predictor.weight_params.grad
        assert (
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum() > 0
        )

    def poison_padding(_, args, state):
        padding = state.key_padding_mask.unsqueeze(-1)
        state.hidden = torch.where(
            padding, torch.randn_like(state.hidden) * 1000, state.hidden
        )
        return state

    poison = model.transformer.register_forward_hook(poison_padding)
    poisoned = model(images)
    poisoned_logits = poisoned[0] if isinstance(poisoned, tuple) else poisoned
    torch.testing.assert_close(poisoned_logits, logits)
    poison.remove()

    restored = package.build_model(config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    for module in expansions(restored):
        hooks.append(module.count_predictor.register_forward_hook(choose_ragged_counts))
    restored_output = restored(images)
    torch.testing.assert_close(restored_output, outputs)
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("name", [name for name in PACKAGES if "adaptive" in name])
def test_expansion_with_active_adaptive_backbone_parameters(name):
    overrides = parse_overrides(name, *ENABLE_ARGUMENTS)
    overrides.update(
        bias_option_flag=True,
        bias_option=AdditiveDynamicBiasConfig,
        recurrent_flag=True,
        recurrent_max_steps=2,
        recurrent_initial_iterations=2,
    )
    model = model_package(name).build_model(configuration(name, overrides)).eval()
    assert any(
        isinstance(module, AdaptiveParameterAugmentationConfig().registry_owner())
        for module in model.modules()
    )
    outputs = execute(model, name)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    loss = sum(output.square().mean() for output in outputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        "bias_model" in parameter_name
        and parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter_name, parameter in model.named_parameters()
    )


@pytest.mark.parametrize(
    "name", [*MIXER_PACKAGES, "bert/expert_linear", "gpt/expert_linear"]
)
def test_expansion_with_soft_expert_routing(name):
    overrides = parse_overrides(name, *ENABLE_ARGUMENTS)
    overrides.update(
        soft_moe_flag=True, soft_moe_num_experts=2, soft_moe_slots_per_expert=2
    )
    prefix = "token_mixer" if name in MIXER_PACKAGES else "attention"
    overrides.update(
        {
            f"{prefix}_soft_moe_flag": True,
            f"{prefix}_soft_moe_num_experts": 2,
            f"{prefix}_soft_moe_slots_per_expert": 2,
        }
    )
    model = model_package(name).build_model(configuration(name, overrides)).eval()
    outputs = execute(model, name)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    loss = sum(output.square().mean() for output in outputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        module.count_predictor.weight_params.grad is not None
        for module in expansions(model)
    )


@pytest.mark.parametrize(
    "variant", ["linear", "linear_adaptive", "expert_linear", "expert_linear_adaptive"]
)
def test_hierarchical_outputs_reject_expansion_before_execution(variant):
    from tests.model_packages.test_gpt_hierarchical_language_model import (
        configuration as hierarchical_configuration,
    )

    package, config = hierarchical_configuration(
        f"gpt/{variant}",
        sequence_expansion_flag=True,
        sequence_expansion_max_new_tokens=2,
        sequence_expansion_model_config=LinearLayerConfig(bias_flag=True),
    )
    behavior = experiment_task_behavior(ExperimentTask.CAUSAL_LANGUAGE_MODELING)
    with pytest.raises(ValueError, match="requires unchanged token positions"):
        behavior.validate_configuration(config)
    with pytest.raises(ValueError, match="requires unchanged token positions"):
        package.build_model(config)
