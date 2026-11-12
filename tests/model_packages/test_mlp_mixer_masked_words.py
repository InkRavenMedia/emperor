"""Standalone MLP-Mixer on BERT-002's masked-word task and feature composition.

The fixture copies BERT-002-C01's normalized overrides on 2026-11-11. Its
historical Trainer timer is configuration evidence only; these tests never fit.
"""

import copy
import io
import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from emperor.attention import MixerAttentionConfig
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
)
from emperor.datasets.text.bert_pretraining import (
    BertMaskedWordBatch,
    WikiText103BertHierarchical,
)
from emperor.experiments import ExperimentTask
from emperor.experts import MixtureOfExpertsModelConfig
from emperor.layers import LayerStackConfig, RecurrentLayerConfig
from emperor.linears import LinearLayerConfig
from model_runtime.task_behavior import experiment_task_behavior
from models.catalog import model_package
from models.cli_selection import resolve_cli_selection
from models.experiment_cli_parser import get_experiment_parser


def reference_overrides():
    return json.loads(
        (Path(__file__).parent / "fixtures/bert_002_runtime_overrides.json").read_text()
    )


def mixer_overrides():
    values = reference_overrides()
    for key, expected in {
        "attn_grouping_input_order": "SEQUENCE_FIRST",
        "attn_num_heads": 4,
        "expert_attention_use_kv_expert_models_flag": False,
        "contextual_embedding_config": None,
        "hierarchical_embedding_flag": True,
        "mlm_decoder_weight_tying_flag": False,
        "router_diagonal_option_flag": False,
    }.items():
        assert values.pop(key) == expected
    for old, new in {
        "attention_soft_moe_flag": "token_mixer_soft_moe_flag",
        "attn_source_memory_chunk_size": "token_mixer_source_memory_chunk_size",
        "attn_source_memory_flag": "token_mixer_source_memory_flag",
        "attn_source_memory_grouping_method": "token_mixer_source_memory_grouping_method",
        "attn_source_memory_model_config": "token_mixer_source_memory_model_config",
        "ff_stack_hidden_dim": "channel_mixer_stack_hidden_dim",
    }.items():
        values[new] = values.pop(old)
    for key in (
        "chunk_size",
        "grouping_input_order",
        "grouping_method",
        "grouping_scope",
        "grouping_sequence_length",
    ):
        values["channel_mixer_" + key] = values.pop(key)
    # Spell out the reference's defaults where the image Mixer's defaults differ.
    values.update(
        layer_norm_position="AFTER",
        token_mixer_num_layers=1,
        token_mixer_stack_hidden_dim=32,
        token_mixer_stack_layer_norm_position="AFTER",
        token_mixer_stack_dropout_probability=0.1,
        token_mixer_stack_apply_output_postprocessing_flag=True,
        channel_mixer_num_layers=1,
        channel_mixer_stack_layer_norm_position="DISABLED",
        channel_mixer_stack_dropout_probability=0.0,
        channel_mixer_stack_apply_output_postprocessing_flag=False,
        expert_stack_num_layers=2,
        expert_stack_apply_output_postprocessing_flag=True,
        submodule_stack_num_layers=2,
        submodule_stack_layer_norm_position="AFTER",
        trainer_max_time=None,
    )
    return values


def configuration(**overrides):
    package = model_package("mlp_mixer/expert_linear_adaptive")
    values = {**mixer_overrides(), **overrides}
    values = {
        key: package.runtime_defaults_spec.parse_value(
            key.upper(), value if isinstance(value, str) else json.dumps(value)
        )
        for key, value in values.items()
    }
    return package.build_configuration(
        dataset=WikiText103BertHierarchical, config_overrides=values
    )


def attention_config(config):
    return config.experiment_config.encoder_config.block_config.layer_config.layer_model_config.attention_config


def batch(*, max_token_bytes=64, mlm_probability=1.0):
    return BertMaskedWordBatch.collate(
        [(("Café", "runs"), ("a", "dog"), 0), (("one",), ("two",), 1)],
        sequence_length=100,
        max_token_bytes=max_token_bytes,
        mlm_probability=mlm_probability,
    )


def test_reference_configuration_has_separate_mixer_backbone_and_matching_text_boundary():
    target = configuration()
    package = model_package("bert/expert_linear_adaptive")
    reference = {
        key: package.runtime_defaults_spec.parse_value(
            key.upper(), value if isinstance(value, str) else json.dumps(value)
        )
        for key, value in reference_overrides().items()
    }
    baseline = package.build_configuration(
        dataset=WikiText103BertHierarchical, config_overrides=reference
    )
    mixer = attention_config(target)
    assert isinstance(mixer, MixerAttentionConfig)
    encoder = target.experiment_config.encoder_config
    assert encoder.halting_config.min_steps == 2
    assert encoder.halting_config.ponder_cost_weight == 0.01
    assert encoder.max_steps == 10
    assert encoder.initial_iterations == 2
    assert encoder.forward_calls_before_iteration_increment == 20_000
    assert mixer.source_memory_config.grouping_config.chunk_size == 5
    assert asdict(target.experiment_config.embedding_config) == asdict(
        baseline.experiment_config.boundary_config.embedding_options.hierarchical_embedding_config
    )
    assert asdict(target.experiment_config.masked_word_decoder_config) == asdict(
        baseline.experiment_config.masked_word_decoder_config
    )
    assert asdict(target.experiment_config.positional_embedding_config) == asdict(
        baseline.experiment_config.positional_embedding_config
    )
    assert (
        target.experiment_config.embedding_dropout_probability
        == baseline.experiment_config.boundary_config.embedding_options.dropout_probability
    )
    reference_encoder = baseline.experiment_config.encoder_config
    reference_attention = attention_config(baseline)
    assert asdict(mixer.source_memory_config) == asdict(
        reference_attention.source_memory_config
    )
    assert isinstance(mixer.mixing_model_config, MixtureOfExpertsModelConfig)
    assert mixer.mixing_model_config.input_dim == target.sequence_length
    assert mixer.mixing_model_config.output_dim == target.sequence_length
    encoder.halting_config.input_dim = reference_encoder.halting_config.input_dim
    target_ff = encoder.block_config.layer_config.layer_model_config.feed_forward_config.stack_config
    reference_ff = reference_encoder.block_config.layer_config.layer_model_config.feed_forward_config.stack_config
    assert type(target_ff.stack_config) is type(reference_ff.stack_config)
    target_ff.stack_config.layer_config.layer_model_config.input_dim = target.hidden_dim
    target_ff.stack_config.layer_config.layer_model_config.output_dim = (
        target.hidden_dim
    )
    encoder.block_config.layer_config.layer_model_config.attention_config = (
        reference_attention
    )
    # Entire non-Mixer backbone, including feed-forward experts, residuals,
    # recurrence, schedules and halting, matches the reference configuration.
    assert asdict(encoder) == asdict(reference_encoder)


@pytest.mark.training
def test_reference_masked_word_path_gradients_optimizer_and_checkpoint():
    torch.manual_seed(61)
    package = model_package("mlp_mixer/expert_linear_adaptive")
    config = configuration()
    model = package.build_model(config).train()
    mixer = next(
        m
        for m in model.modules()
        if isinstance(getattr(m, "cfg", None), MixerAttentionConfig)
    )
    assert mixer.mixing_sequence_length == 120
    assert mixer.mixing_model.uses_soft_moe
    adaptive = [
        m
        for m in mixer.mixing_model.modules()
        if isinstance(getattr(m, "cfg", None), AdaptiveLinearLayerConfig)
    ]
    assert adaptive
    for layer in adaptive:
        assert layer.adaptive_behaviour.grouping_config is None
    data = batch()
    output, sentence_logits, auxiliary = model(data)
    loss = F.cross_entropy(output.logits, data.byte_labels)
    loss = (
        loss
        + model.nsp_loss_fn(sentence_logits, data.next_sentence_labels)
        + auxiliary.sum()
    )
    assert torch.isfinite(loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for module in (
        mixer.mixing_model,
        mixer.source_memory,
        model.word_decoder,
    ):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    # Both adaptive generators (including the Soft MoE scoring layers) train.
    for layer in adaptive:
        for generator in (
            layer.adaptive_behaviour.weight_model,
            layer.adaptive_behaviour.bias_model,
        ):
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in generator.parameters()
            )
    for mixture in (
        m
        for m in model.modules()
        if isinstance(getattr(m, "cfg", None), MixtureOfExpertsModelConfig)
    ):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in mixture.parameters()
        )
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()
    model.eval()
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = package.build_model(config).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    assert (
        model.transformer.recurrent_iteration_schedule.snapshot()
        == restored.transformer.recurrent_iteration_schedule.snapshot()
    )
    with torch.no_grad():
        expected = model(data)
        actual = restored(data)
    torch.testing.assert_close(actual[0].logits, expected[0].logits)
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2])


def test_soft_token_mixing_padding_and_samples_remain_independent():
    config = copy.deepcopy(
        attention_config(configuration(stack_dropout_probability=0.0))
    )
    model = config.build().eval()
    tokens = torch.randn(2, 100, 32, requires_grad=True)
    padding = torch.arange(100).unsqueeze(0) >= torch.tensor([[7], [11]])
    observed = []
    handle = model.mixing_model.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].hidden.detach().clone())
    )
    output, _, _ = model(tokens, tokens, tokens, padding)
    handle.remove()
    assert len(observed) == 1
    assert observed[0].shape == (2, 32, 120)
    masked_tokens = tokens.masked_fill(padding.unsqueeze(-1), 0)
    torch.testing.assert_close(observed[0][..., :100], masked_tokens.transpose(1, 2))
    poisoned = tokens.detach().masked_fill(padding.unsqueeze(-1), float("nan"))
    poisoned[1, :11] += 1000
    changed, _, _ = model(poisoned, poisoned, poisoned, padding)
    torch.testing.assert_close(changed[0], output[0])
    assert (changed[padding] == 0).all()
    single = tokens[:1]
    alone, _, _ = model(single, single, single, padding[:1])
    torch.testing.assert_close(alone, output[:1])
    short = single[:, :7]
    short_output, _, _ = model(short, short, short)
    torch.testing.assert_close(short_output, output[:1, :7])
    empty, _, empty_loss = model(single, single, single, torch.ones_like(padding[:1]))
    assert torch.isfinite(empty).all() and (empty == 0).all()
    assert empty_loss is None or torch.isfinite(empty_loss).all()
    output.square().sum().backward()
    assert torch.isfinite(tokens.grad).all()
    assert (tokens.grad[padding] == 0).all()


@pytest.mark.parametrize("soft_moe", [True, False])
def test_text_uses_the_existing_token_mixer_with_optional_soft_moe(soft_moe):
    config = attention_config(configuration(token_mixer_soft_moe_flag=soft_moe))
    model = config.build().eval()
    assert set(dict(model.named_children())) == {"mixing_model", "source_memory"}
    assert model.mixing_model.uses_soft_moe is soft_moe
    tokens = torch.randn(2, 7, 32)
    output, weights, loss = model(tokens, tokens, tokens)
    assert output.shape == tokens.shape and weights is None
    assert torch.isfinite(output).all()
    assert loss is None or torch.isfinite(loss).all()


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"hierarchical_byte_embedding_dim": 3}, "divisible"),
        ({"token_mixer_source_memory_chunk_size": 0}, "positive integer"),
        (
            {"token_mixer_source_memory_test_time_training_num_inner_steps": 1},
            "test-time training",
        ),
        ({"sequence_expansion_flag": True}, "sequence expansion"),
    ],
)
def test_unsupported_combinations_fail_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        configuration(**overrides)


def test_normal_cli_resolves_mixer_without_executing_an_experiment():
    package = model_package("mlp_mixer/expert_linear_adaptive")
    arguments = [
        "--experiment-task",
        "bert-pretraining",
        "--preset",
        "baseline",
        "--datasets",
        "wiki-text103-bert-hierarchical",
        "--config",
    ]
    for key, value in mixer_overrides().items():
        arguments.extend(
            (
                "--" + key.replace("_", "-"),
                value if isinstance(value, str) else json.dumps(value),
            )
        )
    parsed = get_experiment_parser(package).parse_args(arguments)
    selection = resolve_cli_selection(
        parsed, package, package.preset_type, build_monitor_callbacks=False
    )
    config = package.build_configuration(
        dataset=WikiText103BertHierarchical, config_overrides=selection.config_overrides
    )
    assert isinstance(attention_config(config), MixerAttentionConfig)


@pytest.mark.training
@pytest.mark.parametrize("mlm_probability", [0.0, 1.0])
def test_shared_task_data_contract_and_step_metrics(mlm_probability, monkeypatch):
    config = configuration(hierarchical_max_token_bytes=8)
    behavior = experiment_task_behavior(ExperimentTask.BERT_PRETRAINING)
    behavior.validate_dataset(WikiText103BertHierarchical, config)
    assert behavior.dataset_constructor_kwargs(config) == {
        "batch_size": 16,
        "sequence_length": 100,
        "max_token_bytes": 8,
    }
    (synthetic,) = behavior.synthetic_inputs(WikiText103BertHierarchical, config)
    assert isinstance(synthetic, BertMaskedWordBatch)
    model = model_package("mlp_mixer/expert_linear_adaptive").build_model(config)
    metrics = {}
    monkeypatch.setattr(
        model, "log_dict", lambda values, **kwargs: metrics.update(values)
    )
    monkeypatch.setattr(
        model, "log", lambda key, value, **kwargs: metrics.update({key: value})
    )
    data = batch(max_token_bytes=8, mlm_probability=mlm_probability)
    train_loss = model.training_step(data, 0)
    assert torch.isfinite(train_loss)
    train_loss.backward()
    model.eval()
    with torch.no_grad():
        validation_loss = model.validation_step(data, 0)
    assert torch.isfinite(validation_loss)
    assert "validation/mlm/exact_word_accuracy" in metrics
    for stage in ("train", "validation"):
        for name in (
            "loss",
            "mlm/byte_reconstruction_loss",
            "nsp/loss",
            "nsp/accuracy",
            "auxiliary/loss",
            "mlm/oversized_words",
        ):
            assert f"{stage}/{name}" in metrics
    if not mlm_probability:
        assert metrics["train/mlm/byte_reconstruction_loss"] == 0


def test_text_mixer_build_does_not_import_a_bert_implementation():
    code = """
import sys
from models.catalog import model_package
from emperor.datasets.text.bert_pretraining import WikiText103BertHierarchical
catalog_modules = set(sys.modules)
p = model_package("mlp_mixer/expert_linear_adaptive")
c = p.build_configuration(dataset=WikiText103BertHierarchical, config_overrides={
    "stack_num_layers": 1, "num_experts": 2, "hidden_dim": 4,
    "hierarchical_byte_embedding_dim": 4, "hierarchical_byte_num_heads": 1,
})
m = p.build_model(c)
assert type(m).__module__.startswith("models.mlp_mixer.expert_linear_adaptive.")
assert not any(n.startswith("models.bert.") for n in set(sys.modules) - catalog_modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_documented_command_matches_the_reference_model_options():
    root = Path(__file__).resolve().parents[2]
    text = (
        root / "src/models/mlp_mixer/expert_linear_adaptive/masked-word.md"
    ).read_text()
    command = text.split("```bash\n", 1)[1].split("\n```", 1)[0]
    arguments = shlex.split(command.replace("\\\n", ""))
    assert arguments[:4] == ["mise", "run", "experiment", "--"]
    arguments = arguments[4:]
    for selector in ("--model-type", "--model"):
        position = arguments.index(selector)
        del arguments[position : position + 2]
    package = model_package("mlp_mixer/expert_linear_adaptive")
    parsed = get_experiment_parser(package).parse_args(arguments)
    selection = resolve_cli_selection(
        parsed, package, package.preset_type, build_monitor_callbacks=False
    )
    assert selection.config_overrides["trainer_max_time"] is None
    assert selection.config_overrides["num_epochs"] == 100
    config = package.build_configuration(
        dataset=WikiText103BertHierarchical, config_overrides=selection.config_overrides
    )
    assert asdict(config) == asdict(configuration())


@pytest.mark.parametrize(
    "option,config_type",
    [
        ("LINEAR", LinearLayerConfig),
        ("LINEAR_STACK", LayerStackConfig),
        ("RECURRENT_LINEAR", RecurrentLayerConfig),
    ],
)
def test_memory_generator_flat_flags_build_without_json(option, config_type):
    config = configuration(
        token_mixer_source_memory_model_config=None,
        token_mixer_source_memory_model_option=option,
        token_mixer_source_memory_model_bias_flag=False,
        token_mixer_source_memory_model_stack_num_layers=3,
        token_mixer_source_memory_model_stack_hidden_dim=16,
        token_mixer_source_memory_model_recurrent_max_steps=3,
    )
    memory_config = attention_config(config).source_memory_config
    generator = memory_config.model_config
    assert type(generator) is config_type
    if option == "LINEAR_STACK":
        assert generator.num_layers == 3 and generator.hidden_dim == 16
        linear = generator.layer_config.layer_model_config
    elif option == "RECURRENT_LINEAR":
        assert generator.max_steps == generator.initial_iterations == 3
        linear = generator.block_config.layer_model_config
    else:
        linear = generator
    assert linear.bias_flag is False
    memory = memory_config.build(type(memory_config)(embedding_dim=32))
    source = torch.randn(7, 2, 32, requires_grad=True)
    generated, _, _ = memory(source)
    assert generated.shape == (4, 32)
    generated.square().sum().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
