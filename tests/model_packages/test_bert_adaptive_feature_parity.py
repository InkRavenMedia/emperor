"""Adaptive BERT and GPT share reasoning controls and reusable embedding modes."""

import copy
from importlib import import_module

import pytest
import torch

from models.catalog import model_package
from models.cli_selection import resolve_cli_selection
from models.experiment_cli_parser import get_experiment_parser
from tests.model_packages.test_bert_hierarchical_soft_moe import (
    TOKEN_TEXTS,
    configuration,
    inputs,
)
from tests.unit.test_byte_contextual_embedding import contextual_config

ADAPTIVE_VARIANTS = ("linear_adaptive", "expert_linear_adaptive")
TASK_SPECIFIC_GPT_PREFIXES = (
    "HIERARCHICAL_",
    "CONTEXTUAL_",
    "LM_HEAD_",
    "DECODER_OUTPUT_",
)


def gpt_configuration(variant, **overrides):
    values = dict(
        batch_size=2,
        input_dim=len(TOKEN_TEXTS),
        output_dim=len(TOKEN_TEXTS),
        hidden_dim=8,
        sequence_length=4,
        stack_num_layers=1,
        attn_num_heads=2,
        ff_stack_hidden_dim=8,
        stack_dropout_probability=0.0,
        embedding_dropout_probability=0.0,
        adaptive_generator_stack_hidden_dim=8,
    )
    if "expert" in variant:
        values.update(
            num_experts=3,
            top_k=2,
            capacity_factor=0.0,
            expert_stack_hidden_dim=8,
            router_stack_hidden_dim=8,
        )
    values.update(overrides)
    package = model_package("gpt/" + variant)
    return package.build_configuration(config_overrides=values)


@pytest.mark.parametrize("variant", ADAPTIVE_VARIANTS)
def test_every_shared_gpt_adaptive_control_is_available_in_bert(variant):
    gpt_module = import_module("models.gpt." + variant + ".config")
    bert_module = import_module("models.bert." + variant + ".config")
    gpt_settings = {
        name
        for name in gpt_module.__annotations__
        if name.isupper() and not name.startswith(TASK_SPECIFIC_GPT_PREFIXES)
    }
    assert gpt_settings <= set(bert_module.__annotations__)
    for name in gpt_settings:
        assert bert_module.__annotations__[name] == gpt_module.__annotations__[name]


@pytest.mark.parametrize("variant", ("expert_linear_adaptive",))
def test_bert_cli_exposes_reasoning_growth_and_halting_controls(variant):
    package = model_package("bert/" + variant)
    parser = get_experiment_parser(package)
    arguments = [
        "--preset",
        "baseline",
        "--recurrent-flag",
        "true",
        "--recurrent-max-steps",
        "5",
        "--recurrent-initial-iterations",
        "2",
        "--recurrent-forward-calls-before-iteration-increment",
        "20000",
        "--recurrent-smooth-iteration-growth-flag",
        "true",
        "--recurrent-no-gradient-transition-count",
        "0",
    ]
    if "expert" in variant:
        arguments.extend(
            [
                "--recurrent-stack-halting-flag",
                "true",
                "--recurrent-min-steps",
                "2",
                "--recurrent-ponder-cost-weight",
                "0.25",
            ]
        )
    parsed_arguments = parser.parse_args(arguments)
    preset_module = import_module("models.bert." + variant + ".presets")
    selection = resolve_cli_selection(
        parsed_arguments, package, preset_module.ExperimentPreset
    )
    model_config = package.build_configuration(
        preset=selection.preset, config_overrides=selection.config_overrides
    )
    recurrent_config = model_config.experiment_config.encoder_config
    assert recurrent_config.initial_iterations == 2
    assert recurrent_config.max_steps == 5
    assert recurrent_config.forward_calls_before_iteration_increment == 20_000
    assert recurrent_config.smooth_iteration_growth_flag is True
    assert recurrent_config.no_gradient_transition_count == 0
    if "expert" in variant:
        assert recurrent_config.halting_config.min_steps == 2
        assert recurrent_config.halting_config.ponder_cost_weight == 0.25


def test_minimum_steps_and_ponder_weight_reach_bert_recurrent_halting():
    options = dict(
        recurrent_flag=True,
        recurrent_max_steps=5,
        recurrent_initial_iterations=2,
        recurrent_stack_halting_flag=True,
        recurrent_min_steps=2,
        recurrent_ponder_cost_weight=0.25,
    )
    bert_config = configuration(
        "bert/expert_linear_adaptive", hierarchical=False, soft=False, **options
    )
    gpt_config = gpt_configuration("expert_linear_adaptive", **options)
    bert_halting = bert_config.experiment_config.encoder_config.halting_config
    gpt_halting = gpt_config.experiment_config.decoder_config.halting_config
    assert bert_halting.min_steps == gpt_halting.min_steps == 2
    assert bert_halting.ponder_cost_weight == gpt_halting.ponder_cost_weight == 0.25
    model = model_package("bert/expert_linear_adaptive").build_model(bert_config).eval()
    outputs = model(*inputs())
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"recurrent_min_steps": True},
        {"recurrent_min_steps": 0},
        {"recurrent_min_steps": 6},
        {"recurrent_ponder_cost_weight": True},
        {"recurrent_ponder_cost_weight": -0.1},
        {"recurrent_ponder_cost_weight": float("nan")},
        {"recurrent_min_steps": 2, "recurrent_flag": False},
        {"recurrent_ponder_cost_weight": 0.25, "recurrent_stack_halting_flag": False},
    ],
)
def test_bert_halting_options_reject_invalid_or_inactive_settings(overrides):
    options = dict(
        recurrent_flag=True, recurrent_max_steps=5, recurrent_stack_halting_flag=True
    )
    options.update(overrides)
    with pytest.raises((ValueError, TypeError), match="recurrent_"):
        configuration(
            "bert/expert_linear_adaptive", hierarchical=False, soft=False, **options
        )


@pytest.mark.parametrize("variant", ("linear", "expert_linear", *ADAPTIVE_VARIANTS))
def test_contextual_byte_embedding_composes_with_bert_and_soft_moe(variant):
    contextual_embedding_config = contextual_config(
        hidden_dim=8, maximum_sequence_length=5
    )
    bert_config = configuration(
        "bert/" + variant,
        hierarchical=False,
        contextual_embedding_config=contextual_embedding_config,
        mlm_decoder_weight_tying_flag=False,
    )
    model = model_package("bert/" + variant).build_model(bert_config).eval()
    model.set_token_vocabulary(TOKEN_TEXTS)
    position_calls = []
    model.positional_embedding.register_forward_hook(
        lambda module, arguments, output: position_calls.append(output)
    )
    token_ids, attention_mask, token_type_ids = inputs()
    mlm_logits, nsp_logits, auxiliary_loss = model(
        token_ids, attention_mask, token_type_ids
    )
    assert not position_calls
    changed_ids = token_ids.clone()
    changed_ids[attention_mask == 0] = 11
    changed_mlm, changed_nsp, _ = model(changed_ids, attention_mask, token_type_ids)
    torch.testing.assert_close(
        mlm_logits[attention_mask.bool()], changed_mlm[attention_mask.bool()]
    )
    torch.testing.assert_close(nsp_logits, changed_nsp)
    total_loss = (
        mlm_logits.square().mean() + nsp_logits.square().mean() + auxiliary_loss.sum()
    )
    total_loss.backward()
    assert torch.isfinite(total_loss)
    assert model.token_embedding.embedding.gamma.grad is not None
    restored_model = (
        model_package("bert/" + variant).build_model(copy.deepcopy(bert_config)).eval()
    )
    restored_model.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(model(*inputs())[0], restored_model(*inputs())[0])


def test_contextual_and_hierarchical_modes_are_mutually_exclusive():
    contextual_embedding_config = contextual_config(
        hidden_dim=8, maximum_sequence_length=5
    )
    bert_config = configuration(
        "bert/expert_linear_adaptive",
        contextual_embedding_config=contextual_embedding_config,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        model_package("bert/expert_linear_adaptive").build_model(bert_config)
