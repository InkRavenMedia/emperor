"""Adaptive BERT and GPT share reasoning controls and reusable embedding modes."""

import copy
from importlib import import_module

import pytest
import torch
import torch.nn.functional as F

from emperor.attention._base import MultiHeadAttentionAbstract
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    CombinedDynamicDiagonalConfig,
    DualModelDynamicWeightConfig,
    GeneratorDynamicBiasConfig,
    SumGroupingConfig,
)
from emperor.embedding.contextual import ByteContextualEmbeddingConfig
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
from emperor.halting import HaltingHiddenStateModeOptions, StickBreakingConfig
from emperor.layers import RecurrentLayerConfig
from model_runtime.inspection.schema import configuration_schema
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


@pytest.mark.training
def test_grouped_expert_bert_trains_with_halting_soft_feed_forward_and_source_memory():
    torch.manual_seed(25)
    package = model_package("bert/expert_linear_adaptive")
    model_config = configuration(
        "bert/expert_linear_adaptive",
        hierarchical=False,
        soft=True,
        sequence_length=100,
        recurrent_flag=True,
        recurrent_initial_iterations=2,
        recurrent_max_steps=3,
        recurrent_forward_calls_before_iteration_increment=20_000,
        recurrent_no_gradient_transition_count=0,
        recurrent_stack_halting_flag=True,
        recurrent_halting_option=StickBreakingConfig,
        recurrent_halting_hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
        recurrent_min_steps=2,
        recurrent_ponder_cost_weight=0.01,
        expert_attention_use_kv_expert_models_flag=False,
        attn_source_memory_flag=True,
        attn_source_memory_chunk_size=10,
        attn_grouping_scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
        attn_chunk_size=5,
        attn_grouping_sequence_length=100,
        attn_grouping_input_order=AdaptiveParameterInputOrderOptions.SEQUENCE_FIRST,
        attn_grouping_method=SumGroupingConfig,
        weight_option_flag=True,
        weight_option=DualModelDynamicWeightConfig,
        bias_option_flag=True,
        bias_option=GeneratorDynamicBiasConfig,
        diagonal_option_flag=True,
        diagonal_option=CombinedDynamicDiagonalConfig,
        mask_option_flag=False,
    )
    model = package.build_model(model_config).train()
    attention = next(
        module
        for module in model.transformer.modules()
        if isinstance(module, MultiHeadAttentionAbstract)
    )
    projections = (
        attention.projector.key_model,
        attention.projector.value_model,
    )
    generated_weight_shapes = []

    def record_weights(_module, arguments, output):
        assert arguments[1].shape == (40, 8)
        generated_weight_shapes.append(tuple(output.shape))

    handles = [
        layer.model.adaptive_behaviour.weight_model.register_forward_hook(
            record_weights
        )
        for projection in projections
        for layer in projection
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    try:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            token_ids = torch.randint(5, len(TOKEN_TEXTS), (2, 100))
            mlm_labels = torch.full_like(token_ids, -100)
            mlm_labels[:, ::5] = token_ids[:, ::5]
            token_ids[:, ::5] = 4
            attention_mask = torch.ones_like(token_ids)
            token_type_ids = torch.zeros_like(token_ids)
            token_type_ids[:, 50:] = 1
            mlm_logits, nsp_logits, auxiliary_loss = model(
                token_ids, attention_mask, token_type_ids
            )
            assert mlm_logits.shape == (2, 100, len(TOKEN_TEXTS))
            assert nsp_logits.shape == (2, 2)
            mlm_loss = F.cross_entropy(mlm_logits.transpose(1, 2), mlm_labels)
            nsp_loss = F.cross_entropy(nsp_logits, torch.tensor([0, 1]))
            loss = mlm_loss + nsp_loss + auxiliary_loss.sum()
            assert torch.isfinite(loss)
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            assert gradients and all(
                torch.isfinite(gradient).all() for gradient in gradients
            )
            for component in (
                attention.projector.source_memory,
                model.transformer.halting_model,
            ):
                assert any(
                    parameter.grad is not None and torch.count_nonzero(parameter.grad)
                    for parameter in component.parameters()
                )
            optimizer.step()
            assert all(
                torch.isfinite(parameter).all() for parameter in model.parameters()
            )
    finally:
        for handle in handles:
            handle.remove()
    assert len(generated_weight_shapes) == 4 * len(handles)
    assert set(generated_weight_shapes) == {(40, 8, 8)}
    assert any(
        not torch.equal(parameter, initial_parameters[name])
        for name, parameter in model.named_parameters()
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


@pytest.mark.parametrize("variant", ADAPTIVE_VARIANTS)
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


@pytest.mark.parametrize("variant", ADAPTIVE_VARIANTS)
@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize(
    "gradient_window",
    [
        {"recurrent_no_gradient_transition_count": 0},
        {"recurrent_gradient_transition_count": 1},
    ],
)
def test_bert_reasoning_growth_matches_the_gpt_schedule(
    variant, smooth, gradient_window
):
    growth_options = dict(
        recurrent_flag=True,
        recurrent_max_steps=5,
        recurrent_initial_iterations=2,
        recurrent_iteration_increment=1,
        recurrent_forward_calls_before_iteration_increment=2,
        recurrent_smooth_iteration_growth_flag=smooth,
        **gradient_window,
    )
    bert_config = configuration(
        "bert/" + variant, hierarchical=False, soft=False, **growth_options
    )
    gpt_config = gpt_configuration(variant, **growth_options)
    assert isinstance(
        bert_config.experiment_config.encoder_config, RecurrentLayerConfig
    )
    bert_model = model_package("bert/" + variant).build_model(bert_config).eval()
    gpt_model = model_package("gpt/" + variant).build_model(gpt_config).eval()
    bert_schedule = bert_model.transformer.recurrent_iteration_schedule
    gpt_schedule = gpt_model.transformer.recurrent_iteration_schedule
    token_ids, attention_mask, token_type_ids = inputs()
    observed_depths = []
    with torch.no_grad():
        for forward_index in range(7):
            assert bert_schedule.snapshot() == gpt_schedule.snapshot()
            observed_depths.append(bert_schedule.active_iterations)
            bert_model(token_ids, attention_mask, token_type_ids)
            gpt_model(token_ids, attention_mask)
            if forward_index == 2:
                checkpoint = copy.deepcopy(bert_model.state_dict())
                checkpoint_schedule = bert_schedule.snapshot()
    assert observed_depths[0] == 2
    assert observed_depths[-1] == 5
    assert bert_schedule.snapshot() == gpt_schedule.snapshot()
    restored_model = (
        model_package("bert/" + variant).build_model(copy.deepcopy(bert_config)).eval()
    )
    restored_model.load_state_dict(checkpoint, strict=True)
    assert (
        restored_model.transformer.recurrent_iteration_schedule.snapshot()
        == checkpoint_schedule
    )
    bert_model.load_state_dict(checkpoint, strict=True)
    with torch.no_grad():
        original_outputs = bert_model(token_ids, attention_mask, token_type_ids)
        restored_outputs = restored_model(token_ids, attention_mask, token_type_ids)
    for original_output, restored_output in zip(
        original_outputs, restored_outputs, strict=True
    ):
        torch.testing.assert_close(original_output, restored_output)
    assert (
        restored_model.transformer.recurrent_iteration_schedule.snapshot()
        == bert_schedule.snapshot()
    )


@pytest.mark.parametrize("variant", ADAPTIVE_VARIANTS)
def test_reasoning_growth_composes_with_hierarchical_embeddings_and_soft_moe(variant):
    bert_config = configuration(
        "bert/" + variant,
        recurrent_flag=True,
        recurrent_max_steps=3,
        recurrent_initial_iterations=2,
        recurrent_no_gradient_transition_count=0,
        recurrent_forward_calls_before_iteration_increment=2,
        recurrent_smooth_iteration_growth_flag=True,
    )
    model = model_package("bert/" + variant).build_model(bert_config).eval()
    model.set_token_vocabulary(TOKEN_TEXTS)
    schedule = model.transformer.recurrent_iteration_schedule
    assert schedule.active_iterations == 2
    for _ in range(3):
        mlm_logits, nsp_logits, auxiliary_loss = model(*inputs())
    assert schedule.active_iterations == 3
    loss = (
        mlm_logits.square().mean() + nsp_logits.square().mean() + auxiliary_loss.sum()
    )
    loss.backward()
    assert torch.isfinite(loss)
    for router in model.modules():
        if router.__class__.__name__ == "SoftMoERouter":
            assert all(parameter.grad is not None for parameter in router.parameters())


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


@pytest.mark.parametrize("variant", ("linear", "expert_linear", *ADAPTIVE_VARIANTS))
@pytest.mark.parametrize("mode", ("hierarchical", "contextual"))
def test_native_bert_embedding_flags_build_from_cli(variant, mode):
    package_name = f"bert/{variant}"
    package = model_package(package_name)
    length_flag = (
        "--hierarchical-max-token-bytes"
        if mode == "hierarchical"
        else "--contextual-embedding-max-token-bytes"
    )
    fields = {field.flag: field for field in configuration_schema(package).fields}
    assert fields[f"--{mode}-embedding-flag"].value_type == "bool"
    assert fields[length_flag].value_type == "int"
    assert fields["--mlm-decoder-weight-tying-flag"].value_type == "bool"
    parsed = get_experiment_parser(package).parse_args(
        [
            "--preset",
            "baseline",
            "--config",
            f"--{mode}-embedding-flag",
            "true",
            length_flag,
            "9",
            "--mlm-decoder-weight-tying-flag",
            "false",
        ]
    )
    selection = resolve_cli_selection(
        parsed, package, package.preset_type, build_monitor_callbacks=False
    )
    model_config = configuration(
        package_name, hierarchical=False, soft=False, **selection.config_overrides
    )
    model = package.build_model(model_config).eval()
    model.set_token_vocabulary(TOKEN_TEXTS)
    embedding = model.token_embedding.embedding.cfg
    expected = (
        HierarchicalByteEmbeddingConfig
        if mode == "hierarchical"
        else ByteContextualEmbeddingConfig
    )
    assert isinstance(embedding, expected)
    assert embedding.max_token_bytes == 9
    with torch.no_grad():
        outputs = model(*inputs())
    assert outputs[0].shape == (2, 4, len(TOKEN_TEXTS))
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.parametrize("variant", ("linear", "expert_linear", *ADAPTIVE_VARIANTS))
def test_native_bert_contextual_flag_preserves_explicit_config(variant):
    explicit = contextual_config(max_token_bytes=7, hidden_dim=8)
    model_config = configuration(
        f"bert/{variant}",
        hierarchical=False,
        soft=False,
        contextual_embedding_config=explicit,
        contextual_embedding_flag=True,
        contextual_embedding_max_token_bytes=5,
        mlm_decoder_weight_tying_flag=False,
    )
    options = model_config.experiment_config.boundary_config.embedding_options
    assert options.contextual_embedding_config == explicit
