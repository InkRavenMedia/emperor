"""BERT package composition of byte embeddings and sequence-aware Soft MoE."""

import copy
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from torch import nn

from models.catalog import model_package
from tests.support.hierarchical_embedding import embedding_config

BERT_PACKAGES = (
    "bert/linear",
    "bert/linear_adaptive",
    "bert/expert_linear",
    "bert/expert_linear_adaptive",
)
TOKEN_TEXTS = (
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "hello",
    "##world",
    "猫",
    "next",
    "sentence",
    "other",
    "token",
)


def configuration(package_name, *, hierarchical=True, soft=False, **overrides):
    hierarchical_config = None
    if hierarchical:
        hierarchical_config = embedding_config(output_dim=8, max_bytes=12)
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
        hierarchical_embedding_config=hierarchical_config,
        mlm_decoder_weight_tying_flag=not hierarchical,
    )
    if "expert" in package_name:
        values.update(
            num_experts=3,
            top_k=2,
            capacity_factor=0.0,
            expert_stack_hidden_dim=8,
            router_stack_hidden_dim=8,
        )
    if "adaptive" in package_name:
        values["adaptive_generator_stack_hidden_dim"] = 8
    values.update(overrides)
    package = model_package(package_name)
    return package.build_configuration(config_overrides=values)


def build_model(package_name, **options):
    model_config = configuration(package_name, **options)
    package = model_package(package_name)
    model = package.build_model(model_config).eval()
    if options.get("hierarchical", True):
        model.set_token_vocabulary(TOKEN_TEXTS)
    return model


def inputs():
    token_ids = torch.tensor([[2, 5, 4, 3], [2, 7, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    token_type_ids = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]])
    return token_ids, attention_mask, token_type_ids


@pytest.mark.parametrize("package_name", BERT_PACKAGES)
@pytest.mark.parametrize("hierarchical,soft", [(False, False), (True, False)])
def test_features_compose_and_receive_task_gradients(package_name, hierarchical, soft):
    torch.manual_seed(31)
    model = build_model(package_name, hierarchical=hierarchical, soft=soft)
    mlm_logits, nsp_logits, auxiliary_loss = model(*inputs())
    assert mlm_logits.shape == (2, 4, len(TOKEN_TEXTS))
    assert nsp_logits.shape == (2, 2)
    assert auxiliary_loss.numel() == 1
    task_loss = mlm_logits.square().mean() + nsp_logits.square().mean()
    total_loss = task_loss + auxiliary_loss.sum()
    total_loss.backward()
    assert torch.isfinite(total_loss)
    if not hierarchical:
        assert isinstance(model.token_embedding, nn.Embedding)
        assert model.mlm_decoder.weight is model.token_embedding.weight
    if hierarchical:
        byte_parameters = [
            parameter
            for name, parameter in model.token_embedding.named_parameters()
            if "byte_embedding" in name
        ]
        assert byte_parameters
        assert all(parameter.grad is not None for parameter in byte_parameters)
        assert any(parameter.grad.abs().sum() > 0 for parameter in byte_parameters)
    soft_routers = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "SoftMoERouter"
    ]
    assert bool(soft_routers) == soft
    for router in soft_routers:
        for parameter in router.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("package_name", BERT_PACKAGES)
def test_combined_features_preserve_padding_and_batch_independence(package_name):
    model = build_model(package_name)
    token_ids, attention_mask, token_type_ids = inputs()
    original_mlm, original_nsp, _ = model(token_ids, attention_mask, token_type_ids)
    changed_ids = token_ids.clone()
    changed_ids[attention_mask == 0] = 11
    changed_mlm, changed_nsp, _ = model(changed_ids, attention_mask, token_type_ids)
    valid_tokens = attention_mask.bool()
    torch.testing.assert_close(original_mlm[valid_tokens], changed_mlm[valid_tokens])
    torch.testing.assert_close(original_nsp, changed_nsp)
    for sample_index in range(2):
        sample_slice = slice(sample_index, sample_index + 1)
        sample_mlm, sample_nsp, _ = model(
            token_ids[sample_slice],
            attention_mask[sample_slice],
            token_type_ids[sample_slice],
        )
        torch.testing.assert_close(original_mlm[sample_slice], sample_mlm)
        torch.testing.assert_close(original_nsp[sample_slice], sample_nsp)


@pytest.mark.parametrize("package_name", BERT_PACKAGES)
def test_checkpoint_restores_the_exact_masked_vocabulary(package_name):
    model = build_model(package_name)
    restored_config = copy.deepcopy(model.cfg)
    restored_model = model_package(package_name).build_model(restored_config).eval()
    restored_model.load_state_dict(model.state_dict(), strict=True)
    original_outputs = model(*inputs())
    restored_outputs = restored_model(*inputs())
    for original, restored in zip(original_outputs, restored_outputs, strict=True):
        torch.testing.assert_close(original, restored)
    changed_vocabulary = list(TOKEN_TEXTS)
    changed_vocabulary[4] = "unmasked"
    with pytest.raises(ValueError, match="vocabulary"):
        restored_model.set_token_vocabulary(changed_vocabulary)


@pytest.mark.parametrize("package_name", BERT_PACKAGES)
def test_setup_binds_the_bert_tokenizer_without_changing_masked_ids(package_name):
    model_config = configuration(package_name)
    model = model_package(package_name).build_model(model_config).eval()
    with pytest.raises(ValueError, match="unbound"):
        model(*inputs())
    vocabulary = {token: index for index, token in enumerate(TOKEN_TEXTS)}
    tokenizer = Tokenizer(WordPiece(vocabulary, unk_token="[UNK]"))
    model._trainer = SimpleNamespace(datamodule=SimpleNamespace(tokenizer=tokenizer))
    model.setup("fit")
    observed_texts = []
    model.token_embedding.embedding.register_forward_pre_hook(
        lambda module, arguments: observed_texts.append(arguments[0])
    )
    assert model(*inputs())[0].shape == (2, 4, len(TOKEN_TEXTS))
    assert observed_texts[0][0] == ["[CLS]", "hello", "[MASK]", "[SEP]"]


@pytest.mark.parametrize("package_name", BERT_PACKAGES)
def test_hierarchical_embedding_rejects_tied_heads_and_oversized_vocabulary(
    package_name,
):
    model_config = configuration(package_name, mlm_decoder_weight_tying_flag=True)
    with pytest.raises(ValueError, match="weight.tying|weight tying"):
        model_package(package_name).build_model(model_config)
    model_config = configuration(package_name)
    model = model_package(package_name).build_model(model_config)
    oversized_tokens = list(TOKEN_TEXTS)
    oversized_tokens[5] = "x" * 13
    with pytest.raises(ValueError, match="bytes"):
        model.set_token_vocabulary(oversized_tokens)


@pytest.mark.parametrize("package_name", BERT_PACKAGES)
def test_combined_features_keep_bidirectional_context_and_segment_embeddings(
    package_name,
):
    model = build_model(package_name)
    token_ids, attention_mask, token_type_ids = inputs()
    original_mlm, _, _ = model(token_ids, attention_mask, token_type_ids)
    changed_ids = token_ids.clone()
    changed_ids[0, 2] = 8
    context_mlm, _, _ = model(changed_ids, attention_mask, token_type_ids)
    assert not torch.allclose(original_mlm[0, 0], context_mlm[0, 0])
    changed_segments = 1 - token_type_ids
    segment_mlm, _, _ = model(token_ids, attention_mask, changed_segments)
    assert not torch.allclose(original_mlm, segment_mlm)


def test_hierarchical_auxiliary_loss_reaches_the_bert_output():
    from tests.support.hierarchical_embedding import moe_config

    model_config = configuration("bert/linear")
    hierarchical_config = model_config.experiment_config.boundary_config.embedding_options.hierarchical_embedding_config
    hierarchical_config.projection_config = moe_config(8, 8, adaptive=False)
    model = model_package("bert/linear").build_model(model_config).eval()
    model.set_token_vocabulary(TOKEN_TEXTS)
    embedding_losses = []
    model.token_embedding.register_forward_hook(
        lambda module, arguments, state: embedding_losses.append(state.loss)
    )
    _, _, auxiliary_loss = model(*inputs())
    assert embedding_losses[0].item() > 0
    torch.testing.assert_close(
        auxiliary_loss.reshape(()), embedding_losses[0].reshape(())
    )


@pytest.mark.parametrize("package_name", BERT_PACKAGES)
def test_runtime_metadata_accepts_hierarchical_module_configuration(package_name):
    model_config = configuration(package_name)
    hierarchical_config = model_config.experiment_config.boundary_config.embedding_options.hierarchical_embedding_config
    runtime_values = {
        "hierarchical_embedding_config": hierarchical_config,
        "mlm_decoder_weight_tying_flag": False,
    }
    package = model_package(package_name)
    assert package.bind_runtime_defaults(runtime_values) is not None
