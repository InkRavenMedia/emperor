"""Adaptive BERT and GPT share reasoning controls and reusable embedding modes."""

import copy

import pytest
import torch

from models.catalog import model_package
from tests.model_packages.test_bert_hierarchical_soft_moe import (
    TOKEN_TEXTS,
    configuration,
    inputs,
)
from tests.unit.test_byte_contextual_embedding import contextual_config

ADAPTIVE_VARIANTS = ("linear_adaptive", "expert_linear_adaptive")


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
