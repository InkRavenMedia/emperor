"""Native CLI controls compose optional adaptive BERT components."""

import pytest
import torch
from torch import nn

from models.catalog import model_package
from tests.model_packages.test_bert_hierarchical_soft_moe import (
    TOKEN_TEXTS,
    configuration,
)
from tests.model_packages.test_bert_soft_attention import attention_routing_config

PACKAGE_NAME = "bert/expert_linear_adaptive"


@pytest.mark.parametrize(
    "attention,feed_forward,hierarchical",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
def test_component_flags_enable_independent_paths(
    attention, feed_forward, hierarchical
):
    package = model_package(PACKAGE_NAME)
    configuration = package.build_configuration(
        config_overrides={
            "batch_size": 2,
            "sequence_length": 5,
            "input_dim": len(TOKEN_TEXTS),
            "output_dim": len(TOKEN_TEXTS),
            "hidden_dim": 8,
            "attn_num_heads": 2,
            "stack_num_layers": 1,
            "num_experts": 3,
            "top_k": 2,
            "attention_soft_moe_flag": attention,
            "soft_moe_flag": feed_forward,
            "soft_moe_slots_per_expert": 3,
            "soft_moe_scoring_hidden_dim": 12,
            "soft_moe_scoring_num_layers": 1,
            "expert_attention_use_kv_expert_models_flag": False,
            "hierarchical_embedding_flag": hierarchical,
            "hierarchical_byte_embedding_dim": 8,
            "hierarchical_byte_feed_forward_dim": 16,
            "hierarchical_byte_num_heads": 2,
            "hierarchical_max_token_bytes": 12,
            "mlm_decoder_weight_tying_flag": not hierarchical,
        }
    )
    model = package.build_model(configuration)
    assert isinstance(model.token_embedding, nn.Embedding) is not hierarchical
    attention_routers = [
        module
        for name, module in model.named_modules()
        if "self_attention_layer" in name and type(module).__name__ == "SoftMoERouter"
    ]
    feed_forward_routers = [
        module
        for name, module in model.named_modules()
        if "feed_forward_layer" in name and type(module).__name__ == "SoftMoERouter"
    ]
    assert bool(attention_routers) is attention
    assert bool(feed_forward_routers) is feed_forward
    for router in attention_routers + feed_forward_routers:
        assert router.num_experts == 3
        assert router.slots_per_expert == 3
        assert router.router_config.model_config.hidden_dim == 12
        assert router.router_config.model_config.num_layers == 1
    if hierarchical:
        model.set_token_vocabulary(TOKEN_TEXTS)
    token_ids = torch.tensor([[2, 5, 4, 8, 3], [2, 7, 4, 9, 3]])
    outputs = model(token_ids, torch.ones_like(token_ids), torch.zeros_like(token_ids))
    assert outputs[0].shape == (2, 5, len(TOKEN_TEXTS))
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.parametrize(
    "field",
    [
        "soft_moe_slots_per_expert",
        "soft_moe_scoring_hidden_dim",
        "soft_moe_scoring_num_layers",
        "hierarchical_byte_embedding_dim",
        "hierarchical_byte_encoder_num_layers",
        "hierarchical_byte_feed_forward_dim",
        "hierarchical_byte_num_heads",
        "hierarchical_max_token_bytes",
    ],
)
def test_component_dimensions_reject_zero(field):
    from model_runtime.packages.runtime_defaults import RuntimeDefaultsError

    with pytest.raises(
        (ValueError, RuntimeDefaultsError), match="positive|greater than"
    ):
        model_package(PACKAGE_NAME).build_configuration(config_overrides={field: 0})


def test_explicit_component_configs_keep_their_settings_when_flags_are_enabled():
    model_configuration = configuration(
        PACKAGE_NAME,
        attention_soft_moe_config=attention_routing_config(),
        expert_attention_use_kv_expert_models_flag=False,
        attention_soft_moe_flag=True,
        soft_moe_flag=True,
        hierarchical_embedding_flag=True,
        soft_moe_slots_per_expert=7,
        soft_moe_scoring_hidden_dim=24,
        hierarchical_byte_embedding_dim=24,
        hierarchical_max_token_bytes=64,
    )
    model = model_package(PACKAGE_NAME).build_model(model_configuration)
    embedding_config = model_configuration.experiment_config.boundary_config.embedding_options.hierarchical_embedding_config
    assert embedding_config.byte_embedding_dim == 8
    assert embedding_config.max_token_bytes == 12
    routers = [
        module for module in model.modules() if type(module).__name__ == "SoftMoERouter"
    ]
    assert routers
    assert all(router.slots_per_expert == 2 for router in routers)
    assert {router.router_config.model_config.hidden_dim for router in routers} == {
        5,
        8,
    }
