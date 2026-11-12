"""Memory owns sequence layouts, padding, and its generated token count."""

from copy import deepcopy

import pytest
import torch

from emperor.attention import AttentionSourceMemoryConfig
from emperor.augmentations.adaptive_parameters import SumGroupingConfig
from emperor.memory import (
    AttentionDynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,
)
from tests.unit.test_attention_mixer_source_memory import memory_config


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("with_mask", [True, False])
@pytest.mark.parametrize(
    "variant, chunk_size",
    [
        (GatedResidualDynamicMemoryConfig, None),
        (WeightedDynamicMemoryConfig, None),
        (ElementWiseWeightedDynamicMemoryConfig, None),
        (AttentionDynamicMemoryConfig, None),
        (AttentionSourceMemoryConfig, 1),
        (AttentionSourceMemoryConfig, 5),
    ],
)
def test_source_memory_returns_tokens_with_owner_defined_capacity(
    variant, chunk_size, batch_first, with_mask
):
    torch.manual_seed(127)
    if chunk_size is not None:
        config = AttentionSourceMemoryConfig(
            grouping_config=SumGroupingConfig(chunk_size=chunk_size)
        )
    else:
        options = {}
        if variant is AttentionDynamicMemoryConfig:
            options["num_memory_slots"] = 2
        config = memory_config(variant, **options)
    original_config = deepcopy(config)
    memory = config.build_source_memory(4).double().eval()
    assert config == original_config
    tokens = torch.randn(2, 7, 8, dtype=torch.float64)[..., ::2]
    exclusions = torch.tensor([[False] * 6 + [True], [True] * 7])
    source = tokens
    mask = None
    if with_mask:
        source = tokens.masked_fill(exclusions.unsqueeze(-1), float("nan"))
        mask = exclusions
    if not batch_first:
        source = source.transpose(0, 1)
        if mask is not None:
            mask = mask.transpose(0, 1)
    source.requires_grad_()
    generated, loss = memory.generate_source_memory(
        source,
        excluded_token_mask=mask,
        batch_first_flag=batch_first,
    )
    batch_generated = generated
    if not batch_first:
        batch_generated = generated.transpose(0, 1)
    memory_length = config.source_memory_length(7)
    assert batch_generated.shape == (2, memory_length, 4)
    assert config.source_memory_length(8) >= memory_length
    assert generated.dtype == source.dtype and generated.device == source.device
    assert torch.isfinite(generated).all()

    clean_source = source
    if mask is not None:
        clean_source = source.masked_fill(mask.unsqueeze(-1), 0)
    if chunk_size is None:
        expected = memory(clean_source.reshape(-1, 4)).reshape_as(source)
        if mask is not None:
            expected = expected.masked_fill(mask.unsqueeze(-1), 0)
        torch.testing.assert_close(generated, expected)
    else:
        expected = torch.zeros_like(batch_generated)
        for sample_index in range(2):
            for chunk_index, start in enumerate(range(0, 7, chunk_size)):
                members = tokens[sample_index, start : start + chunk_size]
                if with_mask:
                    valid_members = ~exclusions[
                        sample_index, start : start + chunk_size
                    ]
                    members = members[valid_members]
                if members.size(0):
                    expected[sample_index, chunk_index] = memory.model(
                        members.sum(0, keepdim=True)
                    )[0]
        torch.testing.assert_close(batch_generated, expected)
    assert loss is None
    generated.square().sum().backward()
    assert torch.isfinite(source.grad).all()
    if mask is not None:
        assert (source.grad[mask] == 0).all()
        assert (batch_generated[1] == 0).all()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in memory.parameters()
    )
