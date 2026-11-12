"""Mixer source memory retains the attention owner's token-chunk semantics."""

from dataclasses import replace

import pytest
import torch

from emperor.attention import AttentionSourceMemoryConfig, MixerAttentionConfig
from emperor.augmentations.adaptive_parameters import SumGroupingConfig
from emperor.linears import LinearLayerConfig
from tests.unit.test_attention_model_mixer import _linear_stack


def test_chunked_memory_uses_valid_token_sums_and_fixed_capacity():
    config = MixerAttentionConfig(
        embedding_dim=4,
        sequence_length=8,
        batch_first_flag=True,
        mixing_model_config=_linear_stack(8, 8),
        variable_sequence_length_flag=True,
        source_memory_config=AttentionSourceMemoryConfig(
            grouping_config=SumGroupingConfig(chunk_size=5),
            model_config=LinearLayerConfig(bias_flag=True),
        ),
    )
    model = config.build().eval()
    assert model.mixing_sequence_length == 10
    tokens = torch.arange(56, dtype=torch.float32).reshape(2, 7, 4) / 10
    padding = torch.tensor([[False] * 6 + [True], [False] * 3 + [True] * 4])
    observed = []
    handle = model.source_memory.model.register_forward_pre_hook(
        lambda _model, args: observed.append(args[0].detach().clone())
    )
    output, _, _ = model(tokens, tokens, tokens, padding)
    handle.remove()
    expected = torch.stack((tokens[0, :5].sum(0), tokens[0, 5], tokens[1, :3].sum(0)))
    torch.testing.assert_close(observed[0], expected)
    poisoned = tokens.masked_fill(padding.unsqueeze(-1), float("nan"))
    actual, _, _ = model(poisoned, poisoned, poisoned, padding)
    torch.testing.assert_close(actual, output)
    assert output.shape == tokens.shape
    assert (output[padding] == 0).all()
    output.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.source_memory.parameters()
    )
    sequence_first = replace(config, batch_first_flag=False).build().eval()
    sequence_first.load_state_dict(model.state_dict(), strict=True)
    transposed = tokens.transpose(0, 1)
    transposed_output, _, _ = sequence_first(
        transposed, transposed, transposed, padding
    )
    torch.testing.assert_close(transposed_output.transpose(0, 1), output)


@pytest.mark.parametrize("chunk_size", [None, 0, -1, True])
def test_invalid_chunk_size_fails_before_model_construction(chunk_size):
    config = MixerAttentionConfig(
        embedding_dim=4,
        sequence_length=8,
        batch_first_flag=True,
        mixing_model_config=_linear_stack(8, 8),
        source_memory_config=AttentionSourceMemoryConfig(
            grouping_config=SumGroupingConfig(chunk_size=chunk_size)
        ),
    )
    with pytest.raises(ValueError, match="chunk_size"):
        config.build()


def test_chunked_memory_validation_uses_its_registered_owner(monkeypatch):
    memory_config = AttentionSourceMemoryConfig(
        grouping_config=SumGroupingConfig(chunk_size=5)
    )
    memory_owner = memory_config.registry_owner()
    validated_configs = []

    class RegisteredMemoryValidator(memory_owner.VALIDATOR):
        @classmethod
        def validate_context_config(cls, config):
            validated_configs.append(config)
            super().validate_context_config(config)

    monkeypatch.setattr(memory_owner, "VALIDATOR", RegisteredMemoryValidator)
    mixer_validator = MixerAttentionConfig().registry_owner().VALIDATOR
    mixer_validator.validate_memory_config(memory_config)
    assert len(validated_configs) == 1
    assert validated_configs[0] is memory_config
