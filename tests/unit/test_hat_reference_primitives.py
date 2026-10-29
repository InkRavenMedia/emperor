"""Small algebraic references for the primitives used by the HAT configuration."""

from dataclasses import replace

import pytest
import torch

from tests.support.hierarchical_embedding import encoder_layer_config


@pytest.mark.parametrize("causal", [False, True])
def test_rotary_attention_matches_explicit_pair_rotations(causal):
    cfg = replace(
        encoder_layer_config(causal=causal).attention_config,
        rotary_base=10000.0,
        return_attention_weights_flag=True,
    )
    model = cfg.build().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                parameter.copy_(torch.eye(8).repeat(1, parameter.shape[1] // 8))
            else:
                parameter.zero_()
    inputs = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 24
    actual, weights, _ = model(inputs, inputs, inputs)
    heads = inputs.reshape(1, 3, 2, 4).transpose(1, 2)
    rotated = heads.clone()
    for position in range(3):
        for pair in range(2):
            angle = torch.tensor(position / 10000 ** (2 * pair / 4))
            first, second = (
                heads[:, :, position, 2 * pair],
                heads[:, :, position, 2 * pair + 1],
            )
            rotated[:, :, position, 2 * pair] = (
                first * angle.cos() - second * angle.sin()
            )
            rotated[:, :, position, 2 * pair + 1] = (
                first * angle.sin() + second * angle.cos()
            )
    scores = rotated @ rotated.transpose(-1, -2) / 2
    if causal:
        scores.masked_fill_(torch.ones(3, 3, dtype=torch.bool).triu(1), -torch.inf)
    probabilities = scores.softmax(-1)
    expected = (probabilities @ heads).transpose(1, 2).reshape(1, 3, 8)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(weights, probabilities)


def test_swiglu_is_three_bias_free_projections_from_the_same_input():
    from emperor.transformer import SwiGLUFeedForwardConfig

    model = SwiGLUFeedForwardConfig(input_dim=6, output_dim=6, hidden_dim=10).build()
    source = torch.randn(2, 3, 6, requires_grad=True)
    actual, loss = model(source)
    gate = source @ model.gate.weight_params
    up = source @ model.up.weight_params
    expected = ((gate * gate.sigmoid()) * up) @ model.down.weight_params
    torch.testing.assert_close(actual, expected)
    assert loss.item() == 0
    assert all(
        layer.bias_params is None for layer in (model.gate, model.up, model.down)
    )
    actual.square().sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_transformer_can_select_final_rms_normalization():
    from emperor.layers import NormalizationOptions
    from tests.support.hierarchical_embedding import embedding_config

    cfg = replace(
        embedding_config().encoder_config,
        final_normalization=NormalizationOptions.RMS_NORM,
    )
    model = cfg.build()
    assert isinstance(model.encoder_layer_norm, torch.nn.RMSNorm)
