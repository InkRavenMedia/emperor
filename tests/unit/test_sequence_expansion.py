import io
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from emperor.attention import AttentionLayerState
from emperor.config import ConfigBase
from emperor.embedding.absolute import (
    ImageLearnedPositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.halting import HaltingHiddenStateModeOptions, StickBreakingConfig
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    AttentionResidualConfig,
    GateConfig,
    HierarchicalReasoningModelRecurrentConfig,
    InnerThinkingRecurrentConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerGateOptions,
    TinyRecursiveModelRecurrentConfig,
    WeightedResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.nn import Module
from emperor.sampler import RouterConfig, TokenSamplerConfig
from emperor.transformer import (
    SwiGLUFeedForwardConfig,
    TransformerConfig,
    TransformerDecoderBlockLayer,
    TransformerDecoderLayerState,
    TransformerEncoderBlockLayer,
    TransformerEncoderLayerState,
)
from emperor.transformer.sequence_expansion import (
    SequenceExpansion,
    SequenceExpansionConfig,
)
from support.layers import weighted_memory_config
from unit.test_transformer import decoder_stack, encoder_stack, linear_stack, recurrent
from unit.test_transformer_adaptive_grouping import grouped_feed_forward_stack


def expansion_config(**overrides):
    return replace(
        SequenceExpansionConfig(
            embedding_dim=4,
            max_new_tokens=3,
            model_config=LinearLayerConfig(bias_flag=True),
            positional_embedding_config=TextSinusoidalPositionalEmbeddingConfig(
                num_embeddings=3, auto_expand_flag=True
            ),
        ),
        **overrides,
    )


def test_expansion_reuses_batch_padding_on_the_next_call():
    module = expansion_config().build()
    prefer_count(module, 1)
    counts = iter(([2, 3], [1, 1]))
    handle = module.count_predictor.register_forward_hook(
        lambda predictor, args, logits: logits
        + F.one_hot(torch.tensor(next(counts)) - 1, 3).to(logits) * 10
    )
    hidden = torch.randn(2, 4, 4, requires_grad=True)
    try:
        first = module(hidden)
        second = module(first.hidden, first.key_padding_mask)
    finally:
        handle.remove()
    assert first.hidden.shape == (2, 7, 4)
    assert second.hidden.shape == (2, 8, 4)
    assert second.key_padding_mask.tolist() == [[False] * 7 + [True], [False] * 8]
    torch.testing.assert_close(second.hidden[0, :6], first.hidden[0, :6])
    assert second.expansion.token_indices.tolist() == [[6], [7]]
    generated = second.expansion.gather_generated(second.hidden)
    torch.testing.assert_close(generated[0, 0], second.hidden[0, 6])
    generated.square().sum().backward()
    assert module.count_predictor.weight_params.grad.norm() > 0
    assert hidden.grad.norm() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_repeated_placement_preserves_saved_context_and_all_expansion_gradients(dtype):
    torch.manual_seed(29)
    module = (
        expansion_config(
            positional_embedding_config=TextLearnedPositionalEmbeddingConfig(
                num_embeddings=16
            )
        )
        .build()
        .to(dtype=dtype)
    )
    prefer_count(module, 1)
    counts = iter(([1, 2], [2, 1], [3, 2]))
    handle = module.count_predictor.register_forward_hook(
        lambda predictor, args, logits: logits
        + F.one_hot(torch.tensor(next(counts)) - 1, 3).to(logits) * 4
    )
    source = torch.randn(2, 4, 7, dtype=dtype, requires_grad=True)
    hidden = source.transpose(1, 2)
    original_padding = torch.tensor(
        [
            [True, False, True, False, True, True, False],
            [False, True, False, False, True, False, True],
        ]
    )
    padding = torch.zeros(2, 7, dtype=dtype).masked_fill(original_padding, -torch.inf)
    saved_inputs = []
    records = []
    try:
        for lengths in ((4, 6), (6, 7), (9, 9)):
            saved_inputs.append(
                (hidden, hidden.detach().clone(), padding, padding.clone())
            )
            result = module(hidden, padding)
            hidden, padding = result.hidden, result.key_padding_mask
            records.append(result.expansion)
            result.expansion.count_logits.retain_grad()
            assert hidden.shape == (2, max(lengths), 4)
            assert hidden.dtype == dtype
            assert padding.tolist() == [
                [False] * length + [True] * (max(lengths) - length)
                for length in lengths
            ]
            assert torch.count_nonzero(hidden[padding]) == 0
    finally:
        handle.remove()
    for previous_hidden, saved_hidden, previous_padding, saved_padding in saved_inputs:
        torch.testing.assert_close(previous_hidden, saved_hidden, rtol=0, atol=0)
        torch.testing.assert_close(previous_padding, saved_padding, rtol=0, atol=0)
    loss = sum(
        event.gather_generated(hidden)[event.generated_mask].float().square().sum()
        for event in records
    )
    loss.backward()
    assert source.grad.isfinite().all()
    assert torch.count_nonzero(source.grad.transpose(1, 2)[original_padding]) == 0
    for event in records:
        assert event.count_logits.grad.isfinite().all()
        assert event.count_logits.grad.norm() > 0
    for parameter in module.parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all()
        assert parameter.grad.norm() > 0


def test_expansion_compacts_padding_without_changing_valid_token_order():
    module = expansion_config(max_new_tokens=1).build()
    hidden = torch.randn(2, 7, 4, requires_grad=True)
    padding = torch.tensor(
        [
            [True, False, True, False, False, True, True],
            [False, False, True, True, False, True, False],
        ]
    )
    result = module(hidden, padding)
    assert result.hidden.shape == (2, 5, 4)
    assert result.key_padding_mask.tolist() == [[False] * 4 + [True], [False] * 5]
    for row in range(2):
        valid = hidden[row, ~padding[row]]
        torch.testing.assert_close(result.hidden[row, : valid.size(0)], valid)
    assert result.expansion.token_indices.tolist() == [[3], [4]]
    result.hidden.square().sum().backward()
    assert torch.count_nonzero(hidden.grad[padding]) == 0


@pytest.mark.parametrize("decoder", [False, True])
@pytest.mark.parametrize(
    "variant", ["standard", "inner_thinking", "tiny", "hierarchical"]
)
def test_recurrent_expansion_reuses_padding_for_each_sample(variant, decoder):
    model = recurrent_expansion_config(
        variant, decoder=decoder, max_new_tokens=3
    ).build()
    prefer_count(model, 1)
    counts = iter(([1, 2], [2, 1]))
    handles = [
        module.count_predictor.register_forward_hook(
            lambda predictor, args, logits: logits
            + F.one_hot(torch.tensor(next(counts)) - 1, 3).to(logits) * 10
        )
        for module in model.modules()
        if isinstance(module, SequenceExpansion)
    ]
    padding = torch.tensor([[False, True, False], [False, False, False]])
    state = (
        decoder_state()
        if decoder
        else TransformerEncoderLayerState(torch.randn(2, 3, 4, requires_grad=True))
    )
    state.expansion_padding_mask = padding
    hidden = state.hidden
    try:
        result = model(state)
    finally:
        for handle in handles:
            handle.remove()
    assert result.hidden.shape == (2, 6, 4)
    assert result.expansion_padding_mask.tolist() == [[False] * 5 + [True], [False] * 6]
    assert result.sequence_expansions[-1].token_indices.tolist() == [[3, 4], [5, -1]]
    generated = result.sequence_expansions[-1].gather_generated(result.hidden)
    F.linear(generated, torch.randn(7, 4)).square().sum().backward()
    assert hidden.grad.isfinite().all()
    assert torch.count_nonzero(hidden.grad[padding]) == 0
    for module in model.modules():
        if isinstance(module, SequenceExpansion):
            assert module.new_position_vector.grad.norm() > 0


@pytest.mark.parametrize("decoder", [False, True])
@pytest.mark.parametrize("input_width", [4, 7])
@pytest.mark.parametrize(
    "variant", ["standard", "inner_thinking", "tiny", "hierarchical"]
)
def test_recurrence_matches_unpadded_input_when_expansion_compacts_the_batch(
    variant, decoder, input_width
):
    torch.manual_seed(64)
    cfg = recurrent_expansion_config(variant, decoder=decoder)
    if variant in ("tiny", "hierarchical"):
        cfg.initialization_standard_deviation = 0.2
    elif variant == "standard":
        cfg.reinject_original_hidden_flag = True
        cfg.residual_config = AdditiveResidualConfig()
    model = cfg.build().eval()
    reference_model = cfg.build().eval()
    reference_model.load_state_dict(model.state_dict())
    dense_hidden = torch.randn(2, 3, 4)
    state = decoder_state() if decoder else TransformerEncoderLayerState(dense_hidden)
    state.hidden = dense_hidden
    state.expansion_padding_mask = torch.zeros(2, 3, dtype=torch.bool)
    padded = replace(state)
    padding = torch.tensor(
        [
            [True, False, True, False, True, True, False],
            [False, True, False, True, True, False, True],
        ]
        if input_width == 7
        else [[True, False, False, False], [False, False, True, False]]
    )
    padded_hidden = torch.randn(2, input_width, 4)
    padded_hidden[~padding] = dense_hidden.flatten(0, 1)
    padded.hidden = padded_hidden.requires_grad_()
    padded.expansion_padding_mask = padding
    if decoder:
        padded.target_attention_mask = torch.ones(
            input_width, input_width, dtype=torch.bool
        ).triu(1)
        padded.cross_attention_mask = state.cross_attention_mask[:1].expand(
            input_width, -1
        )
    with torch.no_grad():
        expected = reference_model(state)
    result = model(padded)
    assert result.hidden.shape == (2, 5, 4)
    assert not result.expansion_padding_mask.any()
    torch.testing.assert_close(result.hidden, expected.hidden)
    result.hidden.square().sum().backward()
    assert padded_hidden.grad.isfinite().all()
    assert torch.count_nonzero(padded_hidden.grad[padding]) == 0


@dataclass
class _CountPredictorConfig(ConfigBase):
    input_dim: int | None = None
    output_dim: int | None = None

    def _registry_owner(self):
        return _CountPredictor


class _CountPredictor(Module):
    def __init__(self, cfg, overrides=None):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.projection = nn.Linear(self.cfg.input_dim, self.cfg.output_dim)
        self.loss_weight = nn.Parameter(torch.tensor(0.25))
        self.last_context = None

    def forward(self, state):
        self.last_context = state.hidden.detach()
        return replace(
            state,
            hidden=self.projection(state.hidden),
            loss=self.loss_weight.square().expand(state.hidden.size(0)),
        )


def test_count_predictor_is_required_and_must_build_a_module():
    with pytest.raises(ValueError, match="model_config is required"):
        expansion_config(model_config=None).build()
    with pytest.raises(TypeError, match="model_config must be a ConfigBase"):
        expansion_config(model_config=nn.Linear(4, 3)).build()
    predictor_config = _CountPredictorConfig()
    with patch.object(predictor_config, "build", return_value=object()):
        with pytest.raises(TypeError, match="must build a torch.nn.Module"):
            expansion_config(model_config=predictor_config).build()


def test_positions_are_optional_and_disabled_by_default():
    cfg = SequenceExpansionConfig(
        embedding_dim=4,
        max_new_tokens=3,
        model_config=LinearLayerConfig(bias_flag=True),
    )
    module = cfg.build()
    prefer_count(module, 2)
    assert cfg.positional_embedding_config is None
    assert module.position_embedding is None
    assert not any(
        name.startswith("position_embedding.") for name in module.state_dict()
    )
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    result = module(hidden, padding)
    context = hidden.masked_fill(padding.unsqueeze(-1), 0).sum(1) / (~padding).sum(
        1, keepdim=True
    )
    expected = (
        module.new_position_vector
        + result.expansion.selected_probabilities[:, None] * context
    )
    torch.testing.assert_close(
        result.expansion.gather_generated(result.hidden),
        expected[:, None].expand(-1, 2, -1),
    )
    result.expansion.gather_generated(result.hidden).square().sum().backward()
    assert module.count_predictor.weight_params.grad.norm() > 0
    assert module.new_position_vector.grad.norm() > 0
    assert torch.count_nonzero(hidden.grad[padding]) == 0


@pytest.mark.parametrize("padding_idx", [None, 0, 3])
def test_configured_sinusoidal_positions_preserve_options_and_infer_width(padding_idx):
    position_config = TextSinusoidalPositionalEmbeddingConfig(
        embedding_dim=99,
        num_embeddings=1,
        padding_idx=padding_idx,
        auto_expand_flag=True,
    )
    module = expansion_config(positional_embedding_config=position_config).build()
    prefer_count(module, 2)
    with torch.no_grad():
        module.new_position_vector.zero_()
    assert module.position_embedding.cfg.embedding_dim == 4
    assert module.position_embedding.cfg.num_embeddings == 1
    assert module.position_embedding.cfg.padding_idx == padding_idx
    assert module.position_embedding.cfg.auto_expand_flag is True
    assert position_config.embedding_dim == 99
    hidden = torch.zeros(2, 3, 4)
    result = module(hidden)
    index_offset = 0 if padding_idx is None else padding_idx + 1
    expected_indices = torch.tensor([3, 4]) + index_offset
    torch.testing.assert_close(
        result.expansion.gather_generated(result.hidden)[:, :, 0],
        expected_indices.float().sin().expand(2, -1),
    )


def test_configured_learned_positions_receive_gradients_and_round_trip():
    position_config = TextLearnedPositionalEmbeddingConfig(
        embedding_dim=99, num_embeddings=12, padding_idx=0
    )
    cfg = expansion_config(positional_embedding_config=position_config)
    module = cfg.build()
    prefer_count(module, 2)
    with torch.no_grad():
        module.new_position_vector.zero_()
    hidden = torch.zeros(2, 3, 4)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    result = module(hidden, padding)
    weights = module.position_embedding.embedding_model.weight
    assert position_config.embedding_dim == 99
    assert module.position_embedding.cfg.embedding_dim == 4
    torch.testing.assert_close(
        result.expansion.gather_generated(result.hidden),
        weights[torch.tensor([[3, 4], [4, 5]])],
    )
    result.expansion.gather_generated(result.hidden).sum().backward()
    assert weights.grad.norm() > 0
    assert torch.count_nonzero(weights.grad[0]) == 0
    checkpoint = io.BytesIO()
    torch.save(module.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = cfg.build()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(hidden, padding).hidden, result.hidden)


def test_unused_generated_slots_do_not_exhaust_learned_position_capacity():
    module = expansion_config(
        positional_embedding_config=TextLearnedPositionalEmbeddingConfig(
            num_embeddings=6, padding_idx=0
        )
    ).build()
    with torch.no_grad():
        module.count_predictor.weight_params.zero_()
        module.count_predictor.bias_params.zero_()
    handle = module.count_predictor.register_forward_hook(
        lambda predictor, args, logits: logits
        + logits.new_tensor([[4, 0, 0], [0, 0, 4]])
    )
    hidden = torch.zeros(2, 5, 4)
    padding = torch.tensor([[False] * 5, [False, False, False, True, True]])
    try:
        result = module(hidden, padding)
    finally:
        handle.remove()
    assert result.expansion.position_ids.tolist() == [[5, -1, -1], [3, 4, 5]]
    assert result.hidden.shape == (2, 6, 4)
    assert not result.key_padding_mask.any()


@pytest.mark.parametrize(
    "position_config",
    [LinearLayerConfig(), ImageLearnedPositionalEmbeddingConfig(), object()],
)
def test_positions_require_a_supported_text_config(position_config):
    with pytest.raises(TypeError, match="positional_embedding_config"):
        expansion_config(positional_embedding_config=position_config).build()


def test_configured_multilayer_predictor_infers_dimensions_and_round_trips():
    torch.manual_seed(48)
    model_config = linear_stack(
        99, 99, hidden_dim=7, num_layers=2, activation=ActivationOptions.GELU
    )
    cfg = expansion_config(model_config=model_config)
    module = cfg.build()
    predictor = module.count_predictor
    assert predictor.cfg.input_dim == 4 and predictor.cfg.output_dim == 3
    assert len(predictor) == 2 and predictor.hidden_dim == 7
    assert model_config.input_dim == 99 and model_config.output_dim == 99
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    result = module(hidden, padding)
    assert result.expansion.count_logits.shape == (2, 3)
    generated = result.expansion.gather_generated(result.hidden)[
        result.expansion.generated_mask
    ]
    generated.square().mean().backward()
    assert all(
        parameter.grad is not None and parameter.grad.norm() > 0
        for parameter in predictor.parameters()
    )
    assert torch.count_nonzero(hidden.grad[padding]) == 0
    checkpoint = io.BytesIO()
    torch.save(module.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = cfg.build()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual = restored(hidden.detach(), padding)
    torch.testing.assert_close(actual.hidden, result.hidden)
    torch.testing.assert_close(
        actual.expansion.count_logits, result.expansion.count_logits
    )


@pytest.mark.parametrize("decoder", [False, True])
def test_custom_count_model_and_auxiliary_loss_run_in_recurrent_transformer(decoder):
    cfg = recurrent_expansion_config("standard", max_new_tokens=3, decoder=decoder)
    cfg.block_config.layer_config.sequence_expansion_config.model_config = (
        _CountPredictorConfig()
    )
    model = cfg.build()
    incoming_loss = torch.tensor(2.0, requires_grad=True)
    state = (
        decoder_state()
        if decoder
        else TransformerEncoderLayerState(torch.randn(2, 3, 4))
    )
    state.loss = incoming_loss
    output = model(state)
    predictor = next(
        module for module in model.modules() if isinstance(module, _CountPredictor)
    )
    assert predictor.cfg.input_dim == 4 and predictor.cfg.output_dim == 3
    assert predictor.last_context.shape == (2, 4)
    assert len(output.sequence_expansions) == 2
    torch.testing.assert_close(output.loss, torch.tensor(2.125))
    output.loss.backward()
    torch.testing.assert_close(predictor.loss_weight.grad, torch.tensor(1.0))
    torch.testing.assert_close(incoming_loss.grad, torch.tensor(1.0))


def test_predictor_receives_only_valid_pooled_context():
    module = expansion_config(model_config=_CountPredictorConfig()).build()
    hidden = torch.tensor([[[1.0, 2, 3, 4], [3, 4, 5, 6], [999, 999, 999, 999]]])
    result = module(hidden, torch.tensor([[False, False, True]]))
    torch.testing.assert_close(
        module.count_predictor.last_context, torch.tensor([[2.0, 3, 4, 5]])
    )
    assert result.loss.shape == (1,)
    (
        result.expansion.gather_generated(result.hidden).square().mean()
        + result.loss.mean()
    ).backward()
    assert module.count_predictor.projection.weight.grad.norm() > 0
    assert module.count_predictor.loss_weight.grad.norm() > 0


@pytest.mark.parametrize("invalid", ["type", "shape", "nan", "integer", "loss"])
def test_invalid_count_predictor_outputs_fail_clearly(invalid):
    module = expansion_config(model_config=_CountPredictorConfig()).build()

    def corrupt_prediction(model, args, output):
        if invalid == "type":
            return output.hidden
        if invalid == "shape":
            return replace(output, hidden=output.hidden[:, :1])
        if invalid == "nan":
            return replace(output, hidden=output.hidden * float("nan"))
        if invalid == "integer":
            return replace(output, hidden=output.hidden.long())
        return replace(output, loss=float("nan"))

    handle = module.count_predictor.register_forward_hook(corrupt_prediction)
    try:
        with pytest.raises(
            (TypeError, ValueError), match="SequenceExpansion count predictor"
        ):
            module(torch.randn(2, 3, 4))
    finally:
        handle.remove()


def block_config():
    cfg = encoder_stack(embedding_dim=4, num_layers=1).layer_config
    cfg.sequence_expansion_config = expansion_config(embedding_dim=None)
    return cfg


def prefer_count(model, count):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, SequenceExpansion):
                module.count_predictor.weight_params.zero_()
                module.count_predictor.bias_params.zero_()
                module.count_predictor.bias_params[count - 1] = 1


@contextmanager
def invalid_second_count_prediction(model):
    valid_predictions = iter((True, False))

    def corrupt_second_prediction(module, args, logits):
        return logits if next(valid_predictions) else torch.full_like(logits, torch.nan)

    handles = [
        module.count_predictor.register_forward_hook(corrupt_second_prediction)
        for module in model.modules()
        if isinstance(module, SequenceExpansion)
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def decoder_expansion_config(*, cross_attention=True, max_new_tokens=3):
    cfg = decoder_stack(
        embedding_dim=4, num_layers=1, include_cross_attention=cross_attention
    ).layer_config
    cfg.sequence_expansion_config = expansion_config(
        embedding_dim=None, max_new_tokens=max_new_tokens
    )
    return cfg


def decoder_state(*, explicit_masks=True, additive=False):
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    source = torch.randn(2, 5, 4, requires_grad=True)
    target_padding = torch.tensor([[False, False, True], [False, False, False]])
    encoder_padding = torch.tensor([[False, False, False, False, True], [False] * 5])
    target_mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
    cross_mask = torch.zeros(3, 5, dtype=torch.bool)
    cross_mask[:, 2] = True
    if additive:
        target_mask = torch.zeros_like(target_mask, dtype=torch.float32).masked_fill(
            target_mask, -torch.inf
        )
        cross_mask = torch.zeros_like(cross_mask, dtype=torch.float32).masked_fill(
            cross_mask, -torch.inf
        )
    return TransformerDecoderLayerState(
        hidden,
        target_key_padding_mask=target_padding,
        target_attention_mask=target_mask if explicit_masks else None,
        encoder_output=source,
        encoder_padding_mask=encoder_padding,
        cross_attention_mask=cross_mask if explicit_masks else None,
        controller_state={"sentinel": True},
    )


@pytest.mark.parametrize("decoder", [False, True])
@pytest.mark.parametrize("selected_tokens", [False, True])
def test_state_applies_repeated_expansion_and_leaves_loss_to_the_layer(
    decoder, selected_tokens
):
    state = decoder_state()
    if not decoder:
        state = TransformerEncoderLayerState(
            state.hidden, key_padding_mask=state.target_key_padding_mask
        )
    if selected_tokens:
        selected = replace(state, hidden=state.hidden[:, :1])
        selected.expansion_padding_mask = state.expansion_padding_mask[:, :1]
        selected.resize_expansion_masks(1)
        state = state.select_recurrent_tokens(selected)

    incoming_loss = torch.tensor(2.0, requires_grad=True)
    state.loss = incoming_loss
    module = expansion_config(model_config=_CountPredictorConfig()).build()
    with torch.no_grad():
        module.count_predictor.projection.weight.zero_()
        module.count_predictor.projection.bias.zero_()
    counts = iter(([2, 1], [1, 2]))
    handle = module.count_predictor.projection.register_forward_hook(
        lambda predictor, args, logits: logits
        + F.one_hot(torch.tensor(next(counts)) - 1, 3).to(logits) * 4
    )
    results = []
    try:
        for _ in range(2):
            result = module(
                state.hidden,
                state.expansion_padding_mask,
                position_offset=state.expansion_position_offset,
            )
            state.apply_expansion(result)
            results.append(result)
            assert state.hidden is result.hidden
            assert state.expansion_padding_mask is result.key_padding_mask
            assert state.sequence_expansions[-1] is result.expansion
            assert state.loss is incoming_loss
    finally:
        handle.remove()

    assert [record.position_ids.tolist() for record in state.sequence_expansions] == [
        [[2, 3], [3, -1]],
        [[4, -1], [4, 5]],
    ]
    if selected_tokens:
        assert state.expansion_position_offset.tolist() == [5, 6]
        assert state.hidden.shape == (2, 4, 4)
        assert not state.expansion_padding_mask.any()
    else:
        assert state.expansion_position_offset is None
        assert state.hidden.shape == (2, 6, 4)
        assert state.expansion_padding_mask.tolist() == [
            [False] * 5 + [True],
            [False] * 6,
        ]
    if decoder:
        sequence_length = state.hidden.size(1)
        torch.testing.assert_close(
            state.target_attention_mask,
            torch.ones(sequence_length, sequence_length, dtype=torch.bool).triu(1),
        )
        assert state.cross_attention_mask.shape == (sequence_length, 5)
        assert state.cross_attention_mask[:, 2].all()
    state.hidden.square().sum().backward()
    assert module.count_predictor.projection.weight.grad.norm() > 0
    assert module.count_predictor.loss_weight.grad is None
    assert incoming_loss.grad is None
    assert all(result.loss is not None for result in results)


@pytest.mark.parametrize("mask_kind", ["automatic", "bool", "float", "per_head"])
def test_decoder_expansion_preserves_causal_prefix_and_cross_attention_context(
    mask_kind,
):
    torch.manual_seed(39)
    cfg = decoder_expansion_config()
    cfg.layer_model_config.self_attention_config.return_attention_weights_flag = True
    block = cfg.build().eval()
    prefer_count(block.sequence_expansion, 2)
    state = decoder_state(
        explicit_masks=mask_kind != "automatic", additive=mask_kind == "float"
    )
    if mask_kind == "per_head":
        state.target_attention_mask = state.target_attention_mask.expand(4, -1, -1)
        state.cross_attention_mask = state.cross_attention_mask.expand(4, -1, -1)
    hidden, source = state.hidden, state.encoder_output
    source_padding, controller = state.encoder_padding_mask, state.controller_state
    prefix_output, _ = block.model(
        hidden,
        encoder_output=source,
        key_padding_mask=state.target_key_padding_mask,
        encoder_padding_mask=source_padding,
        attention_mask=state.target_attention_mask,
        encoder_attention_mask=state.cross_attention_mask,
    )
    captured = {}
    self_handle = block.model.self_attention_model.register_forward_hook(
        lambda module, args, output: captured.update(self_attention=output[1])
    )
    cross_handle = block.model.cross_attention_model.register_forward_pre_hook(
        lambda module, args, kwargs: captured.update(cross_attention=kwargs),
        with_kwargs=True,
    )
    try:
        result = block(state)
    finally:
        self_handle.remove()
        cross_handle.remove()
    assert result.hidden.shape == (2, 5, 4)
    assert result.encoder_output is source
    assert result.encoder_padding_mask is source_padding
    assert result.controller_state is controller
    original_valid = ~result.sequence_expansions[0].input_padding_mask
    torch.testing.assert_close(
        result.hidden[:, :3][original_valid], prefix_output[original_valid]
    )
    assert captured["self_attention"].shape[-2:] == (5, 5)
    assert torch.count_nonzero(captured["self_attention"].triu(1)) == 0
    assert torch.count_nonzero(captured["self_attention"][0, :, 4]) == 0
    assert (captured["self_attention"][0, 2:, 2] > 0).all()
    assert captured["cross_attention"]["q"].shape == (2, 5, 4)
    assert captured["cross_attention"]["k"] is source
    assert captured["cross_attention"]["k_padding_mask"] is source_padding
    if mask_kind != "automatic":
        assert result.target_attention_mask.shape[-2:] == (5, 5)
        assert result.cross_attention_mask.shape[-2:] == (5, 5)
        assert (
            captured["cross_attention"]["attention_mask"] is result.cross_attention_mask
        )
    prefix_gradient = torch.autograd.grad(
        result.hidden[:, :3][original_valid].square().sum(),
        block.sequence_expansion.new_position_vector,
        retain_graph=True,
    )[0]
    assert torch.count_nonzero(prefix_gradient) == 0
    logits = F.linear(
        result.sequence_expansions[-1].gather_generated(result.hidden),
        torch.randn(7, 4),
    )
    F.cross_entropy(logits.flatten(0, 1), torch.tensor([1, 2, 3, 4])).backward()
    for parameter in (
        hidden,
        source,
        block.sequence_expansion.count_predictor.weight_params,
        block.sequence_expansion.new_position_vector,
    ):
        assert parameter.grad is not None and parameter.grad.isfinite().all()
        assert parameter.grad.norm() > 0
    assert torch.count_nonzero(hidden.grad[0, 2]) == 0
    assert torch.count_nonzero(source.grad[source_padding]) == 0
    if mask_kind != "automatic":
        assert torch.count_nonzero(source.grad[:, 2]) == 0


@pytest.mark.parametrize(
    "variant", ["standard", "inner_thinking", "tiny", "hierarchical"]
)
def test_decoder_expansion_runs_inside_every_recurrent_variant(variant):
    model = recurrent_expansion_config(variant, decoder=True).build()
    state = decoder_state()
    source, source_padding, controller = (
        state.encoder_output,
        state.encoder_padding_mask,
        state.controller_state,
    )
    hidden = state.hidden
    result = model(state)
    assert result is state
    assert result.hidden.shape == (2, 5, 4)
    assert [record.token_indices.tolist() for record in result.sequence_expansions] == [
        [[2], [3]],
        [[3], [4]],
    ]
    assert [record.position_ids.tolist() for record in result.sequence_expansions] == [
        [[2], [3]],
        [[3], [4]],
    ]
    assert result.target_key_padding_mask.shape == (2, 5)
    assert result.target_attention_mask.shape == (5, 5)
    assert result.cross_attention_mask.shape == (5, 5)
    assert (
        result.encoder_output is source
        and result.encoder_padding_mask is source_padding
    )
    assert result.controller_state is controller
    F.linear(result.hidden[:, -1], torch.randn(7, 4)).square().mean().backward()
    assert hidden.grad is not None and hidden.grad.isfinite().all()
    assert source.grad.norm() > 0
    for module in model.modules():
        if isinstance(module, SequenceExpansion):
            assert module.new_position_vector.grad.norm() > 0


@pytest.mark.parametrize("causal", [False, True])
def test_decoder_only_transformer_supports_expansion(causal):
    stack = decoder_stack(embedding_dim=4, num_layers=2, include_cross_attention=False)
    stack.layer_config.sequence_expansion_config = expansion_config(max_new_tokens=1)
    stack.layer_config.layer_model_config.self_attention_config.causal_attention_mask_flag = causal
    model = TransformerConfig(decoder_stack_config=stack).build()
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    output, loss = model(target_token_embeddings=hidden)
    assert output.shape == (2, 5, 4)
    assert loss.isfinite()
    F.linear(output[:, -1], torch.randn(7, 4)).square().mean().backward()
    assert hidden.grad.norm() > 0


def test_disabled_decoder_expansion_keeps_initialization_outputs_and_gradients():
    cfg = decoder_expansion_config()
    cfg.sequence_expansion_config = None

    # The former block implementation had only the ordinary Layer construction
    # and decoder model-processing override; use that as the compatibility oracle.
    class LegacyDecoderBlock(Layer):
        _handle_model_processing = TransformerDecoderBlockLayer._handle_model_processing

    torch.manual_seed(23)
    legacy = LegacyDecoderBlock(
        LayerConfig(
            **{field.name: getattr(cfg, field.name) for field in fields(LayerConfig)}
        )
    )
    legacy_rng = torch.get_rng_state()
    torch.manual_seed(23)
    current = cfg.build()
    assert torch.equal(torch.get_rng_state(), legacy_rng)
    assert list(current.state_dict()) == list(legacy.state_dict())
    state = decoder_state()
    current_state = replace(
        state,
        hidden=state.hidden.detach().clone().requires_grad_(),
        encoder_output=state.encoder_output.detach().clone().requires_grad_(),
    )
    legacy_output = legacy(state)
    current_output = current(current_state)
    torch.testing.assert_close(
        current_output.hidden, legacy_output.hidden, rtol=0, atol=0
    )
    legacy_output.hidden.square().sum().backward()
    current_output.hidden.square().sum().backward()
    for old, new in zip(legacy.parameters(), current.parameters(), strict=True):
        torch.testing.assert_close(new.grad, old.grad, rtol=0, atol=0)
    assert current_output.sequence_expansions == ()


def test_disabled_decoder_inner_thinking_preserves_unbatched_inputs():
    cfg = recurrent_expansion_config("inner_thinking", decoder=True)
    cfg.block_config.layer_config.sequence_expansion_config = None
    model = cfg.build()
    result = model(
        TransformerDecoderLayerState(
            torch.randn(3, 4),
            target_key_padding_mask=torch.tensor([False, False, True]),
            encoder_output=torch.randn(5, 4),
        )
    )
    assert result.hidden.shape == (3, 4)
    assert result.sequence_expansions == ()


def test_decoder_inner_thinking_grows_ragged_selected_sequences_with_causal_masks():
    torch.manual_seed(45)
    model = recurrent_expansion_config(
        "inner_thinking", decoder=True, max_new_tokens=3
    ).build()
    expansion = next(
        module for module in model.modules() if isinstance(module, SequenceExpansion)
    )
    head = expansion.count_predictor
    with torch.no_grad():
        head.weight_params.zero_()
        head.bias_params.zero_()
    counts = iter(([1, 3], [2, 1]))
    handle = head.register_forward_hook(
        lambda module, args, logits: logits
        + F.one_hot(torch.tensor(next(counts)) - 1, 3).to(logits) * 4
    )
    state = decoder_state()
    state.target_key_padding_mask[0, 1] = True
    hidden = state.hidden
    try:
        result = model(state)
    finally:
        handle.remove()
    assert result.hidden.shape == (2, 7, 4)
    assert [
        record.selected_counts.tolist() for record in result.sequence_expansions
    ] == [[1, 3], [2, 1]]
    assert [record.position_ids.tolist() for record in result.sequence_expansions] == [
        [[1, -1, -1], [3, 4, 5]],
        [[2, 3], [6, -1]],
    ]
    torch.testing.assert_close(
        result.target_attention_mask, torch.ones(7, 7, dtype=torch.bool).triu(1)
    )
    assert result.cross_attention_mask.shape == (7, 5)
    logits = F.linear(
        result.hidden[:, 6:][~result.target_key_padding_mask[:, 6:]], torch.randn(7, 4)
    )
    logits.square().mean().backward()
    assert head.weight_params.grad.norm() > 0
    assert torch.count_nonzero(hidden.grad[0, 1:]) == 0


@pytest.mark.parametrize(
    "variant", ["standard", "inner_thinking", "tiny", "hierarchical"]
)
def test_decoder_recurrent_predictor_failure_keeps_original_context(variant):
    model = recurrent_expansion_config(variant, decoder=True, max_new_tokens=3).build()
    state = decoder_state()
    original = replace(state)
    with (
        invalid_second_count_prediction(model),
        pytest.raises(ValueError, match="finite"),
    ):
        model(state)
    for name in (
        "hidden",
        "target_key_padding_mask",
        "target_attention_mask",
        "cross_attention_mask",
        "encoder_output",
        "encoder_padding_mask",
        "controller_state",
    ):
        assert getattr(state, name) is getattr(original, name)
    assert state.sequence_expansions == ()
    assert model.recurrent_iteration_schedule.forward_call_progress == 0


@pytest.mark.parametrize(
    "invalid_mask", ["target_shape", "target_policy", "cross_shape", "cross_policy"]
)
def test_decoder_rejects_masks_without_an_unambiguous_expansion_policy(invalid_mask):
    model = decoder_expansion_config(max_new_tokens=1).build()
    state = decoder_state()
    if invalid_mask == "target_shape":
        state.target_attention_mask = torch.zeros(2, 3, dtype=torch.bool)
    elif invalid_mask == "target_policy":
        state.target_attention_mask[1, 0] = True
    elif invalid_mask == "cross_shape":
        state.cross_attention_mask = torch.zeros(3, 4, dtype=torch.bool)
    else:
        state.cross_attention_mask[1, 0] = True
    hidden = state.hidden
    with pytest.raises(ValueError, match="SequenceExpansion"):
        model(state)
    assert state.hidden is hidden
    assert state.sequence_expansions == ()


def test_decoder_attention_keeps_its_own_capacity_checks():
    cfg = decoder_expansion_config(max_new_tokens=2)
    cfg.layer_model_config.self_attention_config.target_sequence_length = 5
    cfg.layer_model_config.self_attention_config.source_sequence_length = 5
    cfg.layer_model_config.cross_attention_config.target_sequence_length = 5
    cfg.layer_model_config.cross_attention_config.source_sequence_length = 5
    state = decoder_state()
    model = cfg.build()
    prefer_count(model, 2)
    assert model(state).hidden.shape == (2, 5, 4)
    cfg.layer_model_config.cross_attention_config.target_sequence_length = 3
    limited_model = cfg.build()
    prefer_count(limited_model, 2)
    with pytest.raises(ValueError, match="Runtime target_sequence_length.*exceeds"):
        limited_model(decoder_state())


@pytest.mark.parametrize("expand_encoder", [False, True])
def test_encoder_decoder_transformer_expands_its_recurrent_decoder(expand_encoder):
    cfg = TransformerConfig(
        encoder_stack_config=encoder_stack(embedding_dim=4, num_layers=1),
        decoder_stack_config=recurrent_expansion_config("standard", decoder=True),
    )
    if expand_encoder:
        cfg.encoder_stack_config.layer_config.sequence_expansion_config = (
            expansion_config(max_new_tokens=2)
        )
    model = cfg.build()
    if expand_encoder:
        prefer_count(model.encoder_model, 2)
    source = torch.randn(2, 5, 4, requires_grad=True)
    target = torch.randn(2, 3, 4, requires_grad=True)
    source_padding = torch.tensor([[False] * 4 + [True], [False] * 5])
    output, loss = model(
        source_token_embeddings=source,
        target_token_embeddings=target,
        source_key_padding_mask=source_padding,
        encoder_key_padding_mask=source_padding,
    )
    assert output.shape == (2, 5, 4)
    assert loss.isfinite()
    F.linear(output[:, -1], torch.randn(7, 4)).square().mean().backward()
    assert source.grad.norm() > 0 and target.grad.norm() > 0
    assert torch.count_nonzero(source.grad[source_padding]) == 0


@pytest.mark.parametrize(
    "mask_kind", ["implicit", "prefix_bool", "prefix_float", "full_bool", "full_float"]
)
def test_full_transformer_propagates_ragged_encoder_growth_to_cross_attention(
    mask_kind,
):
    encoder = encoder_stack(embedding_dim=4, num_layers=1)
    encoder.layer_config.sequence_expansion_config = expansion_config()
    decoder = decoder_stack(embedding_dim=4, num_layers=1)
    decoder.layer_config.sequence_expansion_config = expansion_config(max_new_tokens=1)
    model = TransformerConfig(
        encoder_stack_config=encoder, decoder_stack_config=decoder
    ).build()
    expansion = next(
        module
        for module in model.encoder_model.modules()
        if isinstance(module, SequenceExpansion)
    )
    with torch.no_grad():
        expansion.count_predictor.weight_params.zero_()
        expansion.count_predictor.bias_params.zero_()
    handle = expansion.count_predictor.register_forward_hook(
        lambda module, args, logits: logits
        + torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]]).to(logits)
    )
    captured = []
    decoder_handle = model.decoder_model.register_forward_pre_hook(
        lambda module, args: captured.append(replace(args[0]))
    )
    source_padding = torch.tensor([[False, False, True], [False, False, False]])
    padding = None
    if mask_kind != "implicit":
        padding = (
            torch.zeros(2, 5, dtype=torch.bool)
            if mask_kind.startswith("full")
            else source_padding
        )
        if mask_kind.endswith("float"):
            padding = torch.zeros_like(padding, dtype=torch.float32).masked_fill(
                padding, -torch.inf
            )
    try:
        hidden, _ = model(
            source_token_embeddings=torch.randn(2, 3, 4),
            target_token_embeddings=torch.randn(2, 3, 4),
            source_key_padding_mask=source_padding,
            encoder_key_padding_mask=padding,
        )
    finally:
        handle.remove()
        decoder_handle.remove()
    assert hidden.shape == (2, 4, 4)
    context = captured[0]
    assert context.encoder_output.shape == (2, 5, 4)
    actual_padding = context.encoder_padding_mask
    if actual_padding.dtype != torch.bool:
        actual_padding = actual_padding.isneginf()
    assert actual_padding.tolist() == [[False, False, False, True, True], [False] * 5]


@pytest.mark.parametrize("additive", [False, True])
@pytest.mark.parametrize("input_width", [5, 7])
def test_transformer_remaps_original_cross_attention_padding_when_reusing_slots(
    additive, input_width
):
    encoder = encoder_stack(embedding_dim=4, num_layers=1)
    encoder.layer_config.sequence_expansion_config = expansion_config(max_new_tokens=1)
    model = TransformerConfig(
        encoder_stack_config=encoder,
        decoder_stack_config=decoder_stack(embedding_dim=4, num_layers=1),
    ).build()
    source_padding = torch.tensor(
        [[False, True, False, True, False], [False, True, False, False, False]]
    )
    source_padding = F.pad(source_padding, (0, input_width - 5), value=True)
    cross_padding = source_padding.clone()
    cross_padding[:, 2] = True
    if additive:
        cross_padding = torch.zeros_like(
            cross_padding, dtype=torch.float32
        ).masked_fill(cross_padding, -torch.inf)
    captured = []
    handle = model.decoder_model.register_forward_pre_hook(
        lambda module, args: captured.append(replace(args[0]))
    )
    try:
        model(
            source_token_embeddings=torch.randn(2, input_width, 4),
            target_token_embeddings=torch.randn(2, 3, 4),
            source_key_padding_mask=source_padding,
            encoder_key_padding_mask=cross_padding,
        )
    finally:
        handle.remove()
    padding = captured[0].encoder_padding_mask
    assert padding.dtype == cross_padding.dtype
    if additive:
        padding = padding.isneginf()
    assert padding.tolist() == [
        [False, True, False, False, True],
        [False, True, False, False, False],
    ]


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_decoder_smooth_depth_growth_commits_matching_mask_shapes(weight):
    cfg = recurrent_expansion_config("standard", decoder=True)
    cfg.initial_iterations = 1
    cfg.gradient_transition_count = 1
    cfg.forward_calls_before_iteration_increment = 4
    cfg.smooth_iteration_growth_flag = True
    model = cfg.build().eval()
    schedule = model.recurrent_iteration_schedule
    schedule.load_state_dict({"forward_call_progress": torch.tensor(4)})
    plan = replace(schedule.execution_plan(), transition_weight=weight)
    state = decoder_state()
    reference_state = replace(state, hidden=state.hidden.clone())
    with patch.object(schedule, "execution_plan", return_value=plan):
        output = model(state)
    source = model.block_model(reference_state)
    target = model.block_model(replace(source, hidden=source.hidden.clone()))
    expected = (
        source.hidden
        if weight == 0
        else (1 - weight)
        * F.pad(
            source.hidden.masked_fill(source.target_key_padding_mask.unsqueeze(-1), 0),
            (0, 0, 0, 1),
        )
        + weight * target.hidden
    )
    torch.testing.assert_close(output.hidden, expected)
    length = 4 if weight == 0 else 5
    assert output.target_key_padding_mask.shape == (2, length)
    assert output.target_attention_mask.shape == (length, length)
    assert output.cross_attention_mask.shape == (length, 5)
    assert len(output.sequence_expansions) == (1 if weight == 0 else 2)


def test_decoder_mask_extension_keeps_existing_mask_gradient_connections():
    block = decoder_expansion_config(max_new_tokens=1).build()
    state = decoder_state(additive=True)
    original_target_mask = state.target_attention_mask.requires_grad_()
    original_cross_mask = state.cross_attention_mask.requires_grad_()
    output = block(state)
    target_prefix = output.target_attention_mask[:3, :3]
    cross_prefix = output.cross_attention_mask[:3]
    mask_loss = (
        target_prefix.masked_fill(target_prefix.isneginf(), 0).sum()
        + cross_prefix.masked_fill(cross_prefix.isneginf(), 0).sum()
    )
    mask_loss.backward()
    torch.testing.assert_close(
        original_target_mask.grad, original_target_mask.isfinite().float()
    )
    torch.testing.assert_close(
        original_cross_mask.grad, original_cross_mask.isfinite().float()
    )


def test_counts_context_and_active_outputs_are_independent_across_padded_samples():
    module = expansion_config().build()
    assert isinstance(module, SequenceExpansion)
    with torch.no_grad():
        module.count_predictor.weight_params.zero_()
        module.count_predictor.bias_params.zero_()
        module.count_predictor.weight_params[0] = torch.tensor([2.0, 0.0, -2.0])
    hidden = torch.tensor(
        [
            [[2.0, 1, 0, 0], [2, 0, 1, 0], [-999, 0, 0, 0]],
            [[-2.0, 1, 0, 0], [-2, 0, 1, 0], [-2, 0, 0, 1]],
        ]
    )
    padding = torch.tensor([[False, False, True], [False, False, False]])
    output = module(hidden, padding)
    assert output.expansion.selected_counts.tolist() == [1, 3]
    assert output.expansion.position_ids.tolist() == [[2, -1, -1], [3, 4, 5]]
    assert output.key_padding_mask.tolist() == [
        [False, False, False, True, True, True],
        [False, False, False, False, False, False],
    ]
    torch.testing.assert_close(
        output.hidden[:, :3][~padding], hidden[~padding], rtol=0, atol=0
    )
    torch.testing.assert_close(output.hidden[0, 3:], torch.zeros(3, 4))
    individual = module(hidden[:1], padding[:1])
    torch.testing.assert_close(output.hidden[:1, :3], individual.hidden)
    hidden[0, 2] = 999
    altered = module(hidden, padding)
    torch.testing.assert_close(output.hidden, altered.hidden)
    torch.testing.assert_close(
        output.expansion.selected_probabilities,
        altered.expansion.selected_probabilities,
    )


def test_each_call_selects_the_highest_probability_count_in_the_full_range():
    module = expansion_config(max_new_tokens=5).build()
    hidden = torch.randn(2, 3, 4)
    for count in range(1, 6):
        prefer_count(module, count)
        result = module(hidden)
        assert result.hidden.shape == (2, 3 + count, 4)
        assert result.expansion.selected_counts.tolist() == [count, count]
        probabilities = result.expansion.count_logits.softmax(-1)
        torch.testing.assert_close(
            result.expansion.selected_probabilities, probabilities.max(-1).values
        )


def test_repeated_growth_has_no_total_length_cap_or_new_parameters():
    module = expansion_config(max_new_tokens=5).build()
    prefer_count(module, 5)
    parameters = {name: id(parameter) for name, parameter in module.named_parameters()}
    hidden = torch.randn(2, 20, 4)
    for length in (25, 30, 35):
        result = module(hidden)
        assert result.hidden.shape == (2, length, 4)
        assert result.expansion.selected_counts.tolist() == [5, 5]
        torch.testing.assert_close(
            result.expansion.position_ids,
            torch.arange(length - 5, length).expand(2, -1),
        )
        hidden = result.hidden
    assert {
        name: id(parameter) for name, parameter in module.named_parameters()
    } == parameters


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_generated_encodings_use_full_sequence_positions_and_preserve_dtype(dtype):
    module = expansion_config(embedding_dim=5).build().to(dtype=dtype)
    prefer_count(module, 2)
    with torch.no_grad():
        module.new_position_vector.zero_()
    hidden = torch.zeros(2, 3, 5, dtype=dtype)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    offsets = torch.tensor([17, 23])
    result = module(hidden, padding, position_offset=offsets)
    expected_positions = torch.tensor([[17, 18], [23, 24]])
    torch.testing.assert_close(result.expansion.position_ids, expected_positions)
    assert result.hidden.dtype == dtype
    torch.testing.assert_close(
        result.expansion.gather_generated(result.hidden)[:, :, 0],
        expected_positions.float().sin().to(dtype),
        rtol=0,
        atol=0,
    )


def test_additive_padding_is_equivalent_and_padding_gets_no_context_gradient():
    module = expansion_config().build()
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    padding = torch.tensor([[False, False, True], [False, True, True]])
    additive = torch.zeros(2, 3).masked_fill(padding, float("-inf"))
    boolean_output = module(hidden, padding)
    output = module(hidden, additive)
    torch.testing.assert_close(boolean_output.hidden, output.hidden)
    output.expansion.gather_generated(output.hidden).square().sum().backward()
    torch.testing.assert_close(
        hidden.grad[padding], torch.zeros_like(hidden.grad[padding])
    )
    assert hidden.grad[~padding].abs().sum() > 0


def test_one_count_has_no_count_learning_signal_but_initialization_learns():
    module = expansion_config(max_new_tokens=1).build()
    output = module(torch.randn(2, 3, 4))
    assert output.expansion.selected_counts.tolist() == [1, 1]
    output.expansion.gather_generated(output.hidden).sum().backward()
    assert torch.count_nonzero(module.count_predictor.weight_params.grad) == 0
    assert module.new_position_vector.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "name,value",
    [
        ("embedding_dim", 0),
        ("max_new_tokens", True),
        ("max_new_tokens", 0),
        ("max_new_tokens", -1),
    ],
)
def test_invalid_bounds_are_rejected(name, value):
    with pytest.raises(ValueError, match="positive integer"):
        expansion_config(**{name: value}).build()


@pytest.mark.parametrize("mask_kind", ["none", "boolean", "additive"])
def test_input_validation_returns_none_and_preserves_inputs(mask_kind):
    cfg = expansion_config()
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    original_hidden = hidden.detach().clone()
    expected_padding = torch.tensor([[False, True, False], [False, False, True]])
    padding = expected_padding
    if mask_kind == "none":
        padding = None
        expected_padding = torch.zeros(2, 3, dtype=torch.bool)
    elif mask_kind == "additive":
        padding = torch.zeros(2, 3).masked_fill(expected_padding, -torch.inf)
    original_padding = None if padding is None else padding.clone()

    assert SequenceExpansion.VALIDATOR.validate_inputs(cfg, hidden, padding) is None
    torch.testing.assert_close(hidden, original_hidden)
    if padding is not None:
        torch.testing.assert_close(padding, original_padding)

    result = cfg.build()(hidden, padding)
    torch.testing.assert_close(result.expansion.input_padding_mask, expected_padding)
    if mask_kind == "boolean":
        assert result.expansion.input_padding_mask is padding


@pytest.mark.parametrize(
    "hidden,padding,match",
    [
        (torch.ones(2, 3, dtype=torch.long), None, "floating"),
        (torch.zeros(2, 0, 4), None, "nonempty"),
        (torch.ones(2, 3, 4), torch.ones(2, 3, dtype=torch.bool), "valid context"),
        (torch.ones(2, 3, 4), torch.full((2, 3), -torch.inf), "valid context"),
        (torch.ones(2, 3, 4), torch.ones(2, 3), "0/-inf"),
        (torch.ones(2, 3, 4), torch.zeros(3, 3, dtype=torch.bool), "shape and device"),
    ],
)
def test_invalid_inputs_are_rejected(hidden, padding, match):
    with pytest.raises(ValueError, match=match):
        expansion_config().build()(hidden, padding)


class _LegacyEncoderBlock(Layer):
    _handle_model_processing = TransformerEncoderBlockLayer._handle_model_processing


def test_disabled_expansion_preserves_parameters_rng_outputs_and_gradients():
    cfg = encoder_stack(embedding_dim=4, num_layers=1).layer_config
    legacy_cfg = LayerConfig(
        **{field.name: getattr(cfg, field.name) for field in fields(LayerConfig)}
    )
    torch.manual_seed(23)
    legacy = _LegacyEncoderBlock(legacy_cfg)
    legacy_rng = torch.get_rng_state()
    torch.manual_seed(23)
    current = cfg.build()
    assert current.sequence_expansion is None
    assert torch.equal(torch.get_rng_state(), legacy_rng)
    assert list(current.state_dict()) == list(legacy.state_dict())
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(current.state_dict()[name], value, rtol=0, atol=0)
    source = torch.randn(2, 3, 4)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    legacy_output = legacy(
        AttentionLayerState(source.clone(), key_padding_mask=padding)
    )
    current_output = current(
        AttentionLayerState(source.clone(), key_padding_mask=padding)
    )
    torch.testing.assert_close(
        current_output.hidden, legacy_output.hidden, rtol=0, atol=0
    )
    legacy_output.hidden.square().sum().backward()
    current_output.hidden.square().sum().backward()
    for old, new in zip(legacy.parameters(), current.parameters(), strict=True):
        torch.testing.assert_close(new.grad, old.grad, rtol=0, atol=0)


def test_expansion_precedes_residual_capture_and_attention_receives_active_new_positions():
    cfg = block_config()
    from emperor.layers import AdditiveResidualConfig

    cfg.residual_config = AdditiveResidualConfig()
    cfg.layer_model_config.attention_config.return_attention_weights_flag = True
    block = cfg.build().eval()
    with torch.no_grad():
        block.sequence_expansion.count_predictor.weight_params.zero_()
        block.sequence_expansion.count_predictor.weight_params[0] = torch.tensor(
            [2.0, 0.0, -2.0]
        )
        block.sequence_expansion.count_predictor.bias_params.zero_()
    captured = []
    handle = block.model.self_attention_model.register_forward_hook(
        lambda _module, _args, output: captured.append(output[1])
    )
    source = torch.randn(2, 3, 4)
    source[0, :, 0] = 2
    source[1, :, 0] = -2
    padding = torch.tensor([[False, False, True], [False, False, False]])
    state = TransformerEncoderLayerState(source.clone(), key_padding_mask=padding)
    expanded = block.sequence_expansion(source, padding)
    transformed, _loss = block.model(
        expanded.hidden, source_key_padding_mask=expanded.key_padding_mask
    )
    output = block(state)
    handle.remove()
    assert output is state
    torch.testing.assert_close(output.hidden, expanded.hidden + transformed)
    weights = captured[-1]
    assert weights.shape[-2:] == (6, 6)
    assert output.sequence_expansions[0].selected_counts.tolist() == [1, 3]
    assert torch.count_nonzero(weights[0, :, 3:]) == 0
    assert (weights[0, :, 2] > 0).all()
    assert (weights[1, :, 3:] > 0).all()


def test_token_loss_reaches_count_initializer_and_upstream_encoder_with_padding_excluded():
    torch.manual_seed(7)
    cfg = encoder_stack(embedding_dim=8, num_layers=1).layer_config
    cfg.layer_model_config.attention_config.batch_size = 2
    cfg.layer_model_config.attention_config.target_sequence_length = 12
    cfg.layer_model_config.attention_config.source_sequence_length = 12
    cfg.layer_model_config.feed_forward_config = SwiGLUFeedForwardConfig(
        input_dim=8, output_dim=8, hidden_dim=16
    )
    encoder = nn.Sequential(
        cfg.build(),
        replace(
            cfg, sequence_expansion_config=expansion_config(embedding_dim=None)
        ).build(),
    )
    head = nn.Linear(8, 13)
    source = torch.randn(2, 4, 8, requires_grad=True)
    padding = torch.tensor([[False, False, False, True], [False, False, False, False]])
    # Set distinct counts without saturating the real probability/gradient path.
    with torch.no_grad():
        contextual = encoder[0](
            TransformerEncoderLayerState(source.clone(), key_padding_mask=padding)
        ).hidden
        contexts = contextual.masked_fill(padding.unsqueeze(-1), 0).sum(1)
        contexts /= (~padding).sum(1, keepdim=True)
        direction = F.normalize(contexts[0] - contexts[1], dim=0)
        midpoint = direction @ contexts.mean(0)
        predictor = encoder[1].sequence_expansion.count_predictor
        predictor.weight_params.zero_()
        predictor.weight_params[:, 0] = direction
        predictor.weight_params[:, 2] = -direction
        predictor.bias_params.copy_(
            torch.stack((-midpoint, midpoint.new_zeros(()), midpoint))
        )
    state = encoder(TransformerEncoderLayerState(source, key_padding_mask=padding))
    event = state.sequence_expansions[-1]
    assert event.selected_counts.tolist() == [1, 3]
    logits = head(event.gather_generated(state.hidden))
    logits.retain_grad()
    targets = torch.ones(logits.shape[:2], dtype=torch.long)
    loss = F.cross_entropy(logits[event.generated_mask], targets[event.generated_mask])
    loss.backward()
    expansion = encoder[1].sequence_expansion
    for parameter in (
        expansion.count_predictor.weight_params,
        expansion.new_position_vector,
        next(encoder[0].parameters()),
        source,
    ):
        assert parameter.grad is not None
        assert parameter.grad.isfinite().all()
        assert parameter.grad.abs().sum() > 0
    assert torch.count_nonzero(logits.grad[~event.generated_mask]) == 0
    assert torch.count_nonzero(source.grad[padding]) == 0


def test_records_are_forward_local_and_repeated_expansion_continues_logical_positions():
    cfg = block_config()
    block = cfg.build()
    assert cfg.sequence_expansion_config.embedding_dim is None
    prefer_count(block.sequence_expansion, 2)
    source = torch.randn(2, 3, 4)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    first = block(
        TransformerEncoderLayerState(source.clone(), key_padding_mask=padding)
    )
    independent = block(
        TransformerEncoderLayerState(source.clone(), key_padding_mask=padding)
    )
    assert len(first.sequence_expansions) == len(independent.sequence_expansions) == 1
    assert first.sequence_expansions[0] is not independent.sequence_expansions[0]
    torch.testing.assert_close(first.hidden, independent.hidden)
    first_event = first.sequence_expansions[0]
    extended = block(first)
    assert len(extended.sequence_expansions) == 2
    assert extended.sequence_expansions[0] is first_event
    assert extended.sequence_expansions[1].token_indices.tolist() == [[4, 5], [5, 6]]
    assert extended.sequence_expansions[1].position_ids.tolist() == [[4, 5], [5, 6]]
    assert len(independent.sequence_expansions) == 1


def test_registered_parameters_round_trip_without_forward_state():
    cfg = block_config()
    model = cfg.build().eval()
    source = torch.randn(2, 3, 4)
    expected = model(TransformerEncoderLayerState(source.clone()))
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = cfg.build().eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual = restored(TransformerEncoderLayerState(source.clone()))
    torch.testing.assert_close(actual.hidden, expected.hidden)
    assert len(actual.sequence_expansions) == 1
    assert "sequence_expansion.count_predictor.weight_params" in dict(
        model.named_parameters()
    )
    assert "sequence_expansion.position_embedding.weights" not in model.state_dict()
    assert "sequence_expansion.position_embedding.weights" in dict(
        model.named_buffers()
    )


def test_layer_stack_expands_at_every_enabled_template_layer():
    cfg = encoder_stack(embedding_dim=4, num_layers=2)
    cfg.layer_config.sequence_expansion_config = expansion_config(max_new_tokens=1)
    model = cfg.build()
    result = model(TransformerEncoderLayerState(torch.randn(2, 3, 4)))
    assert result.hidden.shape == (2, 5, 4)
    assert [event.token_indices.tolist() for event in result.sequence_expansions] == [
        [[3], [3]],
        [[4], [4]],
    ]


@pytest.mark.parametrize(
    "option",
    [
        "causal",
        "sequence_first",
        "weighted_residual",
        "stateful_residual",
        "rotary",
        "grouping",
    ],
)
def test_unsupported_encoder_configurations_are_rejected(option):
    cfg = block_config()
    encoder = cfg.layer_model_config
    if option == "causal":
        encoder.attention_config.causal_attention_mask_flag = True
    elif option == "sequence_first":
        encoder.attention_config.batch_first_flag = False
    elif option == "weighted_residual":
        encoder.residual_config = WeightedResidualConfig(
            coefficient_activation=ActivationOptions.SIGMOID
        )
    elif option == "stateful_residual":
        cfg.residual_config = AttentionResidualConfig(
            block_size=2, rms_norm_epsilon=1e-6
        )
    elif option == "rotary":
        encoder.attention_config.rotary_base = 10000.0
    else:
        encoder.feed_forward_config.stack_config = grouped_feed_forward_stack()
    with pytest.raises(ValueError, match="SequenceExpansion"):
        cfg.build()


def recurrent_expansion_config(variant, *, max_new_tokens=1, decoder=False):
    cfg = (decoder_stack if decoder else encoder_stack)(embedding_dim=4, num_layers=1)
    cfg.layer_config.sequence_expansion_config = expansion_config(
        max_new_tokens=max_new_tokens
    )
    transformer = cfg.layer_config.layer_model_config
    attention = (
        transformer.self_attention_config if decoder else transformer.attention_config
    )
    attention.target_sequence_length = attention.source_sequence_length = 12
    if decoder:
        transformer.cross_attention_config.target_sequence_length = 12
    recurrent_cfg = recurrent(cfg)
    if variant == "inner_thinking":
        recurrent_cfg = InnerThinkingRecurrentConfig(
            **{
                field.name: getattr(recurrent_cfg, field.name)
                for field in fields(recurrent_cfg)
            },
            sampler_config=TokenSamplerConfig(
                selection_ratio=0.5,
                router_config=RouterConfig(
                    num_experts=1,
                    noisy_topk_flag=False,
                    model_config=linear_stack(4, 1),
                ),
            ),
        )
    elif variant in ("tiny", "hierarchical"):
        shared = dict(
            input_dim=4,
            output_dim=4,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        )
        if variant == "tiny":
            recurrent_cfg = TinyRecursiveModelRecurrentConfig(
                **shared,
                block_config=cfg,
                latent_updates_per_answer_update=1,
                answer_update_count=1,
            )
        else:
            recurrent_cfg = HierarchicalReasoningModelRecurrentConfig(
                **shared,
                high_block_config=cfg,
                low_block_config=cfg,
                high_cycles=1,
                low_cycles=1,
            )
    return recurrent_cfg


@pytest.mark.parametrize("decoder", [False, True])
@pytest.mark.parametrize("position_kind", ["disabled", "learned", "sinusoidal"])
def test_configured_positions_run_inside_recurrent_encoder_and_decoder(
    decoder, position_kind
):
    cfg = recurrent_expansion_config("standard", max_new_tokens=2, decoder=decoder)
    position_config = None
    if position_kind == "learned":
        position_config = TextLearnedPositionalEmbeddingConfig(
            num_embeddings=12, padding_idx=0
        )
    elif position_kind == "sinusoidal":
        position_config = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=1, auto_expand_flag=True
        )
    cfg.block_config.layer_config.sequence_expansion_config.positional_embedding_config = position_config
    model = cfg.build()
    prefer_count(model, 2)
    state = (
        decoder_state()
        if decoder
        else TransformerEncoderLayerState(
            torch.randn(2, 3, 4, requires_grad=True),
            key_padding_mask=torch.tensor(
                [[False, False, True], [False, False, False]]
            ),
        )
    )
    result = model(state)
    assert result.hidden.shape == (2, 7, 4)
    assert [event.position_ids.tolist() for event in result.sequence_expansions] == [
        [[2, 3], [3, 4]],
        [[4, 5], [5, 6]],
    ]
    F.linear(result.hidden[:, -1], torch.randn(7, 4)).square().mean().backward()
    for module in model.modules():
        if isinstance(module, SequenceExpansion):
            assert module.count_predictor.weight_params.grad.norm() > 0
            assert module.new_position_vector.grad.norm() > 0
            if position_kind == "learned":
                assert module.position_embedding.embedding_model.weight.grad.norm() > 0


@pytest.mark.parametrize("decoder", [False, True])
def test_recurrent_growth_uses_full_distribution_beyond_the_former_expansion_cap(
    decoder,
):
    cfg = recurrent_expansion_config("standard", max_new_tokens=5, decoder=decoder)
    cfg.max_steps = cfg.initial_iterations = 4
    transformer = cfg.block_config.layer_config.layer_model_config
    attention = (
        transformer.self_attention_config if decoder else transformer.attention_config
    )
    attention.target_sequence_length = attention.source_sequence_length = 32
    if decoder:
        transformer.cross_attention_config.target_sequence_length = 32
    model = cfg.build()
    prefer_count(model, 5)
    state = (
        decoder_state()
        if decoder
        else TransformerEncoderLayerState(torch.randn(2, 3, 4))
    )
    result = model(state)
    assert result.hidden.shape == (2, 23, 4)
    assert [event.selected_counts.tolist() for event in result.sequence_expansions] == [
        [5, 5]
    ] * 4


@pytest.mark.parametrize(
    "variant", ["standard", "inner_thinking", "tiny", "hierarchical"]
)
def test_recurrent_expansion_grows_on_each_transition_and_preserves_gradients(variant):
    torch.manual_seed(12)
    model = recurrent_expansion_config(variant).build()
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    state = TransformerEncoderLayerState(hidden, key_padding_mask=padding)
    output = model(state)
    assert output is state
    assert output.hidden.shape == (2, 5, 4)
    assert [event.token_indices.tolist() for event in output.sequence_expansions] == [
        [[2], [3]],
        [[3], [4]],
    ]
    assert [event.position_ids.tolist() for event in output.sequence_expansions] == [
        [[2], [3]],
        [[3], [4]],
    ]
    assert output.key_padding_mask.tolist() == [
        [False, False, False, False, True],
        [False] * 5,
    ]
    F.linear(output.hidden[:, -1], torch.randn(7, 4)).square().mean().backward()
    assert hidden.grad is not None and hidden.grad.isfinite().all()
    for module in model.modules():
        if isinstance(module, SequenceExpansion):
            assert module.count_predictor is not None
            assert module.new_position_vector.grad is not None
            assert module.new_position_vector.grad.norm() > 0


@pytest.mark.parametrize("enabled", [False, True])
def test_three_recurrent_passes_match_manual_loop_with_residual_and_reinjection(
    enabled,
):
    cfg = recurrent_expansion_config("standard", max_new_tokens=2)
    cfg.max_steps = cfg.initial_iterations = 3
    cfg.reinject_original_hidden_flag = True
    cfg.residual_config = AdditiveResidualConfig()
    cfg.block_config.layer_config.sequence_expansion_config = (
        expansion_config(max_new_tokens=2) if enabled else None
    )
    attention = cfg.block_config.layer_config.layer_model_config.attention_config
    attention.target_sequence_length = attention.source_sequence_length = 12
    model = cfg.build().eval()
    prefer_count(model, 2)
    source = torch.randn(2, 3, 4)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    observed = []
    with patch.object(
        model,
        "_observe_recurrent_step",
        side_effect=lambda before, after: observed.append((before.shape, after.shape)),
    ):
        actual = model(
            TransformerEncoderLayerState(source.clone(), key_padding_mask=padding)
        )
    manual = TransformerEncoderLayerState(source.clone(), key_padding_mask=padding)
    fixed_input = source.masked_fill(padding.unsqueeze(-1), 0) if enabled else source
    for _ in range(3):
        previous = manual.hidden
        previous_padding = manual.key_padding_mask
        manual.hidden = previous + F.pad(fixed_input, (0, 0, 0, previous.size(1) - 3))
        manual = model.block_model(manual)
        residual = (
            previous.masked_fill(previous_padding.unsqueeze(-1), 0)
            if enabled
            else previous
        )
        manual.hidden = manual.hidden + F.pad(
            residual, (0, 0, 0, manual.hidden.size(1) - previous.size(1))
        )
    torch.testing.assert_close(actual.hidden, manual.hidden)
    torch.testing.assert_close(actual.key_padding_mask, manual.key_padding_mask)
    assert actual.hidden.size(1) == (9 if enabled else 3)
    assert len(actual.sequence_expansions) == (3 if enabled else 0)
    assert all(before == after for before, after in observed)


@pytest.mark.parametrize("variant", ["standard", "inner_thinking"])
def test_learned_recurrent_counts_keep_ragged_masks_and_logical_positions(variant):
    torch.manual_seed(17)
    model = recurrent_expansion_config(variant, max_new_tokens=3).build()
    expansion = next(
        module for module in model.modules() if isinstance(module, SequenceExpansion)
    )
    head = expansion.count_predictor
    with torch.no_grad():
        head.weight_params.zero_()
        head.bias_params.zero_()
    biases = iter(
        (
            torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
            torch.tensor([[0.0, 4.0, 0.0], [4.0, 0.0, 0.0]]),
        )
    )
    handle = head.register_forward_hook(
        lambda module, args, output: output + next(biases).to(output)
    )
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    # Float padding is canonicalized before subsequent passes and token sampling.
    additive = torch.zeros(2, 3).masked_fill(padding, -torch.inf)
    try:
        result = model(TransformerEncoderLayerState(hidden, key_padding_mask=additive))
    finally:
        handle.remove()
    assert result.hidden.shape == (2, 6, 4)
    assert [
        record.selected_counts.tolist() for record in result.sequence_expansions
    ] == [[1, 2], [2, 1]]
    assert [record.position_ids.tolist() for record in result.sequence_expansions] == [
        [[2, -1], [3, 4]],
        [[3, 4], [5, -1]],
    ]
    assert result.key_padding_mask.tolist() == [
        [False, False, False, False, False, True],
        [False, False, False, False, False, False],
    ]
    F.linear(
        result.hidden[:, 5:][~result.key_padding_mask[:, 5:]], torch.randn(7, 4)
    ).square().mean().backward()
    assert head.weight_params.grad.norm() > 0
    assert hidden.grad.norm() > 0
    torch.testing.assert_close(
        hidden.grad[padding], torch.zeros_like(hidden.grad[padding])
    )


@pytest.mark.parametrize(
    "variant", ["standard", "inner_thinking", "tiny", "hierarchical"]
)
def test_recurrent_predictor_failure_does_not_commit_partial_sequence(variant):
    cfg = recurrent_expansion_config(variant, max_new_tokens=3)
    model = cfg.build()
    source = torch.randn(2, 3, 4)
    padding = torch.zeros(2, 3, dtype=torch.bool)
    caller_loss = torch.tensor(0.5)
    state = TransformerEncoderLayerState(
        source, loss=caller_loss, key_padding_mask=padding
    )
    with (
        invalid_second_count_prediction(model),
        pytest.raises(ValueError, match="finite"),
    ):
        model(state)
    assert state.hidden is source
    assert state.loss is caller_loss
    assert state.key_padding_mask is padding
    assert state.sequence_expansions == ()
    assert model.recurrent_iteration_schedule.forward_call_progress == 0


def test_recurrent_gradient_window_still_detaches_earlier_expansion():
    cfg = recurrent_expansion_config("standard", max_new_tokens=3)
    cfg.gradient_transition_count = 1
    model = cfg.build()
    expansion = next(
        module for module in model.modules() if isinstance(module, SequenceExpansion)
    )
    prefer_count(expansion, 1)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    result = model(TransformerEncoderLayerState(hidden))
    assert [
        event.count_logits.requires_grad for event in result.sequence_expansions
    ] == [False, True]
    F.linear(result.hidden[:, -1], torch.randn(7, 4)).square().mean().backward()
    assert hidden.grad is None
    assert expansion.count_predictor.weight_params.grad.norm() > 0


@pytest.mark.parametrize("suffix", [None, 1])
@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_smooth_recurrent_depth_growth_aligns_appended_positions(suffix, weight):
    torch.manual_seed(22)
    cfg = recurrent_expansion_config("standard")
    cfg.initial_iterations = 1
    cfg.forward_calls_before_iteration_increment = 4
    cfg.smooth_iteration_growth_flag = True
    cfg.gradient_transition_count = suffix
    cfg.no_gradient_transition_count = 0 if suffix is None else None
    model = cfg.build().eval()
    schedule = model.recurrent_iteration_schedule
    schedule.load_state_dict({"forward_call_progress": torch.tensor(4)})
    plan = replace(schedule.execution_plan(), transition_weight=weight)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    with patch.object(schedule, "execution_plan", return_value=plan):
        result = model(TransformerEncoderLayerState(hidden))
    source = model.block_model(TransformerEncoderLayerState(hidden.clone()))
    target = model.block_model(replace(source, hidden=source.hidden.clone()))
    expected = (
        source.hidden
        if weight == 0
        else (1 - weight) * F.pad(source.hidden, (0, 0, 0, 1)) + weight * target.hidden
    )
    torch.testing.assert_close(result.hidden, expected)
    assert len(result.sequence_expansions) == (1 if weight == 0 else 2)
    assert result.key_padding_mask.shape == result.hidden.shape[:2]
    result.hidden.square().mean().backward()
    assert all(
        parameter.grad.isfinite().all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.parametrize("removed_option", ["fixed_new_tokens", "max_sequence_length"])
def test_removed_count_and_capacity_options_are_rejected(removed_option):
    with pytest.raises(TypeError, match=removed_option):
        expansion_config(**{removed_option: 2})


def test_smooth_learned_counts_align_logical_positions_across_different_padding():
    cfg = recurrent_expansion_config("standard", max_new_tokens=3)
    cfg.max_steps = 3
    cfg.initial_iterations = 2
    cfg.gradient_transition_count = 2
    cfg.forward_calls_before_iteration_increment = 4
    cfg.smooth_iteration_growth_flag = True
    block = cfg.block_config.layer_config
    block.layer_model_config.attention_config.target_sequence_length = 12
    block.layer_model_config.attention_config.source_sequence_length = 12
    model = cfg.build().eval()
    expansion = next(
        module for module in model.modules() if isinstance(module, SequenceExpansion)
    )
    with torch.no_grad():
        expansion.count_predictor.weight_params.zero_()
        expansion.count_predictor.bias_params.zero_()
    # Shared boundary, provisional source, then two committed target passes.
    counts = iter(([1, 1], [1, 3], [3, 1], [1, 1]))

    def choose_counts(module, args, logits):
        indices = torch.tensor(next(counts)) - 1
        return logits + F.one_hot(indices, 3).to(logits) * 4

    handle = expansion.count_predictor.register_forward_hook(choose_counts)
    states = []
    output_handle = model.block_model.register_forward_hook(
        lambda module, args, output: states.append(replace(output))
    )
    schedule = model.recurrent_iteration_schedule
    schedule.load_state_dict({"forward_call_progress": torch.tensor(4)})
    plan = replace(schedule.execution_plan(), transition_weight=0.5)
    try:
        with patch.object(schedule, "execution_plan", return_value=plan):
            result = model(TransformerEncoderLayerState(torch.randn(2, 3, 4)))
    finally:
        handle.remove()
        output_handle.remove()
    source, target = states[1], states[-1]
    aligned_source = torch.zeros_like(target.hidden)
    aligned_source[:, :3] = source.hidden[:, :3]
    for sample in range(2):
        source_positions = (
            ~source.key_padding_mask[sample, 3:]
        ).nonzero().flatten() + 3
        target_positions = (
            ~target.key_padding_mask[sample, 3:]
        ).nonzero().flatten() + 3
        matched = min(source_positions.numel(), target_positions.numel())
        aligned_source[sample, target_positions[:matched]] = source.hidden[
            sample, source_positions[:matched]
        ]
    torch.testing.assert_close(
        result.hidden, 0.5 * aligned_source + 0.5 * target.hidden
    )
    torch.testing.assert_close(result.key_padding_mask, target.key_padding_mask)
    assert [event.selected_counts.tolist() for event in result.sequence_expansions] == [
        [1, 1],
        [3, 1],
        [1, 1],
    ]
    result.hidden.square().mean().backward()
    assert expansion.count_predictor.weight_params.grad.isfinite().all()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("controller", ["gate", "halting", "memory"])
def test_layer_controllers_reject_expansion_during_construction(shared, controller):
    cfg = encoder_stack(embedding_dim=4, num_layers=2)
    cfg.layer_config.sequence_expansion_config = expansion_config()
    if controller == "gate":
        controller_config = GateConfig(
            gate_dim=4,
            option=LayerGateOptions.ADDITION,
            model_config=linear_stack(4, 4),
        )
    elif controller == "halting":
        halting_gate = linear_stack(4, 2)
        halting_gate.last_layer_bias_option = LastLayerBiasOptions.DISABLED
        controller_config = StickBreakingConfig(
            input_dim=4,
            threshold=0.99,
            ponder_cost_weight=1.0,
            min_steps=1,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=halting_gate,
        )
    else:
        controller_config = weighted_memory_config(4)
    owner = cfg if shared else cfg.layer_config
    field_name = f"shared_{controller}_config" if shared else f"{controller}_config"
    setattr(owner, field_name, controller_config)
    with pytest.raises(ValueError, match="SequenceExpansion"):
        cfg.build()


@pytest.mark.parametrize("decoder", [False, True])
@pytest.mark.parametrize("additive", [False, True])
def test_block_state_validation_returns_none_and_preserves_context(decoder, additive):
    state = decoder_state(additive=additive)
    block_type = TransformerDecoderBlockLayer
    if not decoder:
        state = TransformerEncoderLayerState(
            state.hidden, key_padding_mask=state.target_key_padding_mask
        )
        block_type = TransformerEncoderBlockLayer
    state.loss = torch.tensor(2.0, requires_grad=True)
    original_fields = vars(state).copy()
    original_tensors = {
        name: value.detach().clone()
        for name, value in original_fields.items()
        if isinstance(value, torch.Tensor)
    }

    assert block_type.VALIDATOR.validate_expansion_state(state) is None
    assert vars(state).keys() == original_fields.keys()
    for name, value in original_fields.items():
        assert getattr(state, name) is value
    for name, value in original_tensors.items():
        torch.testing.assert_close(getattr(state, name), value)


def test_state_requires_records_and_rejects_pairwise_masks_and_controller_history():
    block = block_config().build()
    source = torch.randn(2, 3, 4)
    with pytest.raises(TypeError, match="TransformerEncoderLayerState"):
        block(AttentionLayerState(source))
    with pytest.raises(ValueError, match="pairwise"):
        block(TransformerEncoderLayerState(source, attention_mask=torch.zeros(3, 3)))
    with pytest.raises(ValueError, match="history"):
        block(TransformerEncoderLayerState(source, residual_state=object()))
