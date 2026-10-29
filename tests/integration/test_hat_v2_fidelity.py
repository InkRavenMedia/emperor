"""Paper-derived behavior checks, independent of package implementation choices."""

import copy

import pytest
import torch

from emperor.transformer.hierarchical import (
    HierarchicalLanguageModelBatch,
    HierarchicalTextCodec,
)
from tests.integration.test_hierarchical_language_model import language_model_config


def test_one_character_table_receives_both_learning_signals_and_roundtrips():
    torch.manual_seed(61)
    model = language_model_config().build().eval()
    character_table = model.encoder.byte_embedding.weight
    assert character_table is model.decoder.byte_embedding.weight
    assert character_table.shape == (256, 8)
    assert sum(parameter is character_table for parameter in model.parameters()) == 1

    encoded = model.encoder([["é"]]).hidden.square().sum()
    conditioning = torch.randn(1, 1, 8)
    decoded = (
        model.decoder(conditioning, torch.tensor([[[195, 169]]]), torch.tensor([[2]]))
        .logits.square()
        .sum()
    )
    encoder_gradient = torch.autograd.grad(encoded, character_table, retain_graph=True)[
        0
    ]
    decoder_gradient = torch.autograd.grad(decoded, character_table, retain_graph=True)[
        0
    ]
    assert encoder_gradient.abs().sum() > 0
    assert decoder_gradient.abs().sum() > 0
    (encoded + decoded).backward()
    torch.testing.assert_close(
        character_table.grad, encoder_gradient + decoder_gradient
    )

    restored = language_model_config().build().eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    assert (
        restored.encoder.byte_embedding.weight is restored.decoder.byte_embedding.weight
    )
    batch = HierarchicalLanguageModelBatch.collate(
        list(HierarchicalTextCodec(8).training_windows("é", 4))
    )
    torch.testing.assert_close(restored(batch).logits, model(batch).logits)


def test_protocol_and_training_targets_use_reserved_utf8_values():
    from emperor.embedding.hierarchical import (
        END_OF_DOCUMENT,
        END_OF_TOKEN,
        OUTPUT_SYMBOLS,
    )
    from tests.model_packages.test_gpt_hierarchical_language_model import (
        GPT_PACKAGES,
        configuration,
    )

    assert OUTPUT_SYMBOLS == 256
    assert END_OF_TOKEN == 254 and END_OF_DOCUMENT == 255
    examples = list(HierarchicalTextCodec(8).training_windows("é", 4))
    batch = HierarchicalLanguageModelBatch.collate(examples)
    assert batch.labels.tolist() == [195, 169, 254, 255]
    for name in GPT_PACKAGES:
        package, config = configuration(name)
        model = package.build_model(config).eval()
        output = model(batch)
        assert output.logits.shape == (4, 256)
        assert torch.isfinite(model._model_step(batch))


def test_checkpoint_rejects_legacy_protocol_and_conflicting_shared_weights():
    model = language_model_config().build()
    state = copy.deepcopy(model.state_dict())
    state.pop("_hat_protocol_version", None)
    with pytest.raises(RuntimeError, match="HAT.*protocol"):
        model.load_state_dict(state, strict=False)

    state = copy.deepcopy(model.state_dict())
    state["decoder.byte_embedding.weight"] = (
        state["encoder.byte_embedding.weight"].clone() + 1
    )
    with pytest.raises(RuntimeError, match="shared character"):
        model.load_state_dict(state)


def test_explicit_objective_configuration_reaches_each_gpt_experiment():
    from emperor.experiments.language_model import HierarchicalObjectiveConfig
    from tests.model_packages.test_gpt_hierarchical_language_model import (
        GPT_PACKAGES,
        configuration,
    )

    for name in GPT_PACKAGES:
        package, config = configuration(name)
        objective = HierarchicalObjectiveConfig(reduction="mean", auxiliary_weight=0.5)
        config.experiment_config.hierarchical_objective_config = objective
        model = package.build_model(config)
        assert model.hierarchical_handler.cfg == objective
        assert model.hierarchical_handler.cfg is not objective


@pytest.mark.parametrize("reduction, auxiliary_weight", [("sum", 0.0), ("mean", 2.0)])
def test_equation_five_sum_and_explicit_variants_keep_byte_only_metrics(
    reduction, auxiliary_weight
):
    from emperor.config import ModelConfig
    from emperor.embedding.hierarchical.decoder import HierarchicalByteDecoderOutput
    from emperor.experiments.language_model import (
        HierarchicalObjectiveConfig,
        LanguageModelExperiment,
    )

    class UniformModel(LanguageModelExperiment):
        def __init__(self):
            super().__init__(ModelConfig(learning_rate=0.01, output_dim=256))
            self.scores = torch.nn.Parameter(torch.zeros(256))

        def forward(self, batch):
            counts = torch.where(
                batch.attention_mask, batch.byte_lengths + 1, 0
            ).flatten()
            offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            return HierarchicalByteDecoderOutput(
                self.scores.expand(int(offsets[-1]), -1),
                offsets,
                self.scores.new_tensor(0.25),
            )

    model = UniformModel()
    model.hierarchical_handler = HierarchicalObjectiveConfig(
        reduction=reduction, auxiliary_weight=auxiliary_weight
    ).build()
    captured = {}
    for document in ("", "é"):
        batch = HierarchicalLanguageModelBatch.collate(
            list(HierarchicalTextCodec(8).training_windows(document, 4))
        )
        output = model._model_step_outputs(batch)
        reference = (
            output.logits.logsumexp(-1)
            - output.logits.gather(1, output.labels[:, None]).squeeze(1)
        ).sum()
        if reduction == "mean":
            reference = reference / output.labels.numel()
        torch.testing.assert_close(
            output.total_loss, reference + auxiliary_weight * 0.25
        )
        model.hierarchical_handler.log(
            lambda values, **kwargs: captured.update(values), "test", output
        )
    assert captured["test/bits_per_byte"].compute().item() == pytest.approx(8.0)
    assert captured["test/control_nll"].compute().item() == pytest.approx(
        3 * torch.log(torch.tensor(256.0)).item()
    )
    assert captured["test/symbol_nll"].compute().item() == pytest.approx(
        5 * torch.log(torch.tensor(256.0)).item()
    )


def test_each_live_gpt_factory_uses_reference_byte_blocks_and_dense_connectors():
    from emperor.transformer import SwiGLUFeedForward
    from tests.model_packages.test_gpt_hierarchical_language_model import (
        GPT_PACKAGES,
        configuration,
    )

    for name in GPT_PACKAGES:
        package, config = configuration(name, hidden_dim=12)
        model = package.build_model(config).eval().hierarchical_model
        assert model.encoder.byte_position is None
        assert model.decoder.byte_position is None
        for component in (model.encoder, model.decoder):
            assert any(
                isinstance(child, SwiGLUFeedForward) for child in component.modules()
            )
            assert not any(
                "bias" in name for name, parameter in component.named_parameters()
            )
        assert isinstance(model.encoder.encoder.encoder_layer_norm, torch.nn.RMSNorm)
        assert isinstance(model.decoder.decoder.decoder_layer_norm, torch.nn.RMSNorm)
        batch = HierarchicalLanguageModelBatch.collate(
            list(HierarchicalTextCodec(8).training_windows("a é", 4))
        )
        assert model(batch).logits.shape[-1] == 256


def test_reference_composition_matches_unpacked_equations_with_unequal_widths():
    from emperor.transformer import TransformerDecoderLayerState
    from emperor.transformer.hierarchical import hat_v2_reference_config

    cfg = hat_v2_reference_config(
        byte_dim=8,
        backbone_dim=12,
        byte_heads=2,
        backbone_heads=3,
        byte_layers=1,
        backbone_layers=1,
        max_token_bytes=8,
        sequence_length=4,
        batch_size=2,
    )
    torch.manual_seed(44)
    model = cfg.build().eval()
    examples = [
        {
            "context_texts": ["", "ab ", "é"],
            "target_texts": ["ab ", "é", None],
            "bos_mask": [True, False, False],
        }
    ]
    batch = HierarchicalLanguageModelBatch.collate(examples)
    actual = model(batch).logits
    word_vectors = [model.beginning_of_document]
    for word in ("ab ", "é"):
        ids = torch.tensor([[254, *word.encode("utf-8")]])
        encoded, _ = model.encoder.encoder(
            source_token_embeddings=model.encoder.byte_embedding(ids)
        )
        word_vectors.append(
            encoded[0, 0] @ model.encoder.projection.layers[0].model.weight_params
        )
    backbone = model.backbone(
        TransformerDecoderLayerState(hidden=torch.stack(word_vectors)[None])
    ).hidden
    backbone = model.output_normalization(backbone)[0]
    expected = []
    for index, word in enumerate(("ab ", "é", None)):
        context = (
            backbone[index]
            @ model.decoder.conditioning_projection.layers[0].model.weight_params
        )
        ids = torch.tensor(
            list(word.encode("utf-8")) if word is not None else [], dtype=torch.long
        )
        prefix = torch.cat((context[None], model.decoder.byte_embedding(ids)))[None]
        decoded, _ = model.decoder.decoder(target_token_embeddings=prefix)
        expected.append(
            decoded[0] @ model.decoder.output_projection.layers[0].model.weight_params
        )
    torch.testing.assert_close(actual, torch.cat(expected), atol=2e-5, rtol=2e-5)


def test_reference_generation_preserves_incomplete_utf8_and_learned_eow_at_cap():
    from emperor.embedding.hierarchical.decoder import ByteGenerationOptions
    from tests.support.hierarchical_decoding import decoder_config

    decoder = decoder_config().build().eval()
    with torch.no_grad():
        for parameter in decoder.output_projection.parameters():
            parameter.zero_()
        scores = decoder.output_projection.layers[0].model.bias_params
        scores[195] = 20
    result = decoder.generate_token(torch.zeros(6), max_new_bytes=1)
    assert result.data == b"\xc3"
    assert result.text is None and result.utf8_status == "incomplete"
    assert result.stop_reason == "byte_budget"
    constrained = decoder.generate_token(
        torch.zeros(6),
        max_new_bytes=1,
        options=ByteGenerationOptions(constrain_utf8=True),
    )
    assert constrained.text is not None
    with torch.no_grad():
        scores.zero_()
        scores[254] = 20
    completed = decoder.generate_token(
        torch.zeros(6), prefix=b"abcdefgh", max_new_bytes=1
    )
    assert completed.stop_reason == "end_of_token"
    assert completed.data == b"abcdefgh" and completed.new_bytes == 0


def test_truncation_keeps_partial_word_and_seeded_continuation_state():
    from emperor.embedding.hierarchical.decoder import ByteGenerationOptions

    model = language_model_config().build().eval()
    with torch.no_grad():
        for parameter in model.decoder.output_projection.parameters():
            parameter.zero_()
        model.decoder.output_projection.layers[0].model.bias_params[97:99] = 20
    options = ByteGenerationOptions(do_sample=True, top_k=2, seed=117)
    whole = model.generate_text("ab", max_new_bytes=4, options=options)
    partial = model.generate_text("ab", max_new_bytes=2, options=options)
    assert partial.new_tokens == 0
    assert partial.completed_tokens == ()
    assert partial.partial_token == partial.data
    resumed = model.generate_text(partial, max_new_bytes=2, options=options)
    assert resumed.data == whole.data
    assert resumed.partial_token == whole.partial_token


def test_raw_words_can_split_a_utf8_character_without_reencoding():
    from dataclasses import replace

    model = language_model_config().build().eval()
    symbols = iter([195, 254, 169, 254, 255])

    def scripted_symbols(module, inputs, output):
        logits = torch.zeros_like(output.logits)
        logits[-1, next(symbols)] = 20
        return replace(output, logits=logits)

    with model.decoder.register_forward_hook(scripted_symbols):
        result = model.generate_text("", max_new_bytes=8, max_new_tokens=3)
    assert result.completed_tokens == (b"\xc3", b"\xa9")
    assert result.data == b"\xc3\xa9" and result.text == "é"
    assert result.stop_reason == "end_of_document" and result.new_tokens == 2


def test_whitespace_reference_never_invents_boundaries_for_an_oversized_word():
    with pytest.raises(ValueError, match="max_token_bytes"):
        list(HierarchicalTextCodec(4).split_text("abcdefgh"))
    assert list(
        HierarchicalTextCodec(4, split_long_tokens=True).split_text("abcdefgh")
    ) == ["abcd", "efgh"]


@pytest.mark.parametrize("maximum, reason", [(8, "byte_budget"), (2, "byte_limit")])
def test_constrained_sampling_does_not_force_eow_at_a_budget(maximum, reason):
    from emperor.embedding.hierarchical.decoder import ByteGenerationOptions
    from tests.support.hierarchical_decoding import decoder_config

    decoder = decoder_config(max_bytes=maximum).build().eval()
    with torch.no_grad():
        for parameter in decoder.output_projection.parameters():
            parameter.zero_()
        decoder.output_projection.layers[0].model.bias_params[97] = 20
    output = decoder.generate_token(
        torch.zeros(6),
        prefix=b"a",
        max_new_bytes=1,
        options=ByteGenerationOptions(constrain_utf8=True),
    )
    assert output.data == b"aa"
    assert output.stop_reason == reason


def test_standalone_sampling_preserves_global_rng_for_a_discarded_boundary_draw():
    from emperor.embedding.hierarchical.decoder import ByteGenerationOptions
    from tests.support.hierarchical_decoding import decoder_config

    decoder = decoder_config().build().eval()
    with torch.no_grad():
        for parameter in decoder.output_projection.parameters():
            parameter.zero_()
        decoder.output_projection.layers[0].model.bias_params[97:99] = 20
    options = ByteGenerationOptions(do_sample=True, top_k=2)
    torch.manual_seed(41)
    whole = decoder.generate_token(torch.zeros(6), max_new_bytes=6, options=options)
    whole_state = torch.random.get_rng_state()
    torch.manual_seed(41)
    partial = decoder.generate_token(torch.zeros(6), max_new_bytes=3, options=options)
    resumed = decoder.generate_token(
        torch.zeros(6), prefix=partial.data, max_new_bytes=3, options=options
    )
    assert resumed.data == whole.data
    assert torch.equal(torch.random.get_rng_state(), whole_state)
