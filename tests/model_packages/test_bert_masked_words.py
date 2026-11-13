import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import torch
from datasets import Dataset

from emperor.datasets.text.bert_pretraining import (
    BertMaskedWordBatch,
    WikiText2BertHierarchical,
    WikiText103BertHierarchical,
)
from emperor.embedding.hierarchical.decoder import (
    END_OF_DOCUMENT,
    END_OF_TOKEN,
    OUTPUT_SYMBOLS,
    HierarchicalByteDecoderOutput,
)
from emperor.experiments import ExperimentTask
from models.bert.expert_linear_adaptive._masked_word import greedy_word_matches
from models.bert.expert_linear_adaptive.model import Model
from models.bert.expert_linear_adaptive.presets import ExperimentPresets
from models.catalog import model_package


def masked_batch(first=("Café",), second=("dog",)):
    return BertMaskedWordBatch.collate(
        [(first, second, 0)],
        sequence_length=8,
        max_token_bytes=6,
        mlm_probability=1.0,
    )


class BertMaskedWordTests(unittest.TestCase):
    def test_targets_are_aligned_and_isolated_from_encoder_words(self):
        batch = masked_batch(("Café", "elephant", ","), ("dog",))
        self.assertEqual(
            batch.input_words[0],
            ("[CLS]", "[MASK]", "[UNK]", ",", "[SEP]", "[MASK]", "[SEP]", "[PAD]"),
        )
        self.assertEqual(batch.prediction_mask.nonzero().tolist(), [[0, 1], [0, 5]])
        self.assertEqual(batch.attention_mask.tolist(), [[True] * 7 + [False]])
        self.assertEqual(batch.token_type_ids.tolist(), [[0, 0, 0, 0, 0, 1, 1, 0]])
        self.assertEqual(batch.oversized_word_count, 1)
        self.assertEqual(
            batch.byte_labels.tolist(),
            [*"Café".encode(), END_OF_TOKEN, *b"dog", END_OF_TOKEN],
        )
        self.assertEqual(batch.byte_lengths.flatten().tolist(), [5, 3])
        alternate = masked_batch(("tree", "elephant", ","), ("cat",))
        self.assertEqual(batch.input_words, alternate.input_words)
        torch.testing.assert_close(batch.prediction_mask, alternate.prediction_mask)

    def test_disk_cache_and_document_boundaries(self):
        source = Dataset.from_dict(
            {
                "text": [
                    " = First = ",
                    "Café runs. A dog doesn’t stop.",
                    " = Second = ",
                    "A cat jumps. It sleeps.",
                ]
            }
        )

        class LocalWikiText(WikiText103BertHierarchical):
            def _dataset(self, split):
                return source

        with tempfile.TemporaryDirectory() as root:
            data = LocalWikiText(
                root=root, num_workers=0, sequence_length=8, random_next_probability=0.0
            )
            data.setup("fit")
            data.setup("test")
            first, second, label = data.train[0]
            self.assertEqual((first[0], second[0], label), ("Café", "A", 0))
            self.assertIn("doesn’t", second)
            self.assertEqual(data.train[1][2], 1)
            self.assertEqual(data.train[2][2], 0)
            cache_files = [
                split.sentences.cache_files[0]["filename"]
                for split in (data.train, data.val, data.test)
            ]
            self.assertEqual(len(set(cache_files)), 3)
            self.assertTrue(all(Path(name).is_file() for name in cache_files))
            self.assertIsInstance(
                next(iter(data.val_dataloader())), BertMaskedWordBatch
            )

    def test_wikitext2_loads_raw_splits_as_masked_word_batches(self):
        package = model_package("bert/expert_linear_adaptive")
        (dataset,) = package.resolve_datasets(
            ["wiki-text2-bert-hierarchical"], ExperimentTask.BERT_PRETRAINING
        )
        self.assertIs(dataset, WikiText2BertHierarchical)
        source = Dataset.from_dict({"text": ["A cat jumps. It sleeps."]})
        with (
            tempfile.TemporaryDirectory() as root,
            patch(
                "emperor.datasets.text.bert_pretraining._wiki_text_103.load_dataset",
                return_value=source,
            ) as load_dataset,
        ):
            data = dataset(root=root, num_workers=0)
            data.setup()
            self.assertEqual(
                load_dataset.call_args_list,
                [
                    call(
                        "Salesforce/wikitext",
                        "wikitext-2-raw-v1",
                        split=split,
                        cache_dir=root,
                    )
                    for split in ("train", "validation", "test")
                ],
            )
            for loader in (
                data.train_dataloader(),
                data.val_dataloader(),
                data.test_dataloader(),
            ):
                self.assertIsInstance(next(iter(loader)), BertMaskedWordBatch)

    def test_byte_loss_backpropagates_and_checkpoint_restores_shared_weights(self):
        (config,) = ExperimentPresets().get_config(
            dataset=WikiText103BertHierarchical,
            config_overrides={
                "batch_size": 1,
                "sequence_length": 8,
                "hierarchical_max_token_bytes": 6,
            },
        )
        model = Model(config).eval()
        batch = masked_batch()
        output, nsp, aux = model(batch)
        self.assertEqual(nsp.shape, (1, 2))
        self.assertTrue(
            torch.isfinite(model.nsp_loss_fn(nsp, batch.next_sentence_labels))
        )
        self.assertTrue(torch.isfinite(aux))
        self.assertEqual(output.token_offsets.tolist(), [0, 6, 10])
        torch.nn.functional.cross_entropy(output.logits, batch.byte_labels).backward()
        for module in (
            model.token_embedding.projection,
            model.transformer,
            model.word_decoder.output_projection,
        ):
            self.assertTrue(
                any(
                    parameter.grad is not None and parameter.grad.abs().sum() > 0
                    for parameter in module.parameters()
                )
            )

        checkpoint = io.BytesIO()
        torch.save({"state_dict": model.state_dict()}, checkpoint)
        checkpoint.seek(0)
        restored = Model(config).eval()
        restored.load_state_dict(
            torch.load(checkpoint, weights_only=True)["state_dict"]
        )
        self.assertIs(
            restored.token_embedding.byte_embedding,
            restored.word_decoder.byte_embedding,
        )
        with torch.no_grad():
            torch.testing.assert_close(restored(batch)[0].logits, output.logits)
            empty = masked_batch(("!",), ("?",))
            self.assertEqual(restored(empty)[0].logits.shape, (0, OUTPUT_SYMBOLS))
            with (
                patch.object(restored, "log_dict"),
                patch.object(restored, "log"),
            ):
                self.assertTrue(torch.isfinite(restored.validation_step(empty, 0)))
                self.assertEqual(
                    restored.word_accuracy["validation_words"].compute().item(), 0.0
                )

    def test_exact_word_generation_requires_spelling_and_end_of_word(self):
        batch = BertMaskedWordBatch.collate(
            [(("dog", "cat"), ("dog",), 0)],
            sequence_length=8,
            max_token_bytes=6,
            mlm_probability=1.0,
        )

        def decoder(conditioning, prefixes, lengths, active):
            words = b"dog"
            counts = torch.where(active, lengths + 1, 0).flatten()
            offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            logits = torch.zeros((offsets[-1], OUTPUT_SYMBOLS))
            for row in active.flatten().nonzero().flatten().tolist():
                length = lengths[row, 0].item()
                self.assertEqual(
                    bytes(prefixes[row, 0, :length].tolist()), words[:length]
                )
                stop = END_OF_DOCUMENT if row == 2 else END_OF_TOKEN
                symbol = words[length] if length < len(words) else stop
                logits[offsets[row + 1] - 1, symbol] = 1.0
            return HierarchicalByteDecoderOutput(logits, offsets, logits.new_zeros(()))

        matches = greedy_word_matches(
            decoder, torch.zeros((3, 1, 2)), batch.byte_prefix_ids, batch.byte_lengths
        )
        self.assertEqual(matches.tolist(), [True, False, False])
