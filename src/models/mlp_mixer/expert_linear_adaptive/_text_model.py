"""Hierarchical masked-word Mixer with byte reconstruction and NSP heads."""

import torch
from torch import Tensor, nn
from torchmetrics.classification import BinaryAccuracy

from emperor.config import ModelConfig
from emperor.datasets.text.bert_pretraining import BertMaskedWordBatch
from emperor.embedding.hierarchical.decoder import (
    OUTPUT_SYMBOLS,
    HierarchicalByteDecoderOutput,
)
from emperor.experiments.bert_pretraining import BertPretrainingExperiment
from emperor.layers import LayerConfig, LayerNormPositionOptions, NormalizationOptions
from emperor.transformer import TransformerEncoderLayerState

from ._masked_word import greedy_word_matches
from .experiment_config import MaskedWordExperimentConfig


class MaskedWordModel(BertPretrainingExperiment):
    """Reuse the shared pretraining task, never a sibling Model Package."""

    def __init__(self, config: ModelConfig):
        experiment = config.experiment_config
        if not isinstance(experiment, MaskedWordExperimentConfig):
            raise TypeError(
                "MaskedWordModel requires a Mixer MaskedWordExperimentConfig"
            )
        super().__init__(config)
        self.experiment_config = experiment
        embedding = experiment.embedding_config
        decoder = experiment.masked_word_decoder_config
        if (
            embedding.output_dim != config.hidden_dim
            or decoder.conditioning_dim != config.hidden_dim
            or embedding.byte_embedding_dim != decoder.byte_embedding_dim
            or embedding.max_token_bytes != decoder.max_token_bytes
        ):
            raise ValueError(
                "Masked-word Mixer encoder and decoder dimensions must agree"
            )
        self.token_embedding = embedding.build()
        self.token_type_embedding = nn.Embedding(2, config.hidden_dim)
        self.positional_embedding = experiment.positional_embedding_config.build()
        self.embedding_layer_norm = self._normalization(NormalizationOptions.LAYER_NORM)
        self.embedding_dropout = nn.Dropout(experiment.embedding_dropout_probability)
        self.transformer = experiment.encoder_config.build()
        stack = getattr(
            experiment.encoder_config, "block_config", experiment.encoder_config
        )
        self.encoder_layer_norm = (
            self._normalization(experiment.encoder_output_normalization)
            if stack.layer_config.layer_model_config.layer_norm_position
            == LayerNormPositionOptions.BEFORE
            else nn.Identity()
        )
        self.word_decoder = decoder.build()
        self.word_decoder.byte_embedding = self.token_embedding.byte_embedding
        self.pooler = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.pooler_activation = nn.Tanh()
        self.nsp_head = nn.Linear(config.hidden_dim, 2)
        self.word_accuracy = nn.ModuleDict(
            {
                f"{stage}_words": BinaryAccuracy(ignore_index=-1)
                for stage in ("validation", "test")
            }
        )

    @property
    def mixer(self):
        return self.transformer

    def _normalization(self, normalization):
        return LayerConfig(
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.hidden_dim,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            normalization=normalization,
        ).build_normalization()

    def forward(self, batch: BertMaskedWordBatch):
        return self._masked_word_outputs(batch)[:3]

    def _masked_word_outputs(self, batch: BertMaskedWordBatch):
        if not isinstance(batch, BertMaskedWordBatch):
            raise TypeError("Masked-word Mixer requires a BertMaskedWordBatch")
        embedding = self.token_embedding(batch.input_words, batch.attention_mask)
        positions = batch.attention_mask.long()
        hidden = embedding.hidden + self.positional_embedding(positions)
        hidden = hidden + self.token_type_embedding(batch.token_type_ids)
        hidden = self.embedding_dropout(self.embedding_layer_norm(hidden))
        state = self.transformer(
            TransformerEncoderLayerState(
                hidden=hidden, key_padding_mask=~batch.attention_mask.bool()
            )
        )
        sequence_output = self.encoder_layer_norm(state.hidden)
        conditioning = sequence_output[batch.prediction_mask].unsqueeze(1)
        predictions = (
            self.word_decoder(conditioning, batch.byte_prefix_ids, batch.byte_lengths)
            if conditioning.shape[0]
            else HierarchicalByteDecoderOutput(
                conditioning.new_empty((0, OUTPUT_SYMBOLS)),
                batch.byte_lengths.new_zeros(1),
                conditioning.new_zeros(()),
            )
        )
        encoder_loss = state.loss if state.loss is not None else hidden.new_zeros(())
        auxiliary_loss = embedding.loss + encoder_loss + predictions.loss
        nsp_logits = self.nsp_head(
            self.pooler_activation(self.pooler(sequence_output[:, 0]))
        )
        return predictions, nsp_logits, auxiliary_loss, conditioning

    def training_step(self, batch: BertMaskedWordBatch, batch_idx: int) -> Tensor:
        return self._masked_word_step("train", batch)

    def validation_step(self, batch: BertMaskedWordBatch, batch_idx: int) -> Tensor:
        return self._masked_word_step("validation", batch)

    def test_step(self, batch: BertMaskedWordBatch, batch_idx: int) -> Tensor:
        return self._masked_word_step("test", batch)

    def _masked_word_step(self, stage: str, batch: BertMaskedWordBatch) -> Tensor:
        batch = batch.to(self.device)
        predictions, nsp_logits, auxiliary_loss, conditioning = (
            self._masked_word_outputs(batch)
        )
        byte_loss = (
            nn.functional.cross_entropy(predictions.logits, batch.byte_labels)
            if batch.byte_labels.numel()
            else conditioning.sum() * 0.0
        )
        nsp_loss = self.nsp_loss_fn(nsp_logits, batch.next_sentence_labels)
        loss = byte_loss + nsp_loss + auxiliary_loss
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/mlm/byte_reconstruction_loss": byte_loss,
                f"{stage}/nsp/loss": nsp_loss,
                f"{stage}/nsp/accuracy": (
                    nsp_logits.argmax(-1) == batch.next_sentence_labels
                )
                .float()
                .mean(),
                f"{stage}/auxiliary/loss": auxiliary_loss,
            },
            batch_size=batch.attention_mask.shape[0],
            sync_dist=True,
        )
        self.log(
            f"{stage}/mlm/oversized_words",
            float(batch.oversized_word_count),
            on_step=False,
            on_epoch=True,
            reduce_fx="sum",
            sync_dist=True,
        )
        if stage != "train":
            matches = greedy_word_matches(
                self.word_decoder,
                conditioning,
                batch.byte_prefix_ids,
                batch.byte_lengths,
            ).long()
            metric = self.word_accuracy[f"{stage}_words"]
            if matches.numel():
                metric.update(matches, torch.ones_like(matches))
            else:
                metric.update(matches.new_zeros(1), matches.new_full((1,), -1))
            self.log(f"{stage}/mlm/exact_word_accuracy", metric, on_epoch=True)
        return loss
