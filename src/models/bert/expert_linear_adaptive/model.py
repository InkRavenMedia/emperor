from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.classification import BinaryAccuracy

from emperor.datasets.text.bert_pretraining import BertMaskedWordBatch
from emperor.embedding.hierarchical.decoder import (
    OUTPUT_SYMBOLS,
    HierarchicalByteDecoderOutput,
)
from emperor.layers import (
    LayerState,
)
from models import _sequence_expansion
from models.bert.expert_linear_adaptive._boundary_config_factory import (
    BertBoundaryConfig,
)
from models.bert.expert_linear_adaptive._masked_word import greedy_word_matches
from models.bert.expert_linear_adaptive.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig

from models.bert._model import BertBackbone


class Model(BertBackbone[ExperimentConfig, BertBoundaryConfig]):
    experiment_type = ExperimentConfig
    boundary_type = BertBoundaryConfig
    invalid_experiment_message = "config.experiment_config must be a BERT Expert Linear Adaptive ExperimentConfig."

    def __init__(
        self,
        config: "ModelConfig",
    ):
        experiment_config = self._validate_experiment_config(config)
        boundary_config = self._validate_boundary_config(experiment_config)
        if (
            experiment_config.positional_embedding_config is None
            and boundary_config.embedding_options.contextual_embedding_config is None
        ):
            raise ValueError(
                "experiment_config.positional_embedding_config is required"
            )
        if experiment_config.encoder_config is None:
            raise ValueError("experiment_config.encoder_config is required")
        super().__init__(config)
        self.experiment_config: ExperimentConfig = experiment_config
        self.sequence_expansion_enabled = (
            _sequence_expansion.expanded_sequence_capacity(
                self.experiment_config.encoder_config, 0
            )
            > 0
        )
        self.boundary_config: BertBoundaryConfig = boundary_config
        self._validate_pretraining_dimensions()
        decoder_config = experiment_config.masked_word_decoder_config
        if decoder_config is None:
            self.token_embedding = self._build_token_embedding()
        else:
            encoder_config = (
                boundary_config.embedding_options.hierarchical_embedding_config
            )
            if (
                encoder_config is None
                or encoder_config.output_dim != config.hidden_dim
                or decoder_config.conditioning_dim != config.hidden_dim
                or encoder_config.byte_embedding_dim
                != decoder_config.byte_embedding_dim
                or encoder_config.max_token_bytes != decoder_config.max_token_bytes
            ):
                raise ValueError(
                    "Masked-word BERT encoder and decoder dimensions must agree."
                )
            self.token_embedding = encoder_config.build()
        self.token_type_embedding = self._build_token_type_embedding()
        self.positional_embedding = self._build_positional_embedding()
        self.embedding_layer_norm = self._build_embedding_layer_norm()
        self.embedding_dropout = self._build_embedding_dropout()
        self.transformer = self._build_encoder_model()
        self.encoder_layer_norm = self._build_encoder_layer_norm()
        self.word_decoder = None
        if decoder_config is not None:
            self.word_decoder = decoder_config.build()
        if self.word_decoder is None:
            self.mlm_dense = self._build_mlm_dense()
            self.mlm_activation = self._build_mlm_activation()
            self.mlm_layer_norm = self._build_mlm_layer_norm()
            self.mlm_decoder = self._build_mlm_decoder()
            self._tie_mlm_decoder_weights()
            self.mlm_decoder_bias = self._build_mlm_decoder_bias()
        else:
            self.word_decoder.byte_embedding = self.token_embedding.byte_embedding
            self.word_accuracy = nn.ModuleDict(
                {
                    f"{stage}_words": BinaryAccuracy(ignore_index=-1)
                    for stage in ("validation", "test")
                }
            )
        self.pooler = self._build_pooler()
        self.pooler_activation = self._build_pooler_activation()
        self.nsp_head = self._build_nsp_head()

    def forward(
        self,
        input_ids: Tensor | BertMaskedWordBatch,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        align_to_input: bool = False,
    ) -> tuple[Tensor | HierarchicalByteDecoderOutput, Tensor, Tensor]:
        """Return vocabulary logits or packed masked-word byte predictions."""
        if isinstance(input_ids, BertMaskedWordBatch):
            predictions, nsp_logits, auxiliary_loss, _ = self.__masked_word_outputs(
                input_ids
            )
            return predictions, nsp_logits, auxiliary_loss
        if self.word_decoder is not None:
            raise TypeError(
                "Hierarchical masked-word BERT requires BertMaskedWordBatch."
            )
        input_ids, attention_mask, token_type_ids = self._prepare_model_inputs(
            input_ids,
            attention_mask,
            token_type_ids,
            token_type_vocab_size=self.token_type_embedding.num_embeddings,
        )
        hidden, embedding_loss = self.__build_input_embeddings(
            input_ids, token_type_ids, attention_mask
        )
        sequence_output, auxiliary_loss = self._run_encoder(
            hidden, attention_mask, align_to_input
        )
        mlm_logits = self._build_mlm_logits(sequence_output)
        nsp_logits = self._build_nsp_logits(sequence_output)
        total_auxiliary_loss = LayerState.sum_losses(
            auxiliary_loss,
            embedding_loss,
            LayerState.module_loss(self.positional_embedding),
        )
        return mlm_logits, nsp_logits, total_auxiliary_loss

    def training_step(
        self, batch: tuple[Tensor, ...] | BertMaskedWordBatch, batch_idx: int
    ) -> Tensor:
        if isinstance(batch, BertMaskedWordBatch):
            return self.__masked_word_step("train", batch)
        return super().training_step(batch, batch_idx)

    def validation_step(
        self, batch: tuple[Tensor, ...] | BertMaskedWordBatch, batch_idx: int
    ) -> Tensor:
        if isinstance(batch, BertMaskedWordBatch):
            return self.__masked_word_step("validation", batch)
        return super().validation_step(batch, batch_idx)

    def test_step(
        self, batch: tuple[Tensor, ...] | BertMaskedWordBatch, batch_idx: int
    ) -> Tensor:
        if isinstance(batch, BertMaskedWordBatch):
            return self.__masked_word_step("test", batch)
        return super().test_step(batch, batch_idx)

    def __masked_word_step(self, stage: str, batch: BertMaskedWordBatch) -> Tensor:
        batch = batch.to(self.device)
        predictions, nsp_logits, auxiliary_loss, conditioning = (
            self.__masked_word_outputs(batch)
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
            self.__log_exact_word_accuracy(stage, conditioning, batch)
        return loss

    def __log_exact_word_accuracy(
        self, stage: str, conditioning: Tensor, batch: BertMaskedWordBatch
    ) -> None:
        matches = greedy_word_matches(
            self.word_decoder, conditioning, batch.byte_prefix_ids, batch.byte_lengths
        ).long()
        metric = self.word_accuracy[f"{stage}_words"]
        if matches.numel():
            metric.update(matches, torch.ones_like(matches))
        else:
            # An ignored observation records zero denominator for this batch.
            metric.update(matches.new_zeros(1), matches.new_full((1,), -1))
        self.log(f"{stage}/mlm/exact_word_accuracy", metric, on_epoch=True)

    def __masked_word_outputs(self, batch: BertMaskedWordBatch):
        if self.word_decoder is None:
            raise TypeError(
                "BertMaskedWordBatch requires hierarchical masked-word mode."
            )
        embedding = self.token_embedding(batch.input_words, batch.attention_mask)
        padding_idx = self.experiment_config.positional_embedding_config.padding_idx
        position_tokens = torch.where(
            batch.attention_mask, 0 if padding_idx == 1 else 1, padding_idx or 0
        )
        hidden = self.__position_and_normalize(
            embedding.hidden, position_tokens, batch.token_type_ids
        )
        sequence_output, encoder_loss = self._run_encoder(
            hidden, batch.attention_mask, self.sequence_expansion_enabled
        )
        conditioning = sequence_output[batch.prediction_mask].unsqueeze(1)
        if conditioning.shape[0]:
            predictions = self.word_decoder(
                conditioning, batch.byte_prefix_ids, batch.byte_lengths
            )
        else:
            predictions = HierarchicalByteDecoderOutput(
                conditioning.new_empty((0, OUTPUT_SYMBOLS)),
                batch.byte_lengths.new_zeros(1),
                conditioning.new_zeros(()),
            )
        masked_word_loss: Tensor = embedding.loss + encoder_loss + predictions.loss
        auxiliary_loss = LayerState.sum_losses(
            masked_word_loss, LayerState.module_loss(self.positional_embedding)
        )
        return (
            predictions,
            self._build_nsp_logits(sequence_output),
            auxiliary_loss,
            conditioning,
        )

    def __build_input_embeddings(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        embedding_state = self._build_token_embedding_state(input_ids, attention_mask)
        return self.__position_and_normalize(
            embedding_state.hidden, input_ids, token_type_ids
        ), embedding_state.loss

    def __position_and_normalize(
        self, token_embedding: Tensor, input_ids: Tensor, token_type_ids: Tensor
    ) -> Tensor:
        positioned_token_embeddings = self._add_sentence_positions(
            token_embedding, input_ids
        )
        segment_embedding = self.token_type_embedding(token_type_ids)
        hidden = positioned_token_embeddings + segment_embedding
        hidden = self.embedding_layer_norm.before_model(
            LayerState(hidden=hidden)
        ).hidden
        return self.embedding_dropout(hidden)
