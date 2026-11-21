from __future__ import annotations

from dataclasses import fields, replace

import torch
from lightning import LightningModule
from torch import Tensor, nn

from emperor.config import ModelConfig
from emperor.datasets.image.classification import (
    JointImageBatch,
    JointImageClassification,
)
from emperor.experiments._auxiliary_loss import AuxiliaryLoss
from emperor.layers import Layer


class JointImageClassifierConfig(ModelConfig):
    """The selected ViT configuration with fixed, dataset-local output heads."""

    @classmethod
    def from_backbone(cls, config: ModelConfig) -> JointImageClassifierConfig:
        return cls(
            **{field.name: getattr(config, field.name) for field in fields(config)}
        )


class JointImageClassifier(LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.cfg = encoder.cfg
        head_config = encoder.experiment_config.output_config
        self.heads = nn.ModuleDict(
            {
                name: replace(head_config, output_dim=source.num_classes).build()
                for name, source in JointImageClassification.sources.items()
            }
        )
        # Each package's normal forward now returns its pooled CLS features
        # and optional encoder loss, including for expert and adaptive presets.
        self.encoder.output = nn.Identity()
        self.loss_fn = nn.CrossEntropyLoss()
        self._auxiliary_loss = AuxiliaryLoss("Joint image classification")
        self._source_indices = {name: index for index, name in enumerate(self.heads)}

    def forward(
        self, images: Tensor, dataset_id: str
    ) -> Tensor | tuple[Tensor, Tensor]:
        head = self.heads[dataset_id]
        output = self.encoder(images)
        features = output[0] if isinstance(output, tuple) else output
        if isinstance(head, Layer):
            logits = Layer.run_model_from_hidden(head, features).hidden
        else:
            logits = head(features)
        return (logits, output[1]) if isinstance(output, tuple) else logits

    def _step(self, batch: JointImageBatch) -> tuple[Tensor, Tensor]:
        images, labels, dataset_id = batch
        # Run the same head on every distributed peer even for a short tail.
        if labels.numel() == 0:
            images = images.new_zeros(1, *images.shape[1:])
        output = self(images, dataset_id)
        logits, auxiliary = output if isinstance(output, tuple) else (output, None)
        if labels.numel() == 0:
            return logits.sum() * 0, labels.new_zeros(())
        loss = self.loss_fn(logits, labels) + self._auxiliary_loss.resolve(
            auxiliary, reference=logits
        )
        return loss, (logits.argmax(dim=1) == labels).sum()

    def training_step(self, batch: JointImageBatch, batch_idx: int) -> Tensor:
        loss, correct = self._step(batch)
        count = batch.labels.numel()
        self.log_dict(
            {"train/loss": loss, "train/accuracy": correct / max(count, 1)},
            on_step=True,
            on_epoch=False,
            batch_size=count,
        )
        if torch.distributed.is_initialized():
            total = loss.new_tensor(count)
            torch.distributed.all_reduce(total)
            loss = loss * (count * torch.distributed.get_world_size() / total)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.encoder.learning_rate)

    def _reset_evaluation(self) -> None:
        # Keep local sums off the DDP broadcast-buffer path until epoch end.
        dtype = torch.float32 if self.device.type == "mps" else torch.float64
        self._evaluation_totals = torch.zeros(
            len(self.heads), 3, device=self.device, dtype=dtype
        )

    def on_validation_epoch_start(self) -> None:
        self._reset_evaluation()

    def on_test_epoch_start(self) -> None:
        self._reset_evaluation()

    def _evaluate(self, batch: JointImageBatch) -> Tensor:
        loss, correct = self._step(batch)
        count = batch.labels.numel()
        totals = self._evaluation_totals[self._source_indices[batch.dataset_id]]
        totals[0] += loss.detach().to(totals.dtype) * count
        totals[1] += correct
        totals[2] += count
        return loss

    def validation_step(self, batch: JointImageBatch, batch_idx: int) -> Tensor:
        return self._evaluate(batch)

    def test_step(self, batch: JointImageBatch, batch_idx: int) -> Tensor:
        return self._evaluate(batch)

    def _log_evaluation(self, stage: str) -> None:
        totals = self._evaluation_totals
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        metrics = {}
        for name, (loss_sum, correct, count) in zip(self.heads, totals, strict=True):
            if count > 0:
                metrics[f"{stage}/{name}/loss"] = loss_sum / count
                metrics[f"{stage}/{name}/accuracy"] = correct / count
        if (totals[:, 2] > 0).all():
            macro = (totals[:, 1] / totals[:, 2]).mean()
            metrics[f"{stage}/macro_accuracy"] = macro
            metrics[f"{stage}/accuracy"] = macro
        if metrics:
            metrics[f"{stage}/loss"] = totals[:, 0].sum() / totals[:, 2].sum()
            self.log_dict(metrics, on_epoch=True, batch_size=1)

    def on_validation_epoch_end(self) -> None:
        self._log_evaluation("validation")

    def on_test_epoch_end(self) -> None:
        self._log_evaluation("test")
