import json
import math
from collections import defaultdict

import pytest
import torch
from torch.nn.functional import cross_entropy

from emperor.datasets.image.classification import (
    JointImageBatch,
    JointImageClassification,
)
from model_runtime.runs import RunRequest, plan_runs
from models.catalog import model_package
from models.project_cli.main import run_experiment
from tests.support.dataset_metadata import offline_dataset_metadata

TASK = "joint-image-classification"
SOURCES = ("mnist", "fashion-mnist", "cifar10", "cifar100")
VARIANTS = ("linear", "linear_adaptive", "expert_linear", "expert_linear_adaptive")
SMALL_MODEL = {"image_patch_size": 16, "stack_num_layers": 1}


@pytest.mark.parametrize("variant", VARIANTS)
def test_joint_task_plans_one_fit_and_ordinary_selection_plans_four(variant):
    package = model_package(f"vit/{variant}")
    joint = plan_runs(
        package, RunRequest(presets=("baseline",), datasets=(), experiment_task=TASK)
    )
    ordinary = plan_runs(package, RunRequest(presets=("baseline",), datasets=SOURCES))
    assert [(run.experiment_task, run.dataset) for run in joint.runs] == [
        (TASK, "JointImageClassification")
    ]
    assert [run.dataset for run in ordinary.runs] == [
        "Mnist",
        "FashionMNIST",
        "Cifar10",
        "Cifar100",
    ]


def test_full_epochs_and_padding(tmp_path):
    with offline_dataset_metadata(JointImageClassification, tmp_path, seed=29) as data:
        data.batch_size = 7
        data.setup("fit")
        loader = data.train_dataloader()
        sizes = {name: len(split) for name, split in data.train.splits.items()}
        expected_batches = sum(math.ceil(n / data.batch_size) for n in sizes.values())
        schedules = []
        for epoch in (0, 1, 0):
            loader.sampler.set_epoch(epoch)
            schedule = list(loader.sampler)
            by_source = defaultdict(list)
            for name, indices in schedule:
                assert 0 < len(indices) <= data.batch_size
                by_source[name].extend(indices)
            assert len(schedule) == len(loader) == expected_batches
            for name, size in sizes.items():
                assert sorted(by_source[name]) == list(range(size))
            schedules.append(schedule)
        assert schedules[0] == schedules[2]
        assert schedules[0] != schedules[1]

        for batch, (name, indices) in zip(loader, schedules[0], strict=True):
            assert batch.dataset_id == name
            assert batch.images.shape == (len(indices), 3, 32, 32)
            if name in ("mnist", "fashion-mnist"):
                torch.testing.assert_close(batch.images[:, 0], batch.images[:, 2])
                mean, std = (0.1307, 0.3081) if name == "mnist" else (0.2860, 0.3530)
                assert torch.allclose(
                    batch.images[..., :2, :], torch.tensor(-mean / std)
                )


def test_joint_configuration_enforces_the_input_shape():
    package = model_package("vit/linear")
    with pytest.raises(ValueError, match="input_channels=3"):
        package.build_configuration(
            dataset=JointImageClassification, config_overrides={"input_channels": 1}
        )


def test_unseeded_checkpoint_reuses_the_training_split_and_next_draw(tmp_path):
    with offline_dataset_metadata(
        JointImageClassification, tmp_path, seed=None
    ) as original:
        original.setup("fit")
        state = original.state_dict()
        original_loader = original.train_dataloader()
        original_loader.sampler.set_epoch(1)
        expected_schedule = list(original_loader.sampler)
        expected_indices = {
            name: split.indices for name, split in original.train.splits.items()
        }
        resumed = JointImageClassification(batch_size=original.batch_size)
        resumed.root = str(tmp_path)
        resumed.setup("fit")
        resumed.load_state_dict(state)
        assert {
            name: split.indices for name, split in resumed.train.splits.items()
        } == expected_indices
        sampler = resumed.train_dataloader().sampler
        sampler.set_epoch(1)
        assert list(sampler) == expected_schedule


def test_distributed_shards_share_the_draw_and_cover_each_tail(tmp_path, monkeypatch):
    with offline_dataset_metadata(JointImageClassification, tmp_path, seed=3) as data:
        data.batch_size = 7
        data.setup("fit")
        single_device_batches = list(data.train_dataloader().sampler)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 3)
        schedules = []
        for rank in range(3):
            monkeypatch.setattr(torch.distributed, "get_rank", lambda rank=rank: rank)
            schedules.append(list(data.train_dataloader().sampler))
        visited = defaultdict(list)
        for (source, batch), group in zip(
            single_device_batches, zip(*schedules, strict=True), strict=True
        ):
            for rank, (name, indices) in enumerate(group):
                assert name == source
                assert list(indices) == list(batch[rank::3])
                visited[name].extend(indices)
        for name, split in data.train.splits.items():
            assert sorted(visited[name]) == list(range(len(split)))
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
        # MNIST has just two validation examples in this fixture.
        empty = next(iter(data.val_dataloader()))
        assert empty.dataset_id == "mnist" and empty.images.shape == (0, 3, 32, 32)


def test_encoder_auxiliary_loss_reaches_backward(monkeypatch):
    package = model_package("vit/linear")
    model = package.build_model(
        package.build_configuration(
            dataset=JointImageClassification, config_overrides=SMALL_MODEL
        )
    )
    model.eval()
    forward = model.encoder.forward
    auxiliary = torch.tensor(0.7, requires_grad=True)
    monkeypatch.setattr(
        model.encoder, "forward", lambda images: (forward(images), auxiliary)
    )
    batch = JointImageBatch(torch.zeros(2, 3, 32, 32), torch.tensor([1, 2]), "mnist")
    logits, returned_auxiliary = model(batch.images, batch.dataset_id)
    assert returned_auxiliary is auxiliary
    loss, _ = model._step(batch)
    torch.testing.assert_close(loss, cross_entropy(logits, batch.labels) + auxiliary)
    loss.backward()
    assert auxiliary.grad.item() == 1


@pytest.mark.training
@pytest.mark.parametrize("variant", VARIANTS)
def test_head_selection_and_optimizer_isolation(variant):
    torch.manual_seed(13)
    package = model_package(f"vit/{variant}")
    config = package.build_configuration(
        dataset=JointImageClassification, config_overrides=SMALL_MODEL
    )
    model = package.build_model(config)
    optimizer = model.configure_optimizers()
    images = torch.randn(2, 3, 32, 32)
    for name in SOURCES:
        output = model(images, name)
        logits = output[0] if isinstance(output, tuple) else output
        assert logits.shape == (2, JointImageClassification.sources[name].num_classes)

    # Check a previously active head too: Adam already owns momentum for it.
    for name in ("mnist", "cifar100"):
        optimizer.zero_grad(set_to_none=True)
        before = {
            key: [p.detach().clone() for p in head.parameters()]
            for key, head in model.heads.items()
        }
        labels = torch.tensor(
            [0, JointImageClassification.sources[name].num_classes - 1]
        )
        loss, _ = model._step(JointImageBatch(images, labels, name))
        loss.backward()
        assert any(
            p.grad is not None and torch.count_nonzero(p.grad)
            for p in model.encoder.transformer.parameters()
        )
        optimizer.step()
        for key, head in model.heads.items():
            changed = any(
                not torch.equal(value, p)
                for value, p in zip(before[key], head.parameters(), strict=True)
            )
            assert changed == (key == name)
            if key != name:
                assert all(p.grad is None for p in head.parameters())


@pytest.mark.training
def test_cli_epoch_checkpoint_and_reported_scores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with offline_dataset_metadata(JointImageClassification, tmp_path, seed=17) as data:
        data.batch_size = 16
        assert (
            run_experiment(
                [
                    "--model-type",
                    "vit",
                    "--model",
                    "linear",
                    "--preset",
                    "baseline",
                    "--experiment-task",
                    TASK,
                    "--config",
                    "--num-epochs",
                    "1",
                    "--batch-size",
                    "16",
                    "--image-patch-size",
                    "16",
                    "--stack-num-layers",
                    "1",
                    "--seed",
                    "17",
                    "--data-num-workers",
                    "0",
                ]
            )
            == 0
        )
        (receipt,) = (tmp_path / "logs").rglob("result.json")
        result = json.loads(receipt.read_text())
        assert result["experimentTask"] == TASK
        checkpoint = torch.load(
            receipt.parent / "checkpoints" / "last.ckpt", weights_only=True
        )
        assert len(checkpoint["optimizer_states"]) == 1
        assert checkpoint["JointImageClassification"]["run_seed"] == 17
        data.setup("fit")
        assert checkpoint["global_step"] == len(data.train_dataloader())

        package = model_package("vit/linear")
        model = package.build_model(
            package.build_configuration(
                dataset=JointImageClassification, config_overrides=SMALL_MODEL
            )
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        data.setup("test")
        metrics = result["metrics"]
        with torch.no_grad():
            for stage, loader in (
                ("validation", data.val_dataloader()),
                ("test", data.test_dataloader()),
            ):
                totals = {key: [0, 0.0, 0] for key in SOURCES}
                for images, labels, dataset_id in loader:
                    logits = model(images, dataset_id)
                    totals[dataset_id][0] += (logits.argmax(1) == labels).sum().item()
                    totals[dataset_id][1] += cross_entropy(
                        logits, labels, reduction="sum"
                    ).item()
                    totals[dataset_id][2] += len(labels)
                for name, (correct, loss, count) in totals.items():
                    assert metrics[f"{stage}/{name}/accuracy"] == pytest.approx(
                        correct / count
                    )
                    assert metrics[f"{stage}/{name}/loss"] == pytest.approx(
                        loss / count, rel=1e-5
                    )
                expected_macro = (
                    sum(correct / n for correct, _, n in totals.values()) / 4
                )
                assert metrics[f"{stage}/macro_accuracy"] == pytest.approx(
                    expected_macro
                )
