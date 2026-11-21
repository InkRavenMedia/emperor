"""Full-pass, single-source batches from the four ViT image datasets."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler, default_collate

from emperor.datasets._base import DataModule

from ._cifar_10 import Cifar10
from ._cifar_100 import Cifar100
from ._fashion_mnist import FashionMNIST
from ._mnist import Mnist


class JointImageBatch(NamedTuple):
    images: Tensor
    labels: Tensor
    dataset_id: str


class _SourceBatches(Dataset):
    """Each sampler key asks a single worker to load one homogeneous batch."""

    def __init__(self, splits: dict[str, Dataset]) -> None:
        self.splits = splits
        self.sizes = {name: len(split) for name, split in splits.items()}

    def __len__(self) -> int:
        return sum(self.sizes.values())

    def __getitem__(self, key: tuple[str, Sequence[int]]) -> JointImageBatch:
        name, indices = key
        if not indices:
            # A distributed peer can have no examples in the global tail.
            return JointImageBatch(
                torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long), name
            )
        images, labels = default_collate(
            [self.splits[name][index] for index in indices]
        )
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        return JointImageBatch(images, labels, name)


class _JointBatchSampler(DistributedSampler):
    """Draw uniformly among nonempty sources. Lightning sets the epoch on this sampler."""

    def __init__(self, data: _SourceBatches, batch_size: int, shuffle: bool, seed: int):
        distributed = torch.distributed.is_initialized()
        super().__init__(
            data,
            num_replicas=torch.distributed.get_world_size() if distributed else 1,
            rank=torch.distributed.get_rank() if distributed else 0,
            shuffle=shuffle,
            seed=seed,
        )
        self.sizes = data.sizes
        # Split each draw across ranks so the epoch schedule also replays across device counts.
        self.global_batch_size = batch_size

    def __len__(self) -> int:
        return sum(
            math.ceil(size / self.global_batch_size) for size in self.sizes.values()
        )

    def __iter__(self) -> Iterator[tuple[str, Sequence[int]]]:
        rng = random.Random(self.seed + self.epoch)
        remaining = {}
        for name, size in self.sizes.items():
            if size:
                remaining[name] = (
                    rng.sample(range(size), size) if self.shuffle else range(size)
                )
        offsets = dict.fromkeys(remaining, 0)
        while remaining:
            name = (
                rng.choice(list(remaining)) if self.shuffle else next(iter(remaining))
            )
            start = offsets[name]
            end = min(start + self.global_batch_size, self.sizes[name])
            yield name, remaining[name][start + self.rank : end : self.num_replicas]
            offsets[name] = end
            if end == self.sizes[name]:
                del remaining[name]


class JointImageClassification(DataModule):
    """One run input made from the existing train, validation, and test splits."""

    sources = {
        "mnist": Mnist,
        "fashion-mnist": FashionMNIST,
        "cifar10": Cifar10,
        "cifar100": Cifar100,
    }
    default_height = default_width = 32
    num_channels = 3
    flattened_input_dim = num_channels * default_height * default_width
    # Upper bound for the package's scalar output-dimension metadata.
    num_classes = max(source.num_classes for source in sources.values())

    def __init__(self, batch_size: int, seed: int | None = None):
        super().__init__()
        if batch_size <= 0:
            raise ValueError(
                "Joint image classification requires a positive batch size."
            )
        self.batch_size = batch_size
        self.seed = seed
        self._run_seed: int | None = None
        self._resolve_metadata(
            flattened_input_dim=self.flattened_input_dim, num_classes=self.num_classes
        )

    def _source(self, source_type, seed):
        padding = {"padding": 2} if source_type.num_channels == 1 else {}
        source = source_type(batch_size=self.batch_size, seed=seed, **padding)
        source.root = self.root
        return source

    def prepare_data(self) -> None:
        for source_type in self.sources.values():
            self._source(source_type, self.seed).prepare_data()

    def _seed_for_run(self) -> int:
        if self._run_seed is None:
            seed = self.seed if self.seed is not None else random.randrange(2**63)
            if torch.distributed.is_initialized():
                values = [seed]
                torch.distributed.broadcast_object_list(values, src=0)
                seed = values[0]
            self._run_seed = seed
        return self._run_seed

    def state_dict(self) -> dict[str, int]:
        return {"run_seed": self._run_seed} if self._run_seed is not None else {}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        seed = state_dict.get("run_seed")
        if seed is None or seed == self._run_seed:
            return
        self._run_seed = seed
        # Lightning sets up the splits before restoring data-module state.
        if hasattr(self, "train"):
            self._setup_fit()
        elif hasattr(self, "val"):
            self._setup_validate()

    def _setup_splits(self, stage: str, *names: str) -> None:
        splits = {name: {} for name in names}
        for dataset_id, source_type in self.sources.items():
            source = self._source(source_type, self._seed_for_run())
            source.setup(stage)
            for name in names:
                splits[name][dataset_id] = getattr(source, name)
        for name in names:
            setattr(self, name, _SourceBatches(splits[name]))

    def _setup_fit(self) -> None:
        self._setup_splits("fit", "train", "val")

    def _setup_validate(self) -> None:
        self._setup_splits("validate", "val")

    def _setup_test(self) -> None:
        self._setup_splits("test", "test")

    def _loader(self, split: _SourceBatches, shuffle: bool) -> DataLoader:
        return DataLoader(
            split,
            batch_size=None,
            sampler=_JointBatchSampler(
                split, self.batch_size, shuffle, self._seed_for_run()
            ),
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed)
            if self.seed is not None
            else None,
        )

    def get_dataloader(self, train: bool) -> DataLoader:
        return self._loader(self.train if train else self.val, train)

    def _get_test_dataloader(self) -> DataLoader:
        return self._loader(self.test, False)
