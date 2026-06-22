from abc import ABC, abstractmethod
from typing import TypedDict
from torch.utils.data import Dataset as TorchDataset, DataLoader, DistributedSampler, Sampler
import torch.distributed as dist
from enum import Enum

from .config import DatasetConfig

class DatasetSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

SplitsDict = dict[DatasetSplit, TorchDataset]

class Dataset(ABC):
    config: DatasetConfig
    """Dict containing train, validation and test splits for the dataset, needs to be seeded during construction"""
    splits: SplitsDict

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    """Get the string used for indexing into audio data per batch"""
    def get_audio_column(self) -> str|None:
        if self.config.__contains__("audio_column"):
            return self.config["audio_column"]
        else:
            return None

    """Get the string used for indexing into text data per batch"""
    def get_text_column(self) -> str|None:
        if self.config.__contains__("text_column"):
            return self.config["text_column"]
        else:
            return None

    """Get underlying torch Dataset for a split"""
    def get_split(self, split: DatasetSplit):
        return self.splits[split]

    @abstractmethod
    def get_train_sampler(self) -> Sampler | None:
        pass

    @abstractmethod
    def get_collate_function(self):
        pass

    """Get train, validation and test dataloader for the dataset"""
    def get_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        batch_size = self.config["batch_size"]
        shuffle = self.config["shuffle_train"]
        collate_fn = self.get_collate_function()
        sampler = self.get_train_sampler()
        if sampler is not None:
            train_loader = DataLoader(self.splits[DatasetSplit.TRAIN], batch_sampler=sampler, collate_fn=collate_fn)
        else:
            train_loader = DataLoader(self.splits[DatasetSplit.TRAIN], shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)
        val_loader = DataLoader(self.splits[DatasetSplit.VALIDATION], shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
        test_loader = DataLoader(self.splits[DatasetSplit.TEST], shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
        return train_loader, val_loader, test_loader

    """Get train, validation and test dataloader for the dataset for distributed training"""
    def get_loaders_distributed(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        batch_size = self.config["batch_size"]
        shuffle = self.config["shuffle_train"]
        train_sampler = DistributedSampler(
            self.splits[DatasetSplit.TRAIN],
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
        )
        val_sampler = DistributedSampler(
            self.splits[DatasetSplit.VALIDATION],
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        )
        test_sampler = DistributedSampler(
            self.splits[DatasetSplit.TEST],
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        )
        train_loader = DataLoader(
            self.splits[DatasetSplit.TRAIN],
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
        )
        val_loader = DataLoader(
            self.splits[DatasetSplit.VALIDATION],
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
        )
        test_loader = DataLoader(
            self.splits[DatasetSplit.TEST],
            batch_size=batch_size,
            sampler=test_sampler,
            shuffle=False,
        )
        return train_loader, val_loader, test_loader

