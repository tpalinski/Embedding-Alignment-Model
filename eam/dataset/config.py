from dataclasses import dataclass
from typing import TypedDict


@dataclass
class DatasetConfig(TypedDict):
    dataset_name: str
    audio_column: str | None
    text_column: str | None
    dataset_path: str
    batch_size: int
    shuffle_train: bool
