from .Dataset import Dataset
from .config import DatasetConfig

DATASET_REGISTRY = {}

def register(cls):
    DATASET_REGISTRY[cls.__name__] = cls
    return cls

def get_dataset(config: DatasetConfig) -> Dataset:
    cls = DATASET_REGISTRY[config['dataset_name']]
    return cls(config)
