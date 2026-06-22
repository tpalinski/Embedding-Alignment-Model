from dataclasses import dataclass
from typing import TypedDict
from eam.unsupervised_model import Data2VecModelWithSharedExtractorConfig

@dataclass
class AudioModelConfig(Data2VecModelWithSharedExtractorConfig):
    num_layers: int
    dim_ffd: int
    num_heads: int
    dropout: float
    device: str

def get_default_config() -> AudioModelConfig:
    return AudioModelConfig(
        num_layers=4,
        dim_ffd=1024,
        num_heads=8,
        K=2,
        mask_p=0.065,
        mask_steps=10,
        dropout=0.2,
        output_dims=768,
        feature_dims=768,
        device="cuda:1"
        )

