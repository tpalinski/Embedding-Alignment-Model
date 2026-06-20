from abc import ABC, abstractmethod
import torch
from torch import nn

from .EncoderConfig import EncoderConfig

class Encoder(ABC, nn.Module):

    config: EncoderConfig

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config


    @abstractmethod
    def encode(self, batch) -> torch.Tensor:
        pass
