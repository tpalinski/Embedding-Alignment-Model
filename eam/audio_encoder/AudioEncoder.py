from abc import ABC, abstractmethod

from .AudioEncoderConfig import AudioEncoderConfig

class AudioEncoder(ABC):

    config: AudioEncoderConfig

    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.config = config


    @abstractmethod
    def encode_audio(self, batch):
        pass
