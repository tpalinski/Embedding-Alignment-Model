from .Encoder import Encoder
from .EncoderConfig import EncoderConfig


ENCODER_REGISTRY = {}

def register(cls):
    ENCODER_REGISTRY[cls.__name__] = cls
    return cls

def get_encoder(config: EncoderConfig) -> Encoder:
    cls = ENCODER_REGISTRY[config['encoder_name']]
    return cls(config)
