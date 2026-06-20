from typing import TypedDict


class EncoderConfig(TypedDict):
    encoder_name: str
    audio_column: str|None
    text_column: str|None
    preprocessor: str
    embedder: str
    sr: int|None
    device: str
