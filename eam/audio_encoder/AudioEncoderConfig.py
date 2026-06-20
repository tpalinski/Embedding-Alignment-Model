from typing import TypedDict


class AudioEncoderConfig(TypedDict):
    audio_column: str
    preprocessor: str
    embedder: str
    sr: int
    device: str
