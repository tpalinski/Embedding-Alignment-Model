import torch
from typing import override
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor

from eam.audio_encoder.AudioEncoder import AudioEncoder
from eam.audio_encoder.AudioEncoderConfig import AudioEncoderConfig


class M4TEncoder(AudioEncoder):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__(config)
        self.processor = AutoProcessor.from_pretrained(config['preprocessor'])
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(config['embedder']).get_encoder().to(config['device'])

    @override
    def encode_audio(self, batch):
        audio = batch[self.config['audio_column']]
        preprocessed = self.processor(audio=audio, return_tensors="pt", sampling_rate=self.config['sr']).to(self.config['device'])
        with torch.no_grad():
            out = self.model(**preprocessed).last_hidden_state
        return out
