import torch
from typing import override
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToSpeech
from eam.encoder.Encoder import Encoder
from eam.encoder.EncoderConfig import EncoderConfig
from eam.encoder.registry import register


@register
class SonarSpaceEncoder(Encoder):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__(config)
        self.processor = AutoProcessor.from_pretrained(config['preprocessor'])
        self.model = SeamlessM4Tv2ForTextToSpeech.from_pretrained(config['embedder']).get_encoder().to(config['device'])

    @override
    def encode(self, batch) -> torch.Tensor:
        text  = batch[self.config['text_column']]

        preprocessed = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.config["device"])
        with torch.no_grad():
            out = self.model(**preprocessed).last_hidden_state
        return out

def get_default_config() -> EncoderConfig:
    return EncoderConfig(
        encoder_name = "SonarSpaceEncoder",
        audio_column = None,
        text_column = "transcript",
        preprocessor = "facebook/seamless-m4t-v2-large",
        embedder = "facebook/seamless-m4t-v2-large",
        sr = 16000,
        device = "cuda:0"
    )
