import torch
from typing import override
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor

from eam.encoder.registry import register
from eam.encoder.Encoder import Encoder
from eam.encoder.EncoderConfig import EncoderConfig


@register
class M4TEncoder(Encoder):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__(config)
        self.processor = AutoProcessor.from_pretrained(config['preprocessor'])
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(config['embedder'], dtype=torch.float16).get_encoder().to(config['device']).eval()
        print(f"M4T encoder loaded in: {next(self.model.parameters()).dtype}")

    @override
    def encode(self, batch) -> torch.Tensor:
        audio = batch[self.config['audio_column']]
        if isinstance(audio, torch.Tensor):
            audio = [a.cpu().numpy() for a in audio]
        preprocessed = self.processor(audio=audio, return_tensors="pt", sampling_rate=self.config['sr']).to(self.config['device'])
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            out = self.model(**preprocessed).last_hidden_state
        return out.cpu().float()

def get_default_config() -> EncoderConfig:
    return EncoderConfig(
        encoder_name = "M4TEncoder",
        audio_column = "audio_array",
        text_column = None,
        preprocessor = "facebook/seamless-m4t-v2-large",
        embedder = "facebook/seamless-m4t-v2-large",
        sr = 16000,
        device = "cuda:0"
    )
