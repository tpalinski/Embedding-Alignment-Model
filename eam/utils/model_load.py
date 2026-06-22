import json
from pathlib import Path
from typing import TypedDict
import torch

from eam.audio_model import AudioModel, AudioModelConfig, PretrainedAudioModel
from eam.encoder import EncoderConfig, get_encoder, Encoder
from eam.teacher_model import EMATrackingModel, EMATrackingModelConfig


class ModelLoadConfig(TypedDict):
    model_config_path: str
    model_path: str
    model_device: str

class SupervisedModelsConfig(TypedDict):
    audio: ModelLoadConfig | None
    text: ModelLoadConfig | None

class SuprvisedEncodersConfig(TypedDict):
    audio: EncoderConfig
    text: EncoderConfig

def load_pretrained_models(config: SupervisedModelsConfig):
    audio = None
    text = None
    if config["audio"] is not None:
        audio, _ = load_model(config["audio"])
    if config["text"] is not None:
        text, _ = load_model(config["text"])
    return text, audio

def load_model(config: ModelLoadConfig):
    """
    Loads model + encoder config + weights from a run directory.
    Skips teacher + dataset.
    """
    with open(Path(config["model_config_path"]), "r") as f:
        model_cfg = AudioModelConfig(**json.load(f))
    model_device = config["model_device"]
    model_cfg["device"] = model_device
    model = PretrainedAudioModel(model_cfg).to(model_device)
    state = torch.load(config["model_path"], map_location=model_device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)

    return model, model_cfg

def construct_audio_model(
        model_config: AudioModelConfig,
        teacher_config: EMATrackingModelConfig[AudioModel],
        seed=67):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = AudioModel(model_config)
    teacher_config["student_model"] = model
    teacher_model = EMATrackingModel(teacher_config)
    teacher_model.requires_grad_(False)
    for param in teacher_model.parameters():
        param.requires_grad = False
    # Move to specific GPU
    return model, teacher_model


def construct_encoder(encoder_config: EncoderConfig) -> Encoder:
    encoder = get_encoder(encoder_config)
    encoder.eval()
    return encoder
