import datetime
import json
from pathlib import Path
from eam.alignment_model import AlignmentModelConfig
from eam.train import UnsupervisedTrainConfig
from eam.encoder import EncoderConfig
from eam.audio_model import AudioModelConfig
from eam.dataset import DatasetConfig
from eam.teacher_model import EMATrackingModelConfig
from eam.train.config import SupervisedTrainConfig
from eam.utils.model_load import SupervisedModelsConfig, SuprvisedEncodersConfig

def load_config_dir(config_path: str):
    config_dir = Path(config_path)

    with open(config_dir / "train.json") as f:
        train_cfg = UnsupervisedTrainConfig(**json.load(f))

    with open(config_dir / "model.json") as f:
        model_cfg = AudioModelConfig(**json.load(f))

    with open(config_dir / "encoder.json") as f:
        encoder_cfg = EncoderConfig(**json.load(f))

    with open(config_dir / "dataset.json") as f:
        dataset_cfg = DatasetConfig(**json.load(f))

    with open(config_dir / "teacher.json") as f:
        teacher_cfg = EMATrackingModelConfig(**json.load(f))

    return train_cfg, model_cfg, encoder_cfg, dataset_cfg, teacher_cfg

def load_supervised_config_dir(config_path: str):
    config_dir = Path(config_path)

    with open(config_dir / "train.json") as f:
        train_cfg = SupervisedTrainConfig(**json.load(f))

    with open(config_dir / "pretrained.json") as f:
        pretrained_cfg = SupervisedModelsConfig(**json.load(f))

    with open(config_dir / "encoders.json") as f:
        encoders_cfg = SuprvisedEncodersConfig(**json.load(f))

    with open(config_dir / "dataset.json") as f:
        dataset_cfg = DatasetConfig(**json.load(f))

    with open(config_dir / "model.json") as f:
        model_cfg = AlignmentModelConfig(**json.load(f))

    return train_cfg, model_cfg, encoders_cfg, dataset_cfg, pretrained_cfg

def create_run_dir(base_dir="runs"):
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_path = Path(base_dir) / f"exp_{run_id}"

    (run_path / "config").mkdir(parents=True, exist_ok=True)
    (run_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_path / "logs").mkdir(parents=True, exist_ok=True)

    return run_path

def save_configs(run_path, train_cfg, model_cfg, encoder_cfg, dataset_cfg, teacher_cfg):
    cfg_dir = run_path / "config"

    def dump(obj, name):
        with open(cfg_dir / f"{name}.json", "w") as f:
            json.dump(obj, f, indent=2)

    dump(train_cfg, "train")
    dump(model_cfg, "model")
    dump(encoder_cfg, "encoder")
    dump(dataset_cfg, "dataset")
    dump(teacher_cfg, "teacher")

def save_supervised_configs(run_path, train_cfg, model_cfg, encoders_cfg, dataset_cfg, pretrained_cfg):
    cfg_dir = run_path / "config"

    def dump(obj, name):
        with open(cfg_dir / f"{name}.json", "w") as f:
            json.dump(obj, f, indent=2)

    dump(train_cfg, "train")
    dump(model_cfg, "model")
    dump(encoders_cfg, "encoders")
    dump(dataset_cfg, "dataset")
    dump(pretrained_cfg, "pretrained")
