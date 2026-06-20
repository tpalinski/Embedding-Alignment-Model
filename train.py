import argparse
import threading
import datetime
from dataclasses import asdict
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from eam.train import UnsupervisedTrainConfig, load_audio_model, load_encoder, train_audio
from eam.encoder import EncoderConfig
from eam.audio_model import AudioModelConfig
from eam.dataset import DatasetConfig, get_dataset
from eam.teacher_model import EMATrackingModelConfig
from eam.train.streaming.encoder import EncoderPipeline

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_dir",
        type=str,
        required=True,
        help="Path to experiment config directory"
    )

    return parser.parse_args()

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

if __name__ == "__main__":
    run_path = create_run_dir()
    args = parse_args()
    config_dir = args.config_dir
    train_cfg, model_cfg, encoder_cfg, dataset_cfg, teacher_cfg = load_config_dir(config_dir)

    save_configs(run_path, train_cfg, model_cfg, encoder_cfg, dataset_cfg, teacher_cfg)

    writer = SummaryWriter(log_dir=run_path / "logs" / "tensorboard")

    stop_event = threading.Event()
    print("Loading model")
    model, teacher_model = load_audio_model(model_cfg, teacher_cfg)
    print("Loading encoder")
    encoder = load_encoder(encoder_cfg)

    print("Fetching dataset loaders")
    ds = get_dataset(dataset_cfg)
    train_loader, val_loader, test_loader = ds.get_loaders()

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    pipeline = EncoderPipeline(
        encoder,
        loaders,
        stop_event,
        device=encoder_cfg["device"],
    )
    print("Preparing val and test cache")
    val_cache, test_cache = pipeline.prepare_cache()
    queue, train_len = pipeline.get_queue()

    encoder_thread = threading.Thread(
        target=pipeline.run,
        daemon=True
    )

    print("Starting encoder thread")
    encoder_thread.start()

    print("Starting model training")
    _ = train_audio(model, teacher_model, queue, train_len, val_cache, test_cache, writer, train_cfg, f"{run_path}/checkpoints/")

    stop_event.set()
    encoder_thread.join()



