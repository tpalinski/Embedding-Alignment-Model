import argparse
import threading
from torch.utils.tensorboard import SummaryWriter
from eam.alignment_model import AlignmentModel
from eam.train import DualEncoderPipeline, train_audio, train_supervised
from eam.dataset import get_dataset
from eam.utils import construct_audio_model, construct_encoder, create_run_dir, load_pretrained_models, load_supervised_config_dir, save_configs, load_config_dir, save_supervised_configs

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_dir",
        type=str,
        required=True,
        help="Path to experiment config directory"
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_path = create_run_dir()
    args = parse_args()
    config_dir = args.config_dir
    train_cfg, model_cfg, encoders_cfg, dataset_cfg, pretrained_cfg = load_supervised_config_dir(config_dir)

    save_supervised_configs(run_path, train_cfg, model_cfg, encoders_cfg, dataset_cfg, pretrained_cfg)

    writer = SummaryWriter(log_dir=run_path / "logs" / "tensorboard")

    stop_event = threading.Event()
    print("Loading pretrained models")
    audio_model, text_model = load_pretrained_models(pretrained_cfg)
    print("Loading encoders")
    audio_encoder = construct_encoder(encoders_cfg["audio"])
    text_encoder = construct_encoder(encoders_cfg["text"])

    print("Constructing alignment model")
    alignment_model = AlignmentModel(model_cfg, audio_model, text_model)

    print("Fetching dataset loaders")
    ds = get_dataset(dataset_cfg)
    train_loader, val_loader, test_loader = ds.get_loaders()

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    pipeline = DualEncoderPipeline(
        text_encoder,
        audio_encoder,
        loaders,
        stop_event,
        audio_device=encoders_cfg["audio"]["device"],
        text_device=encoders_cfg["text"]["device"],
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
    model = train_supervised(alignment_model, queue, train_len, val_cache, test_cache, writer, train_cfg, f"{run_path}/checkpoints/")

    stop_event.set()
    encoder_thread.join()



