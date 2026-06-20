import torchaudio
import torch
import os
import sys
import datetime
from tqdm.auto import tqdm
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )

    print(f"Rank {local_rank} initialized", flush=True)

    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()


if __name__ == "__main__":
    # Setup distributed training
    local_rank = setup_distributed()
    print("RANK:", int(os.environ["RANK"]), "PYTHON:", sys.executable)
    world_size = dist.get_world_size()

    from eam.train import UnsupervisedTrainConfig, load_audio_model_distributed, train_audio_distributed
    from eam.audio_encoder import get_m4t_default_config
    from eam.audio_model import AudioModelConfig, get_default_audio_model_config
    from eam.dataset import get_dataset, get_default_admed_config
    from eam.teacher_model import get_default_audio_teacher_config
        # TODO - parsing of configs from files, support different modalities.
    training_config = UnsupervisedTrainConfig(
        epochs = 50,
        lr = 1e-4,
        reg_term = 1e-4,
        accum_steps = 8,
        warmup_steps = 5000,
        save_path ="models/m4t_pretrain.pt",
        distributed=True,
        tensorboard_output="runs/m4t_pretrain"+str(datetime.datetime.now())
    )

    model_config = get_default_audio_model_config()
    model_config["feature_dims"] = 1024

    teacher_config = get_default_audio_teacher_config()

    encoder_config = get_m4t_default_config()

    dataset_config = get_default_admed_config()


    if local_rank == 0:
        writer = SummaryWriter(log_dir=training_config["tensorboard_output"])
        print(f"Training with {world_size} GPUs")
    else:
        writer = None

    # Load model
    if local_rank == 0:
        print("Loading model...")
    model, teacher_model, encoder = load_audio_model_distributed(local_rank, model_config, teacher_config, encoder_config)

    # Load datasets
    if local_rank == 0:
        print("Loading datasets...")
    print(f"Rank: {local_rank}, loading dataset")
    ds = get_dataset(dataset_config)
    train_loader, val_loader, test_loader = ds.get_loaders_distributed()
    print(f"Rank: {local_rank}, done loading")

    dist.barrier()
    # Train model
    if local_rank == 0:
        print("Starting training...")
    model = train_audio_distributed(model, teacher_model, encoder, train_loader, val_loader, test_loader, local_rank, writer, training_config)

    # Save final model (only rank 0)
    if local_rank == 0:
        torch.save(model.module.state_dict(), training_config["save_path"])
        print(f"Training complete! Final model saved to f{training_config["save_path"]}")
        writer.close()

    cleanup_distributed()
