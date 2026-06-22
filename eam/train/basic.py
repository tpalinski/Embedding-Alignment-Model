import threading
from eam.encoder import Encoder, EncoderConfig, get_encoder
from eam.audio_model import AudioModel, AudioModelConfig
from eam.teacher_model import EMATrackingModel, EMATrackingModelConfig
import torchaudio
import torch
import os
import sys
import datetime
from tqdm.auto import tqdm
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from queue import Queue
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .config import UnsupervisedTrainConfig

def load_audio_model(
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

def load_encoder(encoder_config: EncoderConfig) -> Encoder:
    encoder = get_encoder(encoder_config)
    encoder.eval()
    return encoder

def variance_loss(z, gamma=1.0, eps=1e-4):
    # z: [B, T, D]
    std = torch.sqrt(z.var(dim=(0,1), unbiased=False) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z):
    # z: [B, T, D]
    B, T, D = z.shape

    z = z - z.mean(dim=(0,1), keepdim=True)

    cov = torch.einsum('btd,bte->de', z, z) / (B * T - 1)

    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / D

def masked_data2vec_loss(X_student, X_teacher, mask, lambda_var=10.0, lambda_cov=1.0, gamma=0.5):
    X_t = X_teacher.mean(dim=0)  # [B, T, D]
    X_s = X_student[-1]
    X_s = F.layer_norm(X_s, X_s.shape[-1:])
    X_t = F.layer_norm(X_t, X_t.shape[-1:])

    loss_rec = (X_s - X_t) ** 2
    loss_rec = (loss_rec * mask).sum() / (mask.sum() * X_s.shape[-1])

    loss_var = variance_loss(X_s, gamma=gamma)
    loss_cov = covariance_loss(X_s)

    loss = loss_rec \
         + lambda_var * loss_var \
         + lambda_cov * loss_cov

    return loss

def train_audio(
        model: AudioModel,
        teacher_model: EMATrackingModel[AudioModel],
        train_queue: Queue[torch.Tensor],
        len_train: int,
        val_cache: list[torch.Tensor],
        test_cache: list[torch.Tensor],
        writer,
        config: UnsupervisedTrainConfig,
        save_path: str,
        checkpoint_diff: int = 10
    ):


    num_epochs = config["epochs"]
    reg_term = config["reg_term"]
    lr = config["lr"]
    accum_steps = config["accum_steps"]
    warmup_steps = config["warmup_steps"]
    lambda_var = config["lambda_var"]
    lambda_cov = config["lambda_cov"]
    gamma = config["gamma"]

    device = torch.device(model.config["device"])

    model = model.to(device)
    teacher_model = teacher_model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=reg_term)
    criterion = masked_data2vec_loss
    teacher_stream = torch.cuda.Stream(device=device)

    best_val_loss = float("inf")
    batches = 0
    step = 0

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        train_total = 0

        for train_batch in tqdm(range(len_train), desc="Train"):
            item = train_queue.get()

            inputs = item
            inputs = inputs.to(device, non_blocking=True)

            with torch.cuda.stream(teacher_stream):
                with torch.no_grad():
                    teacher_hat, _ = teacher_model(inputs)
            outputs, mask = model(inputs)
            torch.cuda.current_stream().wait_stream(teacher_stream)


            loss = criterion(outputs, teacher_hat, mask, lambda_var, lambda_cov, gamma)
            loss.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                teacher_model.update(model)

            train_loss += loss.detach() * inputs.size(0)
            train_total += inputs.size(0)
            mask_percent = mask.sum().detach()
            mask_total = torch.tensor(mask.numel(), device=device, dtype=torch.float32)
            mask_percent = (mask_percent / mask_total).item()

            student_var = outputs.var(dim=(0,1,2), unbiased=False).mean()
            teacher_var = teacher_hat.var(dim=(0,1,2), unbiased=False).mean()

            if writer:
                writer.add_scalar("train/batch_loss", loss.item(), batches)
                writer.add_scalar("train/mask_coverage", mask_percent, batches)
                writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], batches)
                writer.add_scalar("train/student_variance", student_var.item(), batches)
                writer.add_scalar("train/teacher_variance", teacher_var.item(), batches)

            batches += 1
            step += 1

        avg_train_loss = train_loss / max(train_total, 1)

        if writer:
            writer.add_scalar("train/loss", avg_train_loss, epoch)

        # =========================
        # VALIDATION LOOP
        # =========================
        model.eval()

        val_loss = 0.0
        val_total = 0

        for inputs in tqdm(val_cache, desc="Validation"):

            inputs = inputs.to(device, non_blocking=True)

            with torch.no_grad():
                outputs, mask = model(inputs)
                teacher_hat, _ = teacher_model(inputs)
                loss = criterion(outputs, teacher_hat, mask)

            val_loss += loss * inputs.size(0)
            val_total += inputs.size(0)

        avg_val_loss = val_loss / max(val_total, 1)

        if writer:
            writer.add_scalar("val/loss", avg_val_loss, epoch)

        # checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "model_best.pt"))
            print(f"Saved new best model: {best_val_loss:.4f}")

        if (epoch+1) % checkpoint_diff == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_checkpoint_{epoch+1}.pt"))
            print("Saved model checkpoint")

        print(
            f"Epoch {epoch+1} | "
            f"Train {avg_train_loss:.4f} | "
            f"Val {avg_val_loss:.4f}"
        )

    # =========================
    # TEST LOOP
    # =========================
    model.eval()

    test_loss = 0.0
    test_total = 0

    for inputs in tqdm(test_cache, desc="Test"):

        inputs = inputs.to(device, non_blocking=True)

        with torch.no_grad():
            outputs, mask = model(inputs)
            teacher_hat, _ = teacher_model(inputs)
            loss = criterion(outputs, teacher_hat, mask)

        test_loss += loss * inputs.size(0)
        test_total += inputs.size(0)

    final_test_loss = test_loss / max(test_total, 1)

    print(f"Final Test Loss: {final_test_loss:.4f}")

    return model
