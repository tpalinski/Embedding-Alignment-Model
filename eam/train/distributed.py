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
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .config import UnsupervisedTrainConfig

def load_audio_model(
        local_rank,
        model_config: AudioModelConfig,
        teacher_config: EMATrackingModelConfig,
        encoder_config: EncoderConfig,
        seed=67):
    print(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Rank {local_rank} - creating model")
    model = AudioModel(model_config)
    model = model.to(device)
    print(f"Rank {local_rank} - creating DDP")
    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False
    )
    print(f"Rank {local_rank} - ddp done")
    teacher_config["student_model"] = model.module
    print(f"Rank {local_rank} - creating teacher model")
    teacher_model = EMATrackingModel(teacher_config)
    teacher_model.requires_grad_(False)
    for param in teacher_model.parameters():
        param.requires_grad = False

    encoder_config["device"] = device
    encoder = get_encoder(encoder_config)
    # Move to specific GPU
    teacher_model = teacher_model.to(device)
    print(f"Rank {local_rank} - moved to {device}")
    # wait for sync cause of DDP weird state

    torch.cuda.empty_cache()
    return model, teacher_model, encoder


def sync_teacher(teacher, student, rank):
    student_state = student.module.state_dict()
    teacher_state = teacher.model.state_dict()
    for k in teacher_state.keys():
        if rank == 0:
            teacher_state[k].copy_(student_state[k])
        dist.broadcast(teacher_state[k], src=0)

def masked_data2vec_loss(X_student, X_teacher, mask):
    X_t = X_teacher.mean(dim=0)  # [B, T, D]
    X_s = X_student[-1]
    X_s = F.layer_norm(X_s, X_s.shape[-1:])
    X_t = F.layer_norm(X_t, X_t.shape[-1:])
    loss = (X_s - X_t) ** 2
    loss = (loss * mask).sum() / (mask.sum() * X_s.shape[-1])
    return loss

def all_reduce_mean(x: torch.Tensor):
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()
    return x

def train_audio_distributed(
        model: AudioModel,
        teacher_model: EMATrackingModel[AudioModel],
        encoder: Encoder,
        train_loader,
        val_loader,
        test_loader,
        rank,
        writer,
        config: UnsupervisedTrainConfig,
        ):
    num_epochs = config["epochs"]
    reg_term = config["reg_term"]
    save_path = "model_dist.pt"
    lr = config["lr"]
    accum_steps = config["accum_steps"]
    warmup_steps = config["warmup_steps"]
    device = torch.device(f'cuda:{rank}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=reg_term)
    criterion = masked_data2vec_loss
    sync_teacher(teacher_model, model, rank)
    teacher_stream = torch.cuda.Stream(device)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_loss = 2137.0

    dist.barrier()
    batches = 0
    for epoch in range(num_epochs):
      model.train()  # Set model to training mode
      train_loss = 0.0
      train_total = 0
      loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")

      # Training loop
      for step, loader in enumerate(loop):
          inputs, _ = loader
          inputs = encoder.encode(inputs)
          inputs = inputs.to(device)

          outputs, mask = model(inputs)
          torch.cuda.current_stream().wait_stream(teacher_stream)
          with torch.no_grad():
              teacher_hat, _ = teacher_model(inputs)
          loss = criterion(outputs, teacher_hat, mask)
          loss.backward()
          if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
              optimizer.step()
              scheduler.step()
              with torch.cuda.stream(teacher_stream):
                  with torch.no_grad():
                      teacher_model.update(model.module)
              optimizer.zero_grad()
          batch_loss = loss.detach()
          train_loss += batch_loss * inputs.size(0)
          train_total += inputs.size(0)
          # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
          mask_percent = mask.sum()/mask.numel()
          loss_for_log = loss.detach()
          dist.all_reduce(loss_for_log, op=dist.ReduceOp.SUM)
          loss_for_log /= dist.get_world_size()

          mask_percent = mask.sum().detach()
          mask_total = torch.tensor(mask.numel(), device=device, dtype=torch.float32)
          dist.all_reduce(mask_percent)
          dist.all_reduce(mask_total)
          mask_percent = (mask_percent / mask_total).item()

          student_var = outputs[-1].var(dim=(0,1), unbiased=False).mean()  # mean over C
          teacher_var = teacher_hat.var(dim=(0,1,2), unbiased=False).mean()
          dist.all_reduce(student_var)
          dist.all_reduce(teacher_var)

          student_var /= dist.get_world_size()
          teacher_var /= dist.get_world_size()
          if writer is not None:
              writer.add_scalar("train/batch_loss", loss_for_log, batches)
              writer.add_scalar("train/mask_coverage", mask_percent, batches)
              writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], batches)
              writer.add_scalar("train/student_variance", student_var.item(), batches)
              writer.add_scalar("train/teacher_variance", teacher_var.item(), batches)
          batches+=1
      train_loss = torch.tensor(train_loss, device=device)
      train_total = torch.tensor(train_total, device=device)

      dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
      dist.all_reduce(train_total, op=dist.ReduceOp.SUM)

      avg_train_loss = (train_loss / train_total).item()
      if writer is not None:
          writer.add_scalar("train/loss", avg_train_loss, epoch)

      # Validation loop
      model.eval()  # Set model to evaluation mode
      val_loss = 0.0
      val_total = 0

      with torch.no_grad():  # Disable gradient calculation for validation
          for inputs, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
              inputs = inputs.to(device)

              outputs, mask = model(inputs)
              teacher_hat, _ = teacher_model(inputs)
              loss = criterion(outputs, teacher_hat, mask)

              val_loss += loss.detach() * inputs.size(0)
              val_total += inputs.size(0)

      val_loss = torch.tensor(val_loss, device=device)
      val_total = torch.tensor(val_total, device=device)

      dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
      dist.all_reduce(val_total, op=dist.ReduceOp.SUM)

      avg_val_loss = (val_loss / val_total).item()
      if writer is not None:
          writer.add_scalar("val/loss", avg_val_loss, epoch)
          if avg_val_loss < best_val_loss:
              torch.save(model.module.state_dict(), save_path)
              print(f"Saved new best model, new best loss: {avg_val_loss}")
              best_val_loss = avg_val_loss

      print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}")

    print("Training complete!")
    model.eval()  # Set model to evaluation mode
    test_total = 0
    test_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation for testing
      for inputs, _ in tqdm(test_loader, desc="Final Test Evaluation"):
          inputs = inputs.to(device)

          outputs, mask = model(inputs)
          teacher_hat, _ = teacher_model(inputs)
          loss = criterion(outputs, teacher_hat, mask)
          test_total += inputs.size(0)
          test_loss += loss * inputs.size(0)

    test_loss = torch.tensor(test_loss, device=device)
    test_total = torch.tensor(test_total, device=device)

    dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(test_total, op=dist.ReduceOp.SUM)

    final_test_loss = (test_loss / test_total).item()
    print(f"Final Test Loss: {final_test_loss:.4f}")
    return model
