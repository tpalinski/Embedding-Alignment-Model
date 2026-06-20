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
from data2vec_model.Data2VecModelConfig import Data2VecModelConfig
from data2vec_model.EegConformer import EegConformer
from data2vec_model.EegSmallEnc import EegSmallEnc
from data2vec_model.EegSmallEnc16 import EegSmallEnc16
from data2vec_model.EegSmallEnc32 import EegSmallEnc32
from models.dataloader import load_data
from teacher.EMATrackingModel import EMATrackingModel
from teacher.EMATrackingModelConfig import EMATrackingModelConfig


suffix="_d2v_16"
num_epochs = 100
reg_term = 1e-4
lr = 1e-4
seed=67
batch_size = 8  # Total batch size across all GPUs (32 per GPU with 4 GPUs)
accum_steps = 8

def load_model(local_rank):
    print(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_config = Data2VecModelConfig(
      mask_p=0.065,
      mask_steps=10,
      K=2
      )
    print(f"Rank {local_rank} - creating model")
    model = EegSmallEnc16(model_config)
    teacher_config = EMATrackingModelConfig[EegSmallEnc16](
      student_model=model,
      annealing_warmup=7500,
      annealing_init_value=0.9990,
      annealing_target_value=0.9999,
      copy_params=model.get_skippable_params()
      )
    print(f"Rank {local_rank} - creating teacher model")
    teacher_model = EMATrackingModel(teacher_config)
    teacher_model.requires_grad_(False)
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Move to specific GPU
    model = model.to(device)
    teacher_model = teacher_model.to(device)
    print(f"Rank {local_rank} - moved to {device}")

    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False
    )
    print(f"Rank {local_rank} - ddp done")

    torch.cuda.empty_cache()
    return model, teacher_model

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

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



def train(model, teacher_model, train_loader, val_loader, test_loader, rank, world_size, writer):
    device = torch.device(f'cuda:{rank}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=reg_term)
    criterion = masked_data2vec_loss
    sync_teacher(teacher_model, model, rank)
    teacher_stream = torch.cuda.Stream(device)
    warmup_steps = 5000
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
          batch_loss = loss.item()
          train_loss += batch_loss * inputs.size(0)
          train_total += inputs.size(0)
          # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
          mask_percent = mask.sum()/mask.numel()
          if writer is not None:
              student_var = outputs[-1].var(dim=(0,1), unbiased=False).mean()  # mean over C
              teacher_var = teacher_hat.var(dim=(0,1,2), unbiased=False).mean()

              writer.add_scalar("train/batch_loss", batch_loss, batches)
              writer.add_scalar("train/mask_coverage", mask_percent, batches)
              writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], batches)
              writer.add_scalar("train/student_variance", student_var.item(), batches)
              writer.add_scalar("train/teacher_variance", teacher_var.item(), batches)
          batches+=1

      avg_train_loss = train_loss / train_total
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

              val_loss += loss.item() * inputs.size(0)
              val_total += inputs.size(0)

      avg_val_loss = val_loss / val_total
      if writer is not None:
          writer.add_scalar("val/loss", avg_val_loss, epoch)
          if avg_val_loss < best_val_loss:
              torch.save(model.module.state_dict(), f"model_best{suffix}.pt")
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

    final_test_loss = test_loss / test_total
    print(f"Final Test Loss: {final_test_loss:.4f}")
    return model


if __name__ == "__main__":
    # Setup distributed training
    local_rank = setup_distributed()
    print("RANK:", int(os.environ["RANK"]), "PYTHON:", sys.executable)
    world_size = dist.get_world_size()

    # Only rank 0 creates tensorboard writer
    if local_rank == 0:
        tensorboard_output = "runs/d2v16/run_" + str(datetime.datetime.now())
        writer = SummaryWriter(log_dir=tensorboard_output)
        print(f"Training with {world_size} GPUs")
    else:
        writer = None

    # Load model
    if local_rank == 0:
        print("Loading model...")
    model, teacher_model = load_model(local_rank)

    # Load datasets
    if local_rank == 0:
        print("Loading datasets...")
    print(f"Rank: {local_rank}, loading dataset")
    train_loader, val_loader, test_loader = load_data(world_size, local_rank, batch_size, 'zpbDataset16')
    print(f"Rank: {local_rank}, done loading")

    dist.barrier()
    # Train model
    if local_rank == 0:
        print("Starting training...")
    model = train(model, teacher_model, train_loader, val_loader, test_loader, local_rank, world_size, writer)

    # Save final model (only rank 0)
    if local_rank == 0:
        torch.save(model.module.state_dict(), f"model_final{suffix}.pt")
        print("Training complete! Final model saved to model_final.pt")
        writer.close()

    cleanup_distributed()
