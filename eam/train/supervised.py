import torch
import os
from tqdm.auto import tqdm
from torch import nn
from queue import Queue
import torch.nn.functional as F

from .config import SupervisedTrainConfig
from eam.alignment_model import AlignmentModel

def contrastive(a, t, temperature=0.05):
    logits = (a @ t.T) / temperature
    labels = torch.arange(len(a), device=a.device)

    return (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    ) * 0.5

def guidance_loss(a_guided, t_base):
    # align guided audio to text geometry
    logits = a_guided @ t_base.T
    labels = torch.arange(len(a_guided), device=a_guided.device)

    return (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    ) * 0.5

def covariance_penalty(z):
    z = z - z.mean(0, keepdim=True)
    cov = (z.T @ z) / (z.size(0) - 1)

    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).mean()

def cosine_sim(a, b):
    return F.cosine_similarity(a, b, dim=-1).mean()


def train_supervised(
        model: AlignmentModel,
        train_queue: Queue[tuple[torch.Tensor, torch.Tensor]],
        len_train: int,
        val_cache: list[tuple[torch.Tensor, torch.Tensor]],
        test_cache: list[tuple[torch.Tensor, torch.Tensor]],
        writer,
        config: SupervisedTrainConfig,
        save_path: str,
        checkpoint_diff: int = 10
    ):


    num_epochs = config["epochs"]
    reg_term = config["reg_term"]
    lr = config["lr"]
    lambda_cov = config["lambda_cov"]
    temperature_start = config["temperature_start"]
    temperature_end = config["temperature_end"]
    lambda_guided = config["lambda_guided"]
    lr_warmup = config["lr_warmup"]

    device = torch.device(model.config["device"])

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=reg_term)

    best_val_loss = float("inf")
    batches = 0
    step = 0
    total_steps = len_train * num_epochs
    warmup_steps = int(lr_warmup * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def get_temperature(step):
        progress = step / total_steps
        temperature_t = temperature_start - (temperature_start - temperature_end) * progress
        return temperature_t

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        train_total = 0
        train_cos_sims = []

        for train_batch in tqdm(range(len_train), desc="Train"):
            text, audio = train_queue.get()

            audio = audio.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)

            outputs = model(audio, text)
            a = outputs["a"]
            t = outputs["t"]

            contrastive_loss = contrastive(a, t, get_temperature(step))

            guided_loss_val = 0.0
            if "a_guided" in outputs:
                guided_loss_val = guidance_loss(outputs["a_guided"], t)

            cov_a = covariance_penalty(a)
            cov_t = covariance_penalty(t)

            loss = (
                contrastive_loss
                + lambda_guided * guided_loss_val
                + lambda_cov * cov_a
                + lambda_cov * cov_t
            )

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pos = F.cosine_similarity(outputs["a"], outputs["t"], dim=-1).mean()
            train_cos_sims.append(pos.item())
            sim = a @ t.T
            labels = torch.arange(sim.size(0), device=sim.device)

            neg_mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
            neg_mean = sim[neg_mask].mean()
            hard_neg = sim.masked_fill(torch.eye(sim.size(0), device=sim.device).bool(), -1e9).max(dim=1).values.mean()
            margin = pos - hard_neg
            train_loss += loss.detach() * audio.size(0)
            train_total += audio.size(0)

            if writer:
                writer.add_scalar("train/batch/loss", loss.detach().item(), batches)
                writer.add_scalar("train/batch/contrastive_loss", contrastive_loss.item(), batches)
                writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], batches)
                writer.add_scalar("train/batch/pos_cos_sim", pos.item(), batches)
                writer.add_scalar("train/batch/neg_cos_sim", neg_mean.item(), batches)
                writer.add_scalar("train/batch/hard_neg", hard_neg.item(), batches)
                writer.add_scalar("train/batch/margin", margin.item(), batches)
                writer.add_scalar("train/temperature", get_temperature(step), batches)
                if "a_guided" in outputs:
                    writer.add_scalar("train/guided_loss", guided_loss_val.item(), batches)

            batches += 1
            step += 1

        avg_train_loss = train_loss / max(train_total, 1)

        if writer:
            writer.add_scalar("train/loss", avg_train_loss, epoch)
            writer.add_scalar("train/cosine_sim", sum(train_cos_sims) / len(train_cos_sims), epoch)

        # =========================
        # VALIDATION LOOP
        # =========================
        model.eval()

        val_loss = 0.0
        val_total = 0
        val_cos_sims = []
        val_loss_contrastive = 0.0

        for inputs in tqdm(val_cache, desc="Validation"):

            text, audio = inputs
            audio = audio.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)

            with torch.no_grad():
                outputs = model(audio, text)
                a = outputs["a"]
                t = outputs["t"]
            loss = contrastive(a, t, get_temperature(step))
            contrastive_for_log = loss.detach()
            loss += lambda_cov * covariance_penalty(a)
            loss += lambda_cov * covariance_penalty(t)


            val_loss += loss * audio.size(0)
            val_loss_contrastive += contrastive_for_log * audio.size(0)
            val_total += audio.size(0)
            cos_sim = F.cosine_similarity(a, t, dim=1).mean()

            val_cos_sims.append(cos_sim.item())

        avg_val_loss = val_loss / max(val_total, 1)
        avg_val_loss_contrastive = val_loss_contrastive / max(val_total, 1)

        if writer:
            writer.add_scalar("val/loss", avg_val_loss, epoch)
            writer.add_scalar("val/loss_contrastive", avg_val_loss_contrastive, epoch)
            writer.add_scalar("val/cosine_sim", sum(val_cos_sims) / len(val_cos_sims), epoch)

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
    test_loss_contrastive = 0.0
    test_total = 0

    for inputs in tqdm(test_cache, desc="Test"):

        text, audio = inputs
        audio = audio.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = model(audio, text)
            a = outputs["a"]
            t = outputs["t"]
        loss = contrastive(a, t, get_temperature(temperature_end))
        contrastive_for_log = loss.detach()
        loss += lambda_cov * covariance_penalty(a)
        loss += lambda_cov * covariance_penalty(t)


        test_loss += loss * audio.size(0)
        test_loss_contrastive += contrastive_for_log * audio.size(0)
        test_total += audio.size(0)


    final_test_loss = test_loss / max(test_total, 1)
    final_test_loss_contrastive = test_loss_contrastive / max(test_total, 1)

    print(f"Final Test Loss: {final_test_loss:.4f}, contrastive: {final_test_loss_contrastive:.4f}")

    return model
