"""
訓練流程與共用工具函式。
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm.auto import tqdm

from diffusion import DiffusionProcess
from logger import CSVLogger


def set_seed(seed: int) -> None:
    """設定隨機種子，提升實驗可重現性。"""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_samples(images: torch.Tensor, sample_dir: str, step: int) -> None:
    """保存採樣結果方便追蹤模型訓練進度。"""

    os.makedirs(sample_dir, exist_ok=True)
    grid = utils.make_grid((images + 1) / 2, nrow=8)
    utils.save_image(grid, os.path.join(sample_dir, f"sample_step_{step:06d}.png"))


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    diffusion: DiffusionProcess,
    device: torch.device,
    max_batches: int,
) -> float:
    """以與訓練相同的噪聲預測任務計算平均 loss。"""

    model.eval()
    losses = []
    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        images = images.to(device)
        noise = torch.randn_like(images)
        timesteps = diffusion.sample_timesteps(images.size(0))
        noisy_images = diffusion.q_sample(images, timesteps, noise)
        predicted_noise = model(noisy_images, timesteps)
        loss = F.mse_loss(predicted_noise, noise)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    diffusion: DiffusionProcess,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    sample_every: int,
    sample_batch_size: int,
    sample_dir: str,
    checkpoint_every: int,
    checkpoint_dir: str,
    image_size: int,
    log_every: int,
    logger: Optional[CSVLogger],
    eval_loader: Optional[DataLoader],
    eval_every: int,
    max_eval_batches: int,
    best_checkpoint_path: Path,
    checkpoint_epoch_every: int,
) -> None:
    """核心訓練迴圈，同時週期性地產生樣本與儲存 checkpoint。"""

    global_step = 0
    for epoch in range(epochs):
        model.train()
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for images, _ in progress:
            images = images.to(device)
            noise = torch.randn_like(images)
            timesteps = diffusion.sample_timesteps(images.size(0))

            noisy_images = diffusion.q_sample(images, timesteps, noise)
            predicted_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress.set_postfix(loss=loss.item())

            if logger and global_step % log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.log(step=global_step, epoch=epoch + 1, split="train", loss=f"{loss.item():.6f}", lr=f"{current_lr:.8f}")

            if global_step % sample_every == 0:
                model.eval()
                with torch.no_grad():
                    samples = diffusion.sample(
                        model, image_size=image_size, batch_size=sample_batch_size, channels=3
                    )
                    save_samples(samples, sample_dir, global_step)
                model.train()

            if global_step % checkpoint_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    global_step,
                    os.path.join(checkpoint_dir, f"ckpt_{global_step:06d}.pt"),
                )

            global_step += 1

        if eval_loader is not None and (epoch + 1) % eval_every == 0:
            eval_loss = evaluate(model, eval_loader, diffusion, device, max_eval_batches)
            if logger:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.log(step=global_step, epoch=epoch + 1, split="eval", loss=f"{eval_loss:.6f}", lr=f"{current_lr:.8f}")
            if eval_loss < train.best_eval_loss:
                train.best_eval_loss = eval_loss
                save_checkpoint(model, optimizer, global_step, best_checkpoint_path.as_posix())

        if checkpoint_epoch_every > 0 and (epoch + 1) % checkpoint_epoch_every == 0:
            save_checkpoint(
                model,
                optimizer,
                global_step,
                os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch + 1:02d}.pt"),
            )


# 初始化 best loss 屬性
train.best_eval_loss = float("inf")

