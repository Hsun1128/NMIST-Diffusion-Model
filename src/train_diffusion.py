"""
訓練入口。此檔案主要負責參數解析與模組組裝。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import build_dataloader
from diffusion import DiffusionConfig, DiffusionProcess
from logger import CSVLogger
from model import UNet
from trainer import train, set_seed
from visualize_diffusion import visualize_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDPM on MNIST from scratch.")
    parser.add_argument("--data-dir", type=str, default="mnist", help="指向題目提供的本地 MNIST 影像資料夾。")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=28, help="輸入影像邊長，預設為 28 以符合 MNIST 原始尺寸。")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--residual-dropout", type=float, default=0.0, help="殘差區塊內的 dropout 機率。")
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50, help="寫入訓練 log 的頻率（step）。")
    parser.add_argument("--eval-every", type=int, default=1, help="每多少 epoch 進行一次評估。")
    parser.add_argument("--eval-batches", type=int, default=50, help="評估時計算的批次數目上限。")
    parser.add_argument("--checkpoint-epoch-every", type=int, default=5, help="每隔多少 epoch 儲存一次 ckpt_epoch 檔案，<=0 代表停用。")
    parser.add_argument("--output-dir", type=str, default="trained_model", help="所有權重/樣本/紀錄的根目錄。")
    parser.add_argument("--run-name", type=str, default=None, help="子資料夾名稱，預設會以時間戳命名。")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-split", type=float, default=0.9, help="訓練集比例（例如 0.9 表示 90%% 訓練，10%% 驗證）。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    output_root = Path(args.output_dir)
    run_dir = output_root / run_name
    checkpoint_dir = run_dir / "checkpoints"
    sample_dir = run_dir
    log_path = run_dir / "train_log.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.image_size,
        shuffle=True,
        drop_last=True,
        train_split=args.train_split,
        is_train=True,
        seed=args.seed,
    )
    eval_loader = build_dataloader(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.image_size,
        shuffle=False,
        drop_last=False,
        train_split=args.train_split,
        is_train=False,
        seed=args.seed,
    )
    
    # 輸出數據劃分資訊
    train_size = len(train_loader.dataset)
    val_size = len(eval_loader.dataset)
    print(f"數據劃分：訓練集 {train_size} 筆 ({train_size/(train_size+val_size)*100:.1f}%)，驗證集 {val_size} 筆 ({val_size/(train_size+val_size)*100:.1f}%)")
    model = UNet(
        in_channels=3,
        base_channels=args.base_channels,
        time_dim=args.base_channels * 4,
        residual_dropout=args.residual_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    diffusion = DiffusionProcess(
        DiffusionConfig(args.timesteps, args.beta_start, args.beta_end),
        device,
    )
    logger = CSVLogger(log_path, fieldnames=["step", "epoch", "split", "loss", "lr"])

    train(
        model=model,
        dataloader=train_loader,
        diffusion=diffusion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        sample_every=args.sample_every,
        sample_batch_size=args.sample_batch_size,
        sample_dir=sample_dir.as_posix(),
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=checkpoint_dir.as_posix(),
        image_size=args.image_size,
        log_every=args.log_every,
        logger=logger,
        eval_loader=eval_loader,
        eval_every=args.eval_every,
        max_eval_batches=args.eval_batches,
        best_checkpoint_path=checkpoint_dir / "best.pt",
        checkpoint_epoch_every=args.checkpoint_epoch_every,
    )

    if log_path.exists() and log_path.stat().st_size > 0:
        df = pd.read_csv(log_path)
        if not df.empty:
            plt.figure(figsize=(10, 5))
            for split, group in df.groupby("split"):
                plt.plot(group["step"], group["loss"], label=split)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training vs Evaluation Loss")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plot_path = run_dir / "train_log.png"
            plt.savefig(plot_path, dpi=200)
            plt.close()

    best_ckpt_path = checkpoint_dir / "best.pt"
    if best_ckpt_path.exists():
        try:
            visualize_from_checkpoint(
                checkpoint_path=best_ckpt_path,
                output_path=run_dir / "diffusion_progress.png",
                batch_size=8,
                segments=7,
                model_image_size=args.image_size,
                output_size=args.image_size,
                timesteps=args.timesteps,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                base_channels=args.base_channels,
                residual_dropout=args.residual_dropout,
                device=device,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"[Warning] 無法自動產生 diffusion 過程圖：{exc}")
    else:
        print("[Info] best checkpoint 不存在，略過 diffusion 可視化。")


if __name__ == "__main__":
    main()
