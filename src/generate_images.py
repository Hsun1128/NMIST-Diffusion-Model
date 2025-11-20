"""
以訓練好的 DDPM 權重生成指定數量的 PNG 影像。

需求重點：
    - 產生 10,000 張影像（可透過參數調整）。
    - 最終解析度需為 28x28，且為 RGB（3 通道）。
    - 檔名需為零填充格式，例如 00001.png ~ 10000.png。
    - 所有影像輸出到 `generated/` 目錄（可由參數指定）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm.auto import tqdm

from diffusion import DiffusionConfig, DiffusionProcess
from model import UNet
from trainer import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PNG images with a trained DDPM model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型 checkpoint 的路徑，需包含 `model` 欄位的 state dict。",
    )
    parser.add_argument("--output-dir", type=str, default="generated", help="輸出影像資料夾。")
    parser.add_argument("--num-images", type=int, default=10000, help="生成的影像數量。")
    parser.add_argument("--batch-size", type=int, default=64, help="每次採樣的批次大小。")
    parser.add_argument("--model-image-size", type=int, default=28, help="模型訓練時使用的影像邊長，需符合 UNet 的多次下採樣需求。")
    parser.add_argument("--output-size", type=int, default=28, help="最終輸出影像邊長，預設為 28x28（與訓練尺寸相同，無需 resize）。")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--residual-dropout", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None, help="指定運算裝置（cuda/cpu）。預設自動偵測，有 GPU 時會使用 cuda。")
    parser.add_argument("--seed", type=int, default=3407, help="隨機種子以利重現。")
    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    base_channels: int,
    residual_dropout: float,
) -> UNet:
    model = UNet(
        in_channels=3,
        base_channels=base_channels,
        time_dim=base_channels * 4,
        residual_dropout=residual_dropout,
    ).to(device)
    checkpoint = torch.load(checkpoint_path.as_posix(), map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("num-images 必須為正整數。")
    if args.batch_size <= 0:
        raise ValueError("batch-size 必須為正整數。")

    set_seed(args.seed)
    device_str = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        base_channels=args.base_channels,
        residual_dropout=args.residual_dropout,
    )
    diffusion = DiffusionProcess(
        DiffusionConfig(timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end),
        device=device,
    )

    total_generated = 0
    progress = tqdm(total=args.num_images, desc="Generating images", unit="img")
    while total_generated < args.num_images:
        current_batch = min(args.batch_size, args.num_images - total_generated)
        samples = diffusion.sample(
            model=model,
            image_size=args.model_image_size,
            batch_size=current_batch,
            channels=3,
        )

        samples = samples.clamp(-1.0, 1.0)
        if args.output_size != samples.shape[-1]:
            samples = F.interpolate(
                samples,
                size=(args.output_size, args.output_size),
                mode="bilinear",
                align_corners=False,
            )
        samples = (samples + 1.0) / 2.0

        for idx in range(current_batch):
            global_index = total_generated + idx + 1
            save_path = output_dir / f"{global_index:05d}.png"
            save_image(samples[idx].cpu(), save_path)

        total_generated += current_batch
        progress.update(current_batch)

    progress.close()
    print(f"完成：在 {output_dir} 產生 {args.num_images} 張影像。")


if __name__ == "__main__":
    main()

