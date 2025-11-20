"""
可視化 DDPM 反向擴散過程，可單獨執行或在訓練後自動呼叫。

功能：
    - 從純噪聲出發，同時生成 8 條獨立軌跡（預設）。
    - 將整體 timestep 分成 7 等份（預設），於每個分段末端紀錄影像。
    - 若提供 `--run-name`，會自動：
        * 以 `--output-root/run-name/checkpoints/best.pt` 作為 checkpoint。
        * 將輸出圖存放於 `--output-root/run-name/diffusion_progress.png`。
    - `visualize_from_checkpoint` 函式可供其他模組（如訓練腳本）直接呼叫，以利在訓練結束後自動生成結果。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from diffusion import DiffusionConfig, DiffusionProcess
from model import UNet
from trainer import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize diffusion sampling trajectories.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="含有 `model` 欄位 state dict 的 checkpoint 檔案（若提供 --run-name 可省略）。",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="輸出 PNG 路徑，若提供 --run-name 則預設為該 run 目錄下的 diffusion_progress.png。",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="trained_model",
        help="run 目錄所屬的根資料夾（與訓練腳本的 --output-dir 對應）。",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="若指定，將自動在該 run 資料夾底下尋找 best checkpoint 並輸出圖檔。",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="同時追蹤的生成樣本數。")
    parser.add_argument("--segments", type=int, default=7, help="將 timestep 切成幾等份（結果圖會有 segments+1 欄）。")
    parser.add_argument("--model-image-size", type=int, default=32, help="模型訓練時的輸入尺寸。")
    parser.add_argument("--output-size", type=int, default=28, help="可選的輸出尺寸（例如題目要求的 28x28）。")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--residual-dropout", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    base_channels: int,
    attention_heads: int,
    residual_dropout: float,
) -> UNet:
    model = UNet(
        in_channels=3,
        base_channels=base_channels,
        attention_heads=attention_heads,
        residual_dropout=residual_dropout,
    ).to(device)
    checkpoint = torch.load(checkpoint_path.as_posix(), map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def sample_snapshots(
    model: UNet,
    diffusion: DiffusionProcess,
    batch_size: int,
    timesteps: int,
    segments: int,
    image_size: int,
) -> torch.Tensor:
    """
    回傳形狀為 [segments+1, batch_size, 1, H, W] 的張量，依時間排序。
    """

    if segments <= 0:
        raise ValueError("segments 必須為正整數。")
    x = torch.randn(batch_size, 3, image_size, image_size, device=diffusion.device)
    snapshots = [x.clone()]
    capture_schedule = torch.linspace(0, timesteps, steps=segments + 1, dtype=torch.long, device=diffusion.device)
    capture_schedule = capture_schedule.tolist()
    capture_schedule_iter = iter(capture_schedule[1:])  # 第一個（0）已儲存
    next_capture = next(capture_schedule_iter, None)
    steps_elapsed = 0

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((batch_size,), t, device=diffusion.device, dtype=torch.long)
        x = diffusion.p_sample(model, x, t_tensor)
        steps_elapsed += 1
        while next_capture is not None and steps_elapsed >= next_capture:
            snapshots.append(x.clone())
            next_capture = next(capture_schedule_iter, None)
        if next_capture is None:
            break

    if len(snapshots) != segments + 1:
        raise RuntimeError(
            f"取得的 snapshot 數量 {len(snapshots)} 與預期 {segments + 1} 不符，"
            "請確認 timesteps 與 segments 設定。"
        )
    return torch.stack(snapshots, dim=0)


def visualize_from_checkpoint(
    checkpoint_path: Path | str,
    output_path: Path | str,
    *,
    batch_size: int = 8,
    segments: int = 7,
    model_image_size: int = 32,
    output_size: int = 28,
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    base_channels: int = 64,
    attention_heads: int = 4,
    residual_dropout: float = 0.0,
    device: Optional[torch.device | str] = None,
    seed: Optional[int] = 3407,
) -> Path:
    if seed is not None:
        set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        base_channels=base_channels,
        attention_heads=attention_heads,
        residual_dropout=residual_dropout,
    )
    diffusion = DiffusionProcess(
        DiffusionConfig(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end),
        device=device,
    )
    snapshots = sample_snapshots(
        model=model,
        diffusion=diffusion,
        batch_size=batch_size,
        timesteps=timesteps,
        segments=segments,
        image_size=model_image_size,
    )

    if output_size != snapshots.shape[-1]:
        snapshots = F.interpolate(
            snapshots.view(-1, 3, snapshots.shape[-2], snapshots.shape[-1]),
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        ).view(snapshots.shape[0], snapshots.shape[1], 3, output_size, output_size)

    num_time = snapshots.shape[0]
    num_samples = snapshots.shape[1]
    ordered = []
    for time_idx in range(num_time):
        for sample_idx in range(num_samples):
            ordered.append(snapshots[time_idx, sample_idx])
    ordered = torch.stack(ordered, dim=0)
    ordered = ordered.clamp(-1.0, 1.0)
    ordered = (ordered + 1.0) / 2.0
    grid = make_grid(ordered, nrow=num_samples, padding=2, normalize=False)
    save_image(grid, output_path)
    print(f"已將 {snapshots.shape[1]} 條軌跡、每條 {num_time} 張影像輸出至：{output_path}")
    return output_path


@torch.no_grad()
def main() -> None:
    args = parse_args()

    run_dir = None
    if args.run_name:
        run_dir = Path(args.output_root).expanduser().resolve() / args.run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"找不到 run 目錄：{run_dir}")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if run_dir and checkpoint_path is None:
        candidate = run_dir / "checkpoints" / "best.pt"
        if not candidate.exists():
            raise FileNotFoundError(f"無法在 {candidate} 找到 best 模型，請指定 --checkpoint。")
        checkpoint_path = candidate
    if checkpoint_path is None:
        raise ValueError("必須指定 --checkpoint 或 --run-name。")

    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else None
    if run_dir and output_path is None:
        output_path = run_dir / "diffusion_progress.png"
    if output_path is None:
        raise ValueError("必須指定 --output-path 或 --run-name。")

    visualize_from_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        batch_size=args.batch_size,
        segments=args.segments,
        model_image_size=args.model_image_size,
        output_size=args.output_size,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        base_channels=args.base_channels,
        attention_heads=args.attention_heads,
        residual_dropout=args.residual_dropout,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

