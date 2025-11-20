"""
資料相關的協助函式與資料集實作。

這個模組提供：
    - `MNISTImageFolder`：讀取 `@mnist` 目錄下的 PNG 檔案並轉為張量。
    - `build_dataloader`：依指定批次大小與資料載入參數建立 `DataLoader`。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MNISTImageFolder(Dataset):
    """
    以硬碟上的 PNG 檔案作為影像來源的簡易資料集。

    由於題目禁止直接下載外部資料集，我們改從使用者提供的 `@mnist`
    目錄中讀取資料，每張圖都會被正規化到 [-1, 1]。
    """

    def __init__(self, root: str, image_size: int = 32, transform: transforms.Compose | None = None) -> None:
        if root.startswith("@"):
            root = root[1:]
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory {self.root} 不存在，請確認 @mnist 的掛載。")

        self.paths: List[Path] = sorted(self.root.glob("*.png"))
        if not self.paths:
            raise RuntimeError(f"{self.root} 內沒有任何 PNG 圖檔，無法進行訓練。")

        # 預設轉換：轉成 RGB 張量並縮放到 [-1, 1]，以符合 DDPM 訓練慣例。
        # 即使原始圖像是灰階，也轉換為 RGB 格式以支持彩色去噪過程。
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # MNIST 為無條件生成任務，因此 label 不使用，統一回傳 0 作為 placeholder。
        return image, 0


def build_dataloader(
    root: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    建立訓練用的 `DataLoader`，並啟用 shuffle、pinned memory 等常見設定。
    """

    dataset = MNISTImageFolder(root, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

