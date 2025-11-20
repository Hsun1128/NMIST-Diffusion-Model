"""
資料相關的協助函式與資料集實作。

這個模組提供：
    - `MNISTImageFolder`：讀取 `mnist` 目錄下的 PNG 檔案並轉為張量。
    - `build_dataloader`：依指定批次大小與資料載入參數建立 `DataLoader`。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class MNISTImageFolder(Dataset):
    """
    以硬碟上的 PNG 檔案作為影像來源的簡易資料集。

    由於題目禁止直接下載外部資料集，我們改從使用者提供的 `mnist`
    目錄中讀取資料，每張圖都會被正規化到 [-1, 1]。
    """

    def __init__(self, root: str, image_size: int = 32, transform: transforms.Compose | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory {self.root} 不存在，請確認 mnist 的掛載。")

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
    train_split: float | None = None,
    is_train: bool = True,
    seed: int = 3407,
) -> DataLoader:
    """
    建立 `DataLoader`，支援 train/val 劃分。
    
    Args:
        root: 數據目錄
        batch_size: 批次大小
        num_workers: 數據載入工作線程數
        image_size: 圖像尺寸
        shuffle: 是否打亂順序
        drop_last: 是否丟棄最後一個不完整的批次
        train_split: 訓練集比例（例如 0.8 表示 80% 訓練，20% 驗證）。若為 None 則不劃分。
        is_train: 是否為訓練集（True 為訓練集，False 為驗證集）
        seed: 隨機種子，確保劃分可重現
    """
    full_dataset = MNISTImageFolder(root, image_size=image_size)
    
    if train_split is not None:
        # 使用固定種子確保劃分可重現
        generator = torch.Generator().manual_seed(seed)
        train_size = int(len(full_dataset) * train_split)
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        dataset = train_dataset if is_train else val_dataset
    else:
        dataset = full_dataset
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

