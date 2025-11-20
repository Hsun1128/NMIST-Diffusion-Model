"""針對 28x28 MNIST 影像重新實作的精簡 U-Net。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """標準的 sinusoidal timestep embedding。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timestep.device) * -emb_scale)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def get_group_norm(channels: int, max_groups: int = 16) -> nn.GroupNorm:
    """動態選擇 GroupNorm 的 groups，避免無法整除的情況。"""
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    """簡化版殘差區塊，使用 FiLM 式時間調整。"""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = get_group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = get_group_norm(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        time_emb = self.time_emb(t)[:, :, None, None]
        h = h + time_emb
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.residual(x)


class DownBlock(nn.Module):
    """兩個殘差區塊 + 可選下採樣。"""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, downsample: bool) -> None:
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1) if downsample else None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, t)
        x = self.res2(x, t)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """與 DownBlock 對稱的上採樣模組。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int, upsample: bool) -> None:
        super().__init__()
        self.res1 = ResidualBlock(in_channels + skip_channels, out_channels, time_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2) if upsample else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t)
        x = self.res2(x, t)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class UNet(nn.Module):
    """精簡版 U-Net，專為 28x28 MNIST 影像設計。"""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 256,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if time_dim < base_channels * 2:
            time_dim = base_channels * 2

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.initial = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList(
            [
                DownBlock(base_channels, base_channels, time_dim, downsample=True),
                DownBlock(base_channels, base_channels * 2, time_dim, downsample=True),
                DownBlock(base_channels * 2, base_channels * 4, time_dim, downsample=False),
            ]
        )

        self.mid_block1 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim, residual_dropout)
        self.mid_block2 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim, residual_dropout)

        self.up_blocks = nn.ModuleList(
            [
                UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_dim, upsample=True),
                UpBlock(base_channels * 2, base_channels * 2, base_channels, time_dim, upsample=True),
                UpBlock(base_channels, base_channels, base_channels, time_dim, upsample=False),
            ]
        )

        self.final = nn.Sequential(
            get_group_norm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t = self.time_embed(timestep)
        x = self.initial(x)

        skips = []
        for block in self.down_blocks:
            x, skip = block(x, t)
            skips.append(skip)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for block in self.up_blocks:
            skip = skips.pop()
            x = block(x, skip, t)

        return self.final(x)

