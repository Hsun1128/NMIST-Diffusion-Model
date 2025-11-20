"""
U-Net 架構與其組成模組。

為了讓整個 diffusion model 易於理解與維護，本檔案整理了：
    - 時間步嵌入 `SinusoidalPosEmb`
    - 殘差區塊 `ResidualBlock`
    - 自注意力區塊 `AttentionBlock`
    - 主體模型 `UNet`
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """將離散 timestep 轉換為連續 embedding 的模組。"""

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


def get_group_norm(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """根據通道數動態選擇 GroupNorm 的 group 數，確保能被整除。"""
    num_groups = min(num_groups, channels)
    while channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, channels)


class ResidualBlock(nn.Module):
    """
    Diffusion 模型常用的殘差區塊，透過 FiLM 式時間調整與 dropout 改善表現。
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels * 2))
        self.block1 = nn.Sequential(
            get_group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            get_group_norm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time_emb = self.time_mlp(t)
        scale, shift = time_emb.chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.block2(h)
        h = self.dropout(h)
        return h + self.residual(x)


class AttentionBlock(nn.Module):
    """在空間維度套用多頭自注意力，協助模型捕捉長距依賴。"""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) 必須能被 num_heads ({num_heads}) 整除。")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.group_norm = get_group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.group_norm(x)
        q, k, v = self.qkv(h_).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w)
        attn = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return self.proj(out) + x


class UNet(nn.Module):
    """
    U-Net 主體：採用 Encoder-Decoder + Skip Connection 架構，
    並在多個尺度插入 Attention 以提升生成品質。
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        time_dim: int = 256,
        attention_heads: int = 4,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        # 改進的時間嵌入，使用更深的網路
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        self.initial = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        # 下採樣路徑
        self.down1 = nn.ModuleList(
            [
                ResidualBlock(base_channels, base_channels, time_dim, residual_dropout),
                ResidualBlock(base_channels, base_channels, time_dim, residual_dropout),
                AttentionBlock(base_channels, attention_heads),
            ]
        )
        self.down2 = nn.ModuleList(
            [
                ResidualBlock(base_channels, base_channels * 2, time_dim, residual_dropout),
                ResidualBlock(base_channels * 2, base_channels * 2, time_dim, residual_dropout),
                AttentionBlock(base_channels * 2, attention_heads),
            ]
        )
        self.down3 = nn.ModuleList(
            [
                ResidualBlock(base_channels * 2, base_channels * 4, time_dim, residual_dropout),
                ResidualBlock(base_channels * 4, base_channels * 4, time_dim, residual_dropout),
                AttentionBlock(base_channels * 4, attention_heads),
            ]
        )

        # Bottleneck
        self.mid = nn.ModuleList(
            [
                ResidualBlock(base_channels * 4, base_channels * 4, time_dim, residual_dropout),
                AttentionBlock(base_channels * 4, attention_heads),
                ResidualBlock(base_channels * 4, base_channels * 4, time_dim, residual_dropout),
            ]
        )

        # 上採樣路徑
        self.up3 = nn.ModuleList(
            [
                ResidualBlock(base_channels * 8, base_channels * 2, time_dim, residual_dropout),
                ResidualBlock(base_channels * 2, base_channels * 2, time_dim, residual_dropout),
                AttentionBlock(base_channels * 2, attention_heads),
            ]
        )
        self.up2 = nn.ModuleList(
            [
                ResidualBlock(base_channels * 4, base_channels, time_dim, residual_dropout),
                ResidualBlock(base_channels, base_channels, time_dim, residual_dropout),
                AttentionBlock(base_channels, attention_heads),
            ]
        )
        self.up1 = nn.ModuleList(
            [
                ResidualBlock(base_channels * 2, base_channels, time_dim, residual_dropout),
                ResidualBlock(base_channels, base_channels, time_dim, residual_dropout),
                AttentionBlock(base_channels, attention_heads),
            ]
        )

        self.final = nn.Sequential(
            get_group_norm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )
        
        # 使用 stride=2 的 Conv2d 進行下採樣，替代 AvgPool2d
        self.downsample1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        self.downsample3 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)
        
        # 初始化權重
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """使用 Xavier 初始化改善訓練穩定性。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t = self.time_embed(timestep)
        x = self.initial(x)

        # 下採樣路徑
        d1 = []
        for layer in self.down1:
            x = layer(x, t) if isinstance(layer, ResidualBlock) else layer(x)
            d1.append(x)
        x = self.downsample1(x)

        d2 = []
        for layer in self.down2:
            x = layer(x, t) if isinstance(layer, ResidualBlock) else layer(x)
            d2.append(x)
        x = self.downsample2(x)

        d3 = []
        for layer in self.down3:
            x = layer(x, t) if isinstance(layer, ResidualBlock) else layer(x)
            d3.append(x)
        x = self.downsample3(x)

        # Bottleneck
        for layer in self.mid:
            x = layer(x, t) if isinstance(layer, ResidualBlock) else layer(x)

        # 上採樣路徑，使用 bilinear 插值替代 nearest
        for idx, layer in enumerate(self.up3):
            if idx == 0:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
                x = torch.cat([x, d3[-1]], dim=1)
            x = layer(x, t) if isinstance(layer, ResidualBlock) else layer(x)

        for idx, layer in enumerate(self.up2):
            if idx == 0:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
                x = torch.cat([x, d2[-1]], dim=1)
            x = layer(x, t) if isinstance(layer, ResidualBlock) else layer(x)

        for idx, layer in enumerate(self.up1):
            if idx == 0:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
                x = torch.cat([x, d1[-1]], dim=1)
            x = layer(x, t) if isinstance(layer, ResidualBlock) else layer(x)

        return self.final(x)

