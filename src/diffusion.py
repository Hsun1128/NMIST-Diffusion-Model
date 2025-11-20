"""前向/反向擴散流程與取樣 API。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm.auto import tqdm


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


class DiffusionProcess:
    """
    封裝 DDPM 中所有與時間序列相關的張量，避免在訓練迴圈重複計算。
    """

    def __init__(self, config: DiffusionConfig, device: torch.device) -> None:
        self.config = config
        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.device = device

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.config.timesteps, (batch_size,), device=self.device)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None, None]

        model_mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * model(x, t))
        if (t == 0).all():
            return model_mean
        posterior_var_t = self.posterior_variance[t][:, None, None, None]
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, image_size: int, batch_size: int, channels: int) -> torch.Tensor:
        """
        反覆呼叫 `p_sample`，從標準高斯開始逐步還原出可視影像。
        """

        x = torch.randn(batch_size, channels, image_size, image_size, device=self.device)
        for t in tqdm(
            reversed(range(self.config.timesteps)),
            total=self.config.timesteps,
            desc="Sampling",
            leave=False,
        ):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t_tensor)
        return x

