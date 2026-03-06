"""
Shared utilities for LDM trainers.

Contains common classes and functions used by both VAE (Stage 1)
and Diffusion (Stage 2) trainers to avoid code duplication.
"""

import math
from typing import Dict

import torch


class MetricTracker:
    """Track running averages of metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = {}
        self._count = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self._sum[k] = self._sum.get(k, 0.0) + v * n
            self._count[k] = self._count.get(k, 0) + n

    def result(self) -> Dict[str, float]:
        return {k: self._sum[k] / self._count[k] for k in self._sum}

    def __str__(self):
        return ", ".join(f"{k}: {v:.6f}" for k, v in self.result().items())


def warmup_cosine_schedule(warmup_steps: int, total_steps: int):
    """Create a warmup + cosine decay LR lambda."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# ---------------------------------------------------------------------------
# SSIM loss (differentiable)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Create a 1D Gaussian kernel."""
    x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_kernel_2d(kernel_size: int, sigma: float,
                        channels: int) -> torch.Tensor:
    """Create a 2D Gaussian kernel for depthwise convolution."""
    k1d = _gaussian_kernel_1d(kernel_size, sigma)
    k2d = k1d.unsqueeze(-1) @ k1d.unsqueeze(0)  # (K, K)
    kernel = k2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """Differentiable SSIM loss: 1 - SSIM(pred, target).

    Implements the Structural Similarity Index (Wang et al., 2004)
    using Gaussian-windowed statistics. Returns 1 - SSIM so it can
    be directly minimized as a loss.

    Assumes input range [-1, 1], so C1/C2 use data_range=2.0 scaling:
        C1 = (0.01 * 2)^2, C2 = (0.03 * 2)^2

    Args:
        pred:        (B, C, H, W) predicted image
        target:      (B, C, H, W) ground truth image
        kernel_size: Gaussian window size (must be odd)
        sigma:       Gaussian standard deviation
        c1, c2:      stability constants
        reduction:   "mean" or "none"

    Returns:
        Scalar loss (1 - SSIM) if reduction="mean", else per-sample loss.
    """
    # Scale C1, C2 for data range of 2.0 (input in [-1, 1])
    data_range = 2.0
    C1 = (c1 * data_range) ** 2
    C2 = (c2 * data_range) ** 2

    channels = pred.shape[1]
    kernel = _gaussian_kernel_2d(kernel_size, sigma, channels).to(
        device=pred.device, dtype=pred.dtype
    )
    pad = kernel_size // 2

    # Compute windowed statistics via depthwise convolution
    mu_x = torch.nn.functional.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_y = torch.nn.functional.conv2d(target, kernel, padding=pad, groups=channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = torch.nn.functional.conv2d(
        pred ** 2, kernel, padding=pad, groups=channels
    ) - mu_x_sq
    sigma_y_sq = torch.nn.functional.conv2d(
        target ** 2, kernel, padding=pad, groups=channels
    ) - mu_y_sq
    sigma_xy = torch.nn.functional.conv2d(
        pred * target, kernel, padding=pad, groups=channels
    ) - mu_xy

    # SSIM formula
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )

    if reduction == "mean":
        return 1.0 - ssim_map.mean()
    else:
        return 1.0 - ssim_map.flatten(1).mean(dim=1)  # (B,)
