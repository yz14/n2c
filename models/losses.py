"""
Loss functions for NCCT→CTA image translation.

Provides:
  - SSIM loss (differentiable structural similarity)
  - Combined L1 + SSIM loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _gaussian_filter(input: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """
    Apply separable Gaussian filter to input.

    Args:
        input:  (N, C, H, W) tensor
        window: 1D Gaussian kernel

    Returns:
        Filtered (N, C, H, W) tensor
    """
    C = input.shape[1]
    # Create 2D separable kernel
    window = window.to(input.device, input.dtype)
    # Horizontal pass
    win_h = window.reshape(1, 1, 1, -1).repeat(C, 1, 1, 1)
    # Vertical pass
    win_v = window.reshape(1, 1, -1, 1).repeat(C, 1, 1, 1)
    pad = window.shape[0] // 2
    out = F.conv2d(input, win_h, padding=(0, pad), groups=C)
    out = F.conv2d(out, win_v, padding=(pad, 0), groups=C)
    return out


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Compute SSIM between two image tensors.

    Args:
        x, y:         (N, C, H, W) tensors in the same value range
        window_size:  Gaussian window size
        sigma:        Gaussian sigma
        data_range:   value range (2.0 for [-1, 1] normalized data)
        size_average: if True, return scalar mean; else return per-sample values

    Returns:
        SSIM value(s) in [0, 1]
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    window = _fspecial_gauss_1d(window_size, sigma)

    mu_x = _gaussian_filter(x, window)
    mu_y = _gaussian_filter(y, window)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = _gaussian_filter(x ** 2, window) - mu_x_sq
    sigma_y_sq = _gaussian_filter(y ** 2, window) - mu_y_sq
    sigma_xy = _gaussian_filter(x * y, window) - mu_xy

    # Clamp for numerical stability
    sigma_x_sq = sigma_x_sq.clamp(min=0)
    sigma_y_sq = sigma_y_sq.clamp(min=0)

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.flatten(1).mean(dim=1)


class SSIMLoss(nn.Module):
    """SSIM loss: 1 - SSIM."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 2.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(
            pred, target,
            window_size=self.window_size,
            sigma=self.sigma,
            data_range=self.data_range,
        )


class CombinedLoss(nn.Module):
    """
    Combined L1 + SSIM loss for image translation.

    loss = l1_weight * L1(pred, target) + ssim_weight * (1 - SSIM(pred, target))
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 1.0,
        window_size: int = 11,
        data_range: float = 2.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=window_size, data_range=data_range)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute combined loss.

        Args:
            pred:   (N, C, H, W) predicted CTA
            target: (N, C, H, W) ground truth CTA

        Returns:
            dict with keys: loss, l1, ssim
        """
        l1 = self.l1_loss(pred, target)
        ssim_val = self.ssim_loss(pred, target)
        total = self.l1_weight * l1 + self.ssim_weight * ssim_val
        return {
            "loss": total,
            "l1": l1.detach(),
            "ssim": ssim_val.detach(),
        }
