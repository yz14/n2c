"""
Loss functions for NCCT→CTA image translation.

Provides:
  - 2D SSIM loss (differentiable structural similarity)
  - 3D SSIM loss (treats C as depth dimension)
  - Combined L1 + SSIM loss (supports both 2D and 3D SSIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2D SSIM
# ---------------------------------------------------------------------------

def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _gaussian_filter_2d(input: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """
    Apply separable 2D Gaussian filter.

    Args:
        input:  (N, C, H, W) tensor
        window: 1D Gaussian kernel

    Returns:
        Filtered (N, C, H, W) tensor
    """
    C = input.shape[1]
    window = window.to(input.device, input.dtype)
    win_h = window.reshape(1, 1, 1, -1).repeat(C, 1, 1, 1)
    win_v = window.reshape(1, 1, -1, 1).repeat(C, 1, 1, 1)
    pad = window.shape[0] // 2
    out = F.conv2d(input, win_h, padding=(0, pad), groups=C)
    out = F.conv2d(out, win_v, padding=(pad, 0), groups=C)
    return out


def ssim_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Compute 2D SSIM between two image tensors.

    Args:
        x, y:         (N, C, H, W) tensors
        window_size:  Gaussian window size
        sigma:        Gaussian sigma
        data_range:   value range (2.0 for [-1, 1])
        size_average: return scalar mean if True

    Returns:
        SSIM value(s) in [0, 1]
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    window = _fspecial_gauss_1d(window_size, sigma)

    mu_x = _gaussian_filter_2d(x, window)
    mu_y = _gaussian_filter_2d(y, window)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = (_gaussian_filter_2d(x ** 2, window) - mu_x_sq).clamp(min=0)
    sigma_y_sq = (_gaussian_filter_2d(y ** 2, window) - mu_y_sq).clamp(min=0)
    sigma_xy = _gaussian_filter_2d(x * y, window) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    if size_average:
        return ssim_map.mean()
    return ssim_map.flatten(1).mean(dim=1)


# ---------------------------------------------------------------------------
# 3D SSIM
# ---------------------------------------------------------------------------

def _gaussian_filter_3d(input: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """
    Apply separable 3D Gaussian filter.

    Args:
        input:  (N, 1, D, H, W) tensor
        window: 1D Gaussian kernel

    Returns:
        Filtered (N, 1, D, H, W) tensor
    """
    window = window.to(input.device, input.dtype)
    pad = window.shape[0] // 2
    # Depth pass
    win_d = window.reshape(1, 1, -1, 1, 1)
    out = F.conv3d(input, win_d, padding=(pad, 0, 0))
    # Height pass
    win_h = window.reshape(1, 1, 1, -1, 1)
    out = F.conv3d(out, win_h, padding=(0, pad, 0))
    # Width pass
    win_w = window.reshape(1, 1, 1, 1, -1)
    out = F.conv3d(out, win_w, padding=(0, 0, pad))
    return out


def ssim_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 7,
    sigma: float = 1.5,
    data_range: float = 2.0,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Compute 3D SSIM. Input (N, C, H, W) is treated as (N, 1, D=C, H, W).

    A smaller default window_size (7) is used because the D dimension
    is typically small (e.g. 3 or 12). The window is clamped to min(window_size, D).

    Args:
        x, y:         (N, C, H, W) tensors
        window_size:  Gaussian window size (clamped to D if D < window_size)
        sigma:        Gaussian sigma
        data_range:   value range (2.0 for [-1, 1])
        size_average: return scalar mean if True

    Returns:
        SSIM value(s) in [0, 1]
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    D = x.shape[1]
    # Clamp window size to depth dimension
    ws = min(window_size, D)
    if ws % 2 == 0:
        ws = max(ws - 1, 1)  # ensure odd
    window = _fspecial_gauss_1d(ws, sigma)

    # Reshape to 5D: (N, 1, D, H, W)
    x_5d = x.unsqueeze(1)
    y_5d = y.unsqueeze(1)

    mu_x = _gaussian_filter_3d(x_5d, window)
    mu_y = _gaussian_filter_3d(y_5d, window)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = (_gaussian_filter_3d(x_5d ** 2, window) - mu_x_sq).clamp(min=0)
    sigma_y_sq = (_gaussian_filter_3d(y_5d ** 2, window) - mu_y_sq).clamp(min=0)
    sigma_xy = _gaussian_filter_3d(x_5d * y_5d, window) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    if size_average:
        return ssim_map.mean()
    return ssim_map.flatten(1).mean(dim=1)


# ---------------------------------------------------------------------------
# Loss modules
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """SSIM loss: 1 - SSIM. Supports 2D and 3D modes."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 2.0,
        use_3d: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.use_3d = use_3d

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fn = ssim_3d if self.use_3d else ssim_2d
        return 1.0 - fn(
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
        use_3d_ssim: bool = False,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(
            window_size=window_size, data_range=data_range, use_3d=use_3d_ssim
        )

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
