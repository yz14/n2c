"""
Loss functions for NCCT→CTA image translation.

Provides:
  - 2D SSIM loss (differentiable structural similarity)
  - 3D SSIM loss (treats C as depth dimension)
  - Mask-weighted L1 + SSIM loss (lung region 10x, other 1x)
  - GANLoss (LSGAN)
  - FeatureMatchingLoss (multi-scale discriminator feature matching)
  - GradLoss (deformation field smoothness for registration)
"""

from typing import List, Optional

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


def ssim_2d_map(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Compute per-pixel 2D SSIM map. Returns (N, C, H, W) SSIM values in [0, 1]."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    window = _fspecial_gauss_1d(window_size, sigma)

    mu_x = _gaussian_filter_2d(x, window)
    mu_y = _gaussian_filter_2d(y, window)
    sigma_x_sq = (_gaussian_filter_2d(x ** 2, window) - mu_x ** 2).clamp(min=0)
    sigma_y_sq = (_gaussian_filter_2d(y ** 2, window) - mu_y ** 2).clamp(min=0)
    sigma_xy = _gaussian_filter_2d(x * y, window) - mu_x * mu_y

    return ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )


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
# Mask-weighted helpers
# ---------------------------------------------------------------------------

def _build_weight_map(
    mask: Optional[torch.Tensor],
    shape: tuple,
    lung_weight: float = 10.0,
    bg_weight: float = 1.0,
) -> Optional[torch.Tensor]:
    """
    Build per-pixel weight map from lung mask.

    Args:
        mask:        (N, C, H, W) binary lung mask, or None
        shape:       target shape (for size matching)
        lung_weight: weight for lung voxels
        bg_weight:   weight for background voxels

    Returns:
        (N, C, H, W) weight tensor, or None if mask is None
    """
    if mask is None:
        return None
    w = torch.where(mask > 0.5, lung_weight, bg_weight)
    # Normalize so that mean weight ≈ 1 (preserves loss magnitude scale)
    w = w / w.mean()
    return w


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


class WeightedL1Loss(nn.Module):
    """L1 loss with optional per-pixel weighting from a mask."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = (pred - target).abs()
        if weight is not None:
            diff = diff * weight
        return diff.mean()


class WeightedSSIMLoss(nn.Module):
    """
    Weighted SSIM loss.

    For 2D mode: computes per-pixel SSIM map, applies mask weighting, then averages.
    For 3D mode: falls back to standard (unweighted) 3D SSIM since the 5D reshape
    makes per-pixel weighting non-trivial; the mask still guides L1 loss.
    """

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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_3d or weight is None:
            fn = ssim_3d if self.use_3d else ssim_2d
            return 1.0 - fn(
                pred, target,
                window_size=self.window_size,
                sigma=self.sigma,
                data_range=self.data_range,
            )
        # 2D weighted: compute SSIM map and apply weight
        ssim_map = ssim_2d_map(
            pred, target,
            window_size=self.window_size,
            sigma=self.sigma,
            data_range=self.data_range,
        )
        # 1 - weighted_mean(ssim_map)
        loss_map = 1.0 - ssim_map
        return (loss_map * weight).mean()


class CombinedLoss(nn.Module):
    """
    Combined L1 + SSIM loss with optional mask weighting.

    loss = l1_weight * WeightedL1 + ssim_weight * WeightedSSIM

    When mask is provided, lung regions get `lung_weight` (default 10x)
    and background gets 1x weighting.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 1.0,
        window_size: int = 11,
        data_range: float = 2.0,
        use_3d_ssim: bool = False,
        lung_weight: float = 10.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.lung_weight = lung_weight
        self.l1_loss = WeightedL1Loss()
        self.ssim_loss = WeightedSSIMLoss(
            window_size=window_size, data_range=data_range, use_3d=use_3d_ssim
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            pred:   (N, C, H, W) predicted CTA
            target: (N, C, H, W) ground truth CTA
            mask:   (N, C, H, W) lung mask (optional)

        Returns:
            dict with keys: loss, l1, ssim
        """
        weight = _build_weight_map(mask, pred.shape, self.lung_weight)
        l1 = self.l1_loss(pred, target, weight)
        ssim_val = self.ssim_loss(pred, target, weight)
        total = self.l1_weight * l1 + self.ssim_weight * ssim_val
        return {
            "loss": total,
            "l1": l1.detach(),
            "ssim": ssim_val.detach(),
        }


# ---------------------------------------------------------------------------
# GAN losses
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """
    LSGAN loss for multi-scale discriminator.

    Uses MSE loss: D should output ~1 for real, ~0 for fake.
    Supports multi-scale discriminator output format:
      List[List[Tensor]] — outer list = scales, inner list = layer features.
      The last element of each inner list is the final prediction.
    """

    def __init__(self, target_real: float = 1.0, target_fake: float = 0.0):
        super().__init__()
        self.target_real = target_real
        self.target_fake = target_fake
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: List[List[torch.Tensor]],
        target_is_real: bool,
        target_value: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute LSGAN loss over all discriminator scales.

        Args:
            predictions:    output from MultiscaleDiscriminator.forward()
            target_is_real: True for real images, False for fake
            target_value:   override target value (e.g. 0.9 for label smoothing).
                            If None, uses target_real/target_fake based on target_is_real.

        Returns:
            scalar loss
        """
        if target_value is not None:
            val = target_value
        else:
            val = self.target_real if target_is_real else self.target_fake
        loss = 0.0
        for scale_preds in predictions:
            pred = scale_preds[-1]  # final layer output
            target_tensor = torch.full_like(pred, val)
            loss = loss + self.mse(pred, target_tensor)
        return loss


class HingeGANLoss(nn.Module):
    """
    Hinge GAN loss for multi-scale discriminator.

    More robust than LSGAN: once D correctly classifies, gradients vanish
    (no "overshooting"). Widely used in SAGAN, BigGAN, ProjectedGAN.

    D_real: max(0, 1 - D(real))    → pushes D(real) above 1
    D_fake: max(0, 1 + D(fake))    → pushes D(fake) below -1
    G_fake: -D(fake)               → pushes D(fake) up

    Compatible with MultiscaleDiscriminator output format:
      List[List[Tensor]] — last element per scale is the prediction.
    """

    def forward(
        self,
        predictions: List[List[torch.Tensor]],
        target_is_real: bool,
        for_discriminator: bool = True,
        target_value: Optional[float] = None,  # unused, for API compat
    ) -> torch.Tensor:
        loss = 0.0
        for scale_preds in predictions:
            pred = scale_preds[-1]
            if for_discriminator:
                if target_is_real:
                    loss = loss + F.relu(1.0 - pred).mean()
                else:
                    loss = loss + F.relu(1.0 + pred).mean()
            else:
                # Generator loss: wants D(fake) to be high
                loss = loss - pred.mean()
        return loss


def r1_gradient_penalty(
    real_pred: List[List[torch.Tensor]],
    real_input: torch.Tensor,
) -> torch.Tensor:
    """
    R1 gradient penalty (Mescheder et al., 2018).

    Penalizes the gradient of D's output w.r.t. real inputs:
        R1 = (1/2) * E[||∇D(x_real)||²]

    This is the single most effective GAN stabilization technique.
    Used in StyleGAN2, ProjectedGAN, etc.

    The caller must ensure real_input.requires_grad_(True) before D forward.

    Args:
        real_pred: D's output on real images (multi-scale format)
        real_input: the real input tensor (must have requires_grad=True)

    Returns:
        scalar R1 penalty (without the γ/2 factor — caller applies weight)
    """
    # Sum all scale predictions to get a single scalar for grad computation
    pred_sum = sum(s[-1].sum() for s in real_pred)
    gradients = torch.autograd.grad(
        outputs=pred_sum,
        inputs=real_input,
        create_graph=True,
        only_inputs=True,
    )[0]
    # ||∇D||² per sample, then mean over batch
    return gradients.pow(2).flatten(1).sum(1).mean()


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss between real and fake discriminator features.

    Encourages the generator to produce features that match the real
    distribution at each layer of each discriminator scale.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        real_features: List[List[torch.Tensor]],
        fake_features: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute feature matching loss.

        Args:
            real_features: discriminator features on real images
            fake_features: discriminator features on fake images

        Returns:
            scalar loss
        """
        loss = 0.0
        num_layers = 0
        for real_scale, fake_scale in zip(real_features, fake_features):
            # Match all intermediate layers (exclude final prediction)
            for real_feat, fake_feat in zip(real_scale[:-1], fake_scale[:-1]):
                loss = loss + self.l1(fake_feat, real_feat.detach())
                num_layers += 1
        if num_layers > 0:
            loss = loss / num_layers
        return loss


# ---------------------------------------------------------------------------
# Registration losses
# ---------------------------------------------------------------------------

class GradLoss(nn.Module):
    """
    Gradient smoothness loss for 2D deformation/displacement fields.

    Penalizes spatial gradients of the displacement field to encourage
    smooth deformations. Uses L1 or L2 penalty.

    Args:
        penalty: 'l1' or 'l2'
    """

    def __init__(self, penalty: str = "l2"):
        super().__init__()
        assert penalty in ("l1", "l2")
        self.penalty = penalty

    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness loss.

        Args:
            displacement: (N, 2, H, W) displacement field

        Returns:
            scalar loss
        """
        # Spatial differences along H and W
        dy = displacement[:, :, 1:, :] - displacement[:, :, :-1, :]
        dx = displacement[:, :, :, 1:] - displacement[:, :, :, :-1]

        if self.penalty == "l2":
            return (dy ** 2).mean() + (dx ** 2).mean()
        else:
            return dy.abs().mean() + dx.abs().mean()
