"""
Spatial + pixel augmentation for registration network (R) pre-training.

Creates misaligned CTA pairs by applying:
  1. Small affine transforms: rotation, translation, scaling
  2. Elastic deformation: smooth random displacement field
  3. Pixel degradation: blur, noise, downsample (approximates G output quality)

The augmented CTA serves as the "source" image for R training, with the
original CTA as the "target". This teaches R to align spatial misalignment
before it sees actual G/G2 output during main training.

Usage:
    reg_aug = RegistrationAugmentation(max_angle=5.0, ...)
    source = reg_aug(cta)  # spatially misaligned + optionally degraded CTA
    reg_out = R(source, cta)  # R learns to align source → target
"""

import math
import random

import torch
import torch.nn.functional as F

from data.aug_utils import gaussian_blur_auto, downsample_upsample_auto


class RegistrationAugmentation:
    """Create spatially misaligned + quality-degraded CTA for R pre-training.

    Applies (in order):
      1. Random affine (rotation + translation + scale) — always applied
      2. Random elastic deformation — applied if alpha > 0
      3. Random pixel degradation (blur, noise, downsample) — if degrade=True

    All transforms are GPU-compatible (pure torch operations).

    Args:
        max_angle: max rotation angle in degrees (symmetric ±)
        max_translate: max translation as fraction of image size (symmetric ±)
        max_scale: max scale deviation from 1.0 (e.g. 0.05 → scale in [0.95, 1.05])
        elastic_alpha: intensity of elastic deformation in pixels (0=disabled)
        elastic_points: control grid size for elastic deformation (lower=smoother)
        degrade: whether to also apply pixel degradation
    """

    def __init__(
        self,
        max_angle: float = 5.0,
        max_translate: float = 0.03,
        max_scale: float = 0.05,
        elastic_alpha: float = 6.0,
        elastic_points: int = 8,
        degrade: bool = True,
    ):
        self.max_angle = max_angle
        self.max_translate = max_translate
        self.max_scale = max_scale
        self.elastic_alpha = elastic_alpha
        self.elastic_points = elastic_points
        self.degrade = degrade

    def __call__(self, cta: torch.Tensor) -> torch.Tensor:
        """Apply spatial misalignment + optional pixel degradation.

        Args:
            cta: (N, C, H, W) CTA tensor in [-1, 1]

        Returns:
            (N, C, H, W) augmented CTA clamped to [-1, 1]
        """
        x = cta
        x = self._affine_transform(x)
        if self.elastic_alpha > 0:
            x = self._elastic_deform(x)
        if self.degrade:
            x = self._pixel_degrade(x)
        return x.clamp(-1.0, 1.0)

    def _affine_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random 2D affine: rotation + translation + scale."""
        N, C, H, W = x.shape
        device = x.device

        # Sample random parameters per batch element
        angles = torch.empty(N, device=device).uniform_(
            -self.max_angle, self.max_angle
        ) * (math.pi / 180.0)
        scales = torch.empty(N, device=device).uniform_(
            1.0 - self.max_scale, 1.0 + self.max_scale
        )
        tx = torch.empty(N, device=device).uniform_(
            -self.max_translate, self.max_translate
        )
        ty = torch.empty(N, device=device).uniform_(
            -self.max_translate, self.max_translate
        )

        # Build 2×3 affine matrices
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        theta = torch.zeros(N, 2, 3, device=device, dtype=x.dtype)
        theta[:, 0, 0] = cos_a * scales
        theta[:, 0, 1] = -sin_a * scales
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a * scales
        theta[:, 1, 1] = cos_a * scales
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

    def _elastic_deform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random elastic deformation via low-res displacement field.

        Generates random displacement at a low-resolution control grid,
        then upsamples to full resolution for smooth deformation.
        This is much more efficient than Gaussian-blurred full-res displacement.
        """
        N, C, H, W = x.shape
        device = x.device
        cp = self.elastic_points

        # Random displacement at low resolution (control points)
        # Scale by alpha/max(H,W) to get normalized coordinate displacement
        scale_factor = self.elastic_alpha / max(H, W)
        dx = torch.randn(N, 1, cp, cp, device=device) * scale_factor
        dy = torch.randn(N, 1, cp, cp, device=device) * scale_factor

        # Upsample to full resolution (bicubic for smooth field)
        dx = F.interpolate(dx, size=(H, W), mode="bicubic", align_corners=False)
        dy = F.interpolate(dy, size=(H, W), mode="bicubic", align_corners=False)

        # Build base identity grid in [-1, 1]
        grid_y = torch.linspace(-1, 1, H, device=device, dtype=x.dtype)
        grid_x = torch.linspace(-1, 1, W, device=device, dtype=x.dtype)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        # (1, H, W, 2)
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(N, -1, -1, -1)

        # Add displacement: dx affects x-coord, dy affects y-coord
        disp = torch.cat([dx, dy], dim=1)  # (N, 2, H, W)
        disp = disp.permute(0, 2, 3, 1)  # (N, H, W, 2)
        grid = base_grid + disp

        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

    @staticmethod
    def _pixel_degrade(x: torch.Tensor) -> torch.Tensor:
        """Apply mild pixel degradation (blur + noise + downsample).

        Uses shared auto-selecting functions from aug_utils:
        - gaussian_blur_auto: 3D blur for C >= 4, 2D otherwise
        - downsample_upsample_auto: 3D trilinear for C >= 4, 2D bilinear otherwise
        """
        x = x.clone()

        # Gaussian blur (50% chance) — auto 2D/3D
        if random.random() < 0.5:
            ks = random.choice([3, 5])
            sigma = random.uniform(0.3, 1.0)
            x = gaussian_blur_auto(x, kernel_size=ks, sigma=sigma)

        # Gaussian noise (50% chance)
        if random.random() < 0.5:
            std = random.uniform(0.01, 0.04)
            x = x + torch.randn_like(x) * std

        # Downsample-upsample (30% chance) — auto 2D/3D
        if random.random() < 0.3:
            scale = random.choice([2, 3])
            x = downsample_upsample_auto(x, scale)

        return x

    def __repr__(self):
        return (
            f"RegistrationAugmentation("
            f"max_angle={self.max_angle}, "
            f"max_translate={self.max_translate}, "
            f"max_scale={self.max_scale}, "
            f"elastic_alpha={self.elastic_alpha}, "
            f"elastic_points={self.elastic_points}, "
            f"degrade={self.degrade})"
        )
