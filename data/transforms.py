"""
GPU-based 3D data augmentation for 2.5D medical image slices.

Key design:
  - All augmentation runs on GPU in batched mode for speed.
  - Spatial transforms treat (3C, H, W) as a 3D volume (1, D, H, W)
    using 3D affine_grid + grid_sample — rotation around D axis,
    scaling in H-W plane, translation, and horizontal flip.
  - Pixel transforms (brightness, contrast, noise) are applied only to ncct.
  - Middle C slices are extracted after augmentation.

Usage:
    augmentor = GPUAugmentor(data_config, num_slices=3)
    ncct, cta, mask = augmentor(ncct_3c, cta_3c, mask_3c, training=True)
    # Input:  (N, 3C, H, W) on GPU
    # Output: (N, C, H, W) on GPU
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple


class GPUAugmentor:
    """
    GPU-based 3D augmentor that processes batches on device.

    Spatial augmentation uses 3D affine transforms (rotation around D axis,
    isotropic H-W scaling, translation, horizontal flip). Pixel augmentation
    applies per-sample brightness, contrast, and Gaussian noise to ncct only.
    """

    def __init__(
        self,
        num_slices: int = 3,
        aug_prob: float = 0.5,
        max_angle: float = 15.0,
        scale_range: float = 0.1,
        translate_frac: float = 0.05,
        noise_std: float = 0.02,
        brightness_range: float = 0.1,
        contrast_range: float = 0.1,
    ):
        self.num_slices = num_slices  # C
        self.aug_prob = aug_prob
        self.max_angle = max_angle
        self.scale_range = scale_range
        self.translate_frac = translate_frac
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    @classmethod
    def from_config(cls, data_cfg) -> "GPUAugmentor":
        """Create augmentor from DataConfig."""
        return cls(
            num_slices=data_cfg.num_slices,
            aug_prob=data_cfg.aug_prob,
            max_angle=data_cfg.max_angle,
            scale_range=data_cfg.scale_range,
            translate_frac=data_cfg.translate_frac,
            noise_std=data_cfg.noise_std,
            brightness_range=data_cfg.brightness_range,
            contrast_range=data_cfg.contrast_range,
        )

    def __call__(
        self,
        ncct: torch.Tensor,
        cta: torch.Tensor,
        mask: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply augmentation and extract middle C slices.

        Args:
            ncct: (N, 3C, H, W) normalized input on GPU
            cta:  (N, 3C, H, W) normalized target on GPU
            mask: (N, 3C, H, W) binary mask on GPU
            training: if False, skip augmentation (only extract middle slices)

        Returns:
            (ncct, cta, mask) each (N, C, H, W)
        """
        if training:
            N = ncct.shape[0]
            # Per-sample augmentation mask
            do_aug = torch.rand(N, device=ncct.device) < self.aug_prob

            if do_aug.any():
                ncct, cta, mask = self._spatial_augment_3d(
                    ncct, cta, mask, do_aug
                )
                ncct = self._pixel_augment(ncct, do_aug)

        # Extract middle C slices from 3C
        C = self.num_slices
        start = C  # middle block starts at index C
        ncct = ncct[:, start:start + C]
        cta = cta[:, start:start + C]
        mask = mask[:, start:start + C]
        return ncct, cta, mask

    def _build_affine_3d(
        self, N: int, do_aug: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build batched 3D affine matrices (N, 3, 4).

        Rotation is around the D (depth/axial) axis only.
        Scaling is isotropic in H-W, identity in D.
        Translation in H-W, none in D.
        Flip is horizontal (W axis).

        For non-augmented samples, returns identity matrix.
        """
        # Start with identity
        theta = torch.zeros(N, 3, 4, device=device, dtype=dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 2, 2] = 1.0

        n_aug = do_aug.sum().item()
        if n_aug == 0:
            return theta

        # Sample random parameters for augmented samples
        angles = torch.empty(n_aug, device=device).uniform_(
            -self.max_angle, self.max_angle
        ) * (math.pi / 180.0)
        scales = torch.empty(n_aug, device=device).uniform_(
            1.0 - self.scale_range, 1.0 + self.scale_range
        )
        tx = torch.empty(n_aug, device=device).uniform_(
            -self.translate_frac, self.translate_frac
        )
        ty = torch.empty(n_aug, device=device).uniform_(
            -self.translate_frac, self.translate_frac
        )
        # Horizontal flip: -1 or 1
        flip_w = (torch.rand(n_aug, device=device) > 0.5).float() * 2.0 - 1.0

        cos_a = torch.cos(angles) * scales
        sin_a = torch.sin(angles) * scales

        # affine_grid maps output coords → input coords
        # For 5D input (N,C,D,H,W), grid last dim is (x=W, y=H, z=D)
        # Row 0 → W (x), Row 1 → H (y), Row 2 → D (z)
        aug_theta = torch.zeros(n_aug, 3, 4, device=device, dtype=dtype)
        aug_theta[:, 0, 0] = cos_a * flip_w   # W←W (with flip)
        aug_theta[:, 0, 1] = -sin_a * flip_w  # W←H (with flip)
        aug_theta[:, 0, 3] = tx * flip_w       # W translation (with flip)
        aug_theta[:, 1, 0] = sin_a             # H←W
        aug_theta[:, 1, 1] = cos_a             # H←H
        aug_theta[:, 1, 3] = ty                # H translation
        aug_theta[:, 2, 2] = 1.0               # D←D (identity)

        theta[do_aug] = aug_theta
        return theta

    def _spatial_augment_3d(
        self,
        ncct: torch.Tensor,
        cta: torch.Tensor,
        mask: torch.Tensor,
        do_aug: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply 3D spatial augmentation treating (N, 3C, H, W) as (N, 1, D, H, W).

        Uses 3D affine_grid + grid_sample for physically correct 3D transforms.
        """
        N, D, H, W = ncct.shape
        device, dtype = ncct.device, ncct.dtype

        # Build batched 3D affine matrices
        theta = self._build_affine_3d(N, do_aug, device, dtype)

        # Reshape to 5D: (N, 1, D, H, W)
        size_5d = torch.Size([N, 1, D, H, W])
        grid = F.affine_grid(theta, size_5d, align_corners=False)

        # Apply same grid to all three inputs
        ncct = F.grid_sample(
            ncct.unsqueeze(1), grid, mode="bilinear",
            padding_mode="reflection", align_corners=False
        ).squeeze(1)
        cta = F.grid_sample(
            cta.unsqueeze(1), grid, mode="bilinear",
            padding_mode="reflection", align_corners=False
        ).squeeze(1)
        mask = F.grid_sample(
            mask.unsqueeze(1), grid, mode="nearest",
            padding_mode="zeros", align_corners=False
        ).squeeze(1)

        return ncct, cta, mask

    def _pixel_augment(
        self, ncct: torch.Tensor, do_aug: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply per-sample pixel augmentation to ncct only (on GPU).

        Brightness, contrast, and Gaussian noise in normalized [-1, 1] space.
        """
        N = ncct.shape[0]
        n_aug = do_aug.sum().item()
        if n_aug == 0:
            return ncct

        device, dtype = ncct.device, ncct.dtype

        # Sample parameters for augmented samples
        brightness = torch.empty(n_aug, device=device, dtype=dtype).uniform_(
            -self.brightness_range, self.brightness_range
        )
        contrast = torch.empty(n_aug, device=device, dtype=dtype).uniform_(
            1.0 - self.contrast_range, 1.0 + self.contrast_range
        )
        apply_noise = torch.rand(n_aug, device=device) < 0.5

        # Extract augmented samples
        aug_data = ncct[do_aug]  # (n_aug, D, H, W)

        # Contrast: scale around per-sample mean
        # mean shape: (n_aug, 1, 1, 1)
        mean_val = aug_data.mean(dim=(1, 2, 3), keepdim=True)
        aug_data = (aug_data - mean_val) * contrast.view(-1, 1, 1, 1) + mean_val

        # Brightness: additive shift
        aug_data = aug_data + brightness.view(-1, 1, 1, 1)

        # Gaussian noise (only for selected samples)
        if apply_noise.any() and self.noise_std > 0:
            noise = torch.randn_like(aug_data) * self.noise_std
            noise_mask = apply_noise.view(-1, 1, 1, 1).float()
            aug_data = aug_data + noise * noise_mask

        # Clamp and write back
        ncct = ncct.clone()
        ncct[do_aug] = aug_data.clamp(-1.0, 1.0)
        return ncct


def extract_middle_slices(
    tensor: torch.Tensor, num_slices: int
) -> torch.Tensor:
    """
    Extract middle C slices from a (N, 3C, H, W) tensor.

    Args:
        tensor:     (N, 3C, H, W)
        num_slices: C

    Returns:
        (N, C, H, W) middle slices
    """
    start = num_slices
    return tensor[:, start:start + num_slices]
