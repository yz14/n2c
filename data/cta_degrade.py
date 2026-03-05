"""
CTA degradation module for G dual-task training.

Provides two degradation modes, randomly selected per call:

  **Mode 1 — Intermediate-state simulation** (``ncct * |cta|^alpha``):
    Approximates the distribution of ``ncct * G(ncct).abs()`` from the two-pass
    inference pipeline. Teaches G to refine the intermediate representation.

  **Mode 2 — Direct CTA degradation** (blur + noise + downsample on CTA):
    Applies realistic quality degradation directly to CTA. Teaches G to
    restore degraded CTA-like inputs, improving robustness to noisy/blurry
    first-pass outputs during two-pass inference.

Two-pass inference pipeline (enabled by this training):
  1. g_pred = G(ncct)                          # coarse CTA
  2. intermediate = ncct * g_pred.abs()         # pseudo-CTA (vessels highlighted)
  3. refined = G(intermediate)                  # sharper CTA

Distribution analysis of ``ncct * |g_pred|`` (Mode 1):
  - Air (outside body): ncct ≈ -1, |cta| ≈ 1  → product ≈ -1 (preserved)
  - Soft tissue:        ncct ≈ 0.3, |cta| ≈ 0.3 → product ≈ 0.09 (compressed)
  - Lung air:           ncct ≈ -0.5, |cta| ≈ 0.5 → product ≈ -0.25 (compressed)
  - Enhanced vessels:   ncct ≈ -0.5, |cta| ≈ 0.5 → product ≈ -0.25 (moderate)
  Key effect: non-linear compression where mid-range values are squeezed toward 0,
  extreme values (-1 for air) are preserved, and vessels stand out against dark
  background — giving a CTA-like visual appearance but with wrong pixel magnitudes.

Usage in trainer:
    from data.cta_degrade import CTADegradation

    cta_degrade_fn = CTADegradation()
    # In training loop, with probability p:
    ncct = cta_degrade_fn(ncct, cta)  # replace ncct input with degraded CTA
"""

import random
import torch

from data.aug_utils import gaussian_blur_auto, downsample_upsample_auto


class CTADegradation:
    """Degrade CTA via two randomly-selected modes for G dual-task training.

    Mode 1 (intermediate): ``ncct * |cta|^alpha`` + optional blur/noise.
    Mode 2 (direct):       blur + noise + downsample applied directly to CTA.

    Args:
        direct_prob: probability of using Mode 2 (direct CTA degradation)
            vs Mode 1 (intermediate-state simulation). Default 0.5 = equal chance.
        alpha_range: (Mode 1) range for the exponent on |cta|.
            alpha=1.0 gives exact ``ncct * |cta|`` (best match for inference).
        blur_prob: probability of applying Gaussian blur (both modes).
        blur_kernels: list of kernel sizes to sample from.
        blur_sigma: (min, max) sigma range for Gaussian blur.
        noise_prob: probability of adding Gaussian noise (both modes).
        noise_std: (min, max) noise standard deviation range.
        downsample_prob: (Mode 2 only) probability of downsample-upsample.
        downsample_scales: (Mode 2 only) list of downsample factors.
    """

    def __init__(
        self,
        direct_prob: float = 0.5,
        alpha_range=(0.7, 1.3),
        blur_prob: float = 0.7,
        blur_kernels=(3, 5),
        blur_sigma=(0.5, 1.5),
        noise_prob: float = 0.5,
        noise_std=(0.01, 0.04),
        downsample_prob: float = 0.5,
        downsample_scales=(2, 3, 4),
    ):
        self.direct_prob = direct_prob
        self.alpha_range = alpha_range
        self.blur_prob = blur_prob
        self.blur_kernels = list(blur_kernels)
        self.blur_sigma = blur_sigma
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.downsample_prob = downsample_prob
        self.downsample_scales = list(downsample_scales)

    def __call__(
        self, ncct: torch.Tensor, cta: torch.Tensor
    ) -> torch.Tensor:
        """Create degraded input via randomly selected mode.

        Args:
            ncct: (N, C, H, W) NCCT tensor (normalized to [-1, 1])
            cta:  (N, C, H, W) CTA tensor (normalized to [-1, 1])

        Returns:
            (N, C, H, W) degraded tensor clamped to [-1, 1].
        """
        if random.random() < self.direct_prob:
            return self._degrade_direct(cta)
        return self._degrade_intermediate(ncct, cta)

    def _degrade_intermediate(
        self, ncct: torch.Tensor, cta: torch.Tensor
    ) -> torch.Tensor:
        """Mode 1: ncct * |cta|^alpha — simulates two-pass intermediate state."""
        alpha = random.uniform(*self.alpha_range)
        degraded = ncct * cta.abs().pow(alpha)
        degraded = self._apply_blur_noise(degraded)
        return degraded.clamp(-1.0, 1.0)

    def _degrade_direct(self, cta: torch.Tensor) -> torch.Tensor:
        """Mode 2: directly degrade CTA with blur, noise, downsample."""
        degraded = cta.clone()

        # Gaussian blur (simulates G's smooth/blurry output)
        degraded = self._apply_blur_noise(degraded)

        # Downsample-upsample (simulates resolution loss)
        if random.random() < self.downsample_prob:
            scale = random.choice(self.downsample_scales)
            degraded = downsample_upsample_auto(degraded, scale=scale)

        return degraded.clamp(-1.0, 1.0)

    def _apply_blur_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Shared blur + noise augmentation for both modes."""
        if random.random() < self.blur_prob:
            ks = random.choice(self.blur_kernels)
            sigma = random.uniform(*self.blur_sigma)
            x = gaussian_blur_auto(x, kernel_size=ks, sigma=sigma)

        if random.random() < self.noise_prob:
            std = random.uniform(*self.noise_std)
            x = x + torch.randn_like(x) * std

        return x

    def __repr__(self):
        return (f"CTADegradation(direct_prob={self.direct_prob}, "
                f"alpha_range={self.alpha_range}, "
                f"blur_prob={self.blur_prob}, noise_prob={self.noise_prob})")
