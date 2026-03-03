"""
CTA degradation module for G dual-task training.

Degrades CTA to approximate the distribution of ``ncct * G(ncct).abs()``,
enabling G to learn both NCCT→CTA translation and degraded_CTA→CTA restoration.

Two-pass inference pipeline (enabled by this training):
  1. g_pred = G(ncct)                          # coarse CTA
  2. intermediate = ncct * g_pred.abs()         # pseudo-CTA (vessels highlighted)
  3. refined = G(intermediate)                  # sharper CTA

During training, ``intermediate`` is approximated by:
  degraded = ncct * |cta|^alpha  (since G aims to produce cta, |g_pred| ≈ |cta|)
  + optional Gaussian blur  (simulates G's smooth output)
  + optional Gaussian noise (for augmentation variety)

Distribution analysis of ``ncct * |g_pred|``:
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

from data.aug_utils import gaussian_blur_auto


class CTADegradation:
    """Degrade CTA to approximate ``ncct * G(ncct).abs()`` distribution.

    The core transform is ``ncct * |cta|^alpha``, which captures the non-linear
    amplitude modulation effect. Optional blur and noise are added for variety.

    Args:
        alpha_range: range for the exponent on |cta|.
            alpha=1.0 gives exact ``ncct * |cta|`` (best match for inference).
            Range provides random variation across training steps.
        blur_prob: probability of applying Gaussian blur.
        blur_kernels: list of kernel sizes to sample from.
        blur_sigma: (min, max) sigma range for Gaussian blur.
        noise_prob: probability of adding Gaussian noise.
        noise_std: (min, max) noise standard deviation range.
    """

    def __init__(
        self,
        alpha_range=(0.7, 1.3),
        blur_prob: float = 0.7,
        blur_kernels=(3, 5),
        blur_sigma=(0.5, 1.5),
        noise_prob: float = 0.5,
        noise_std=(0.01, 0.04),
    ):
        self.alpha_range = alpha_range
        self.blur_prob = blur_prob
        self.blur_kernels = list(blur_kernels)
        self.blur_sigma = blur_sigma
        self.noise_prob = noise_prob
        self.noise_std = noise_std

    def __call__(
        self, ncct: torch.Tensor, cta: torch.Tensor
    ) -> torch.Tensor:
        """Create degraded input approximating ``ncct * G(ncct).abs()``.

        Args:
            ncct: (N, C, H, W) NCCT tensor (normalized to [-1, 1])
            cta:  (N, C, H, W) CTA tensor (normalized to [-1, 1])

        Returns:
            (N, C, H, W) degraded tensor clamped to [-1, 1].
        """
        # Core transform: ncct * |cta|^alpha
        alpha = random.uniform(*self.alpha_range)
        degraded = ncct * cta.abs().pow(alpha)

        # Gaussian blur (simulates G's smooth/blurry output)
        if random.random() < self.blur_prob:
            ks = random.choice(self.blur_kernels)
            sigma = random.uniform(*self.blur_sigma)
            degraded = gaussian_blur_auto(degraded, kernel_size=ks, sigma=sigma)

        # Gaussian noise (augmentation variety)
        if random.random() < self.noise_prob:
            std = random.uniform(*self.noise_std)
            degraded = degraded + torch.randn_like(degraded) * std

        return degraded.clamp(-1.0, 1.0)

    def __repr__(self):
        return (f"CTADegradation(alpha_range={self.alpha_range}, "
                f"blur_prob={self.blur_prob}, noise_prob={self.noise_prob})")
