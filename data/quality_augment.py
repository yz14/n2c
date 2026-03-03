"""
Quality degradation pipeline for GAN discriminator negative samples.

Provides random image degradations (blur, noise, downsample-upsample) that
can be applied to real CTA images to create additional negative samples for D.

This teaches D to assess image quality — sharpness, noise level, resolution —
rather than only distinguishing "real vs. generated". The quality-aware D
gradients push G to produce sharper, cleaner images.

Usage in D training step:
    degraded_cta = quality_degrade(cta)   # create degraded negative
    d_loss_degraded = D_loss(D(degraded_cta), fake_label)
    d_loss += quality_weight * d_loss_degraded

Also supports mild degradation of G's input (NCCT) to improve G's robustness.
"""

import random
import torch
import torch.nn.functional as F

from data.aug_utils import gaussian_blur_auto, downsample_upsample_auto


def _gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """Add Gaussian noise to (N, C, H, W) tensor."""
    return x + torch.randn_like(x) * std


def _downsample_upsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Downsample then upsample to create resolution artifacts.

    Auto-selects 2D or 3D interpolation based on channel count.
    """
    return downsample_upsample_auto(x, scale)


class QualityDegradation:
    """Random quality degradation pipeline.

    Randomly applies 1-2 degradation types from the configured policy set.
    Each call applies a different random combination, ensuring diverse negatives.

    Args:
        policies: comma-separated string of degradation types.
            Options: "blur", "noise", "downsample"
        severity: controls the degradation strength.
            "mild" — suitable for G input augmentation (subtle degradations)
            "strong" — suitable for D negative samples (clearly degraded)
    """

    # Severity-dependent parameter ranges
    _PARAMS = {
        "mild": {
            "blur_kernels": [3, 5],
            "blur_sigma": (0.3, 1.0),
            "noise_std": (0.01, 0.04),
            "downsample_scales": [2],
        },
        "strong": {
            "blur_kernels": [3, 5, 7, 9],
            "blur_sigma": (0.5, 2.5),
            "noise_std": (0.03, 0.12),
            "downsample_scales": [2, 3, 4],
        },
    }

    def __init__(self, policies: str = "blur,noise,downsample",
                 severity: str = "strong"):
        self.policy_list = [p.strip() for p in policies.split(",") if p.strip()]
        if not self.policy_list:
            self.policy_list = ["blur", "noise", "downsample"]
        assert severity in self._PARAMS, f"severity must be 'mild' or 'strong', got {severity}"
        self.params = self._PARAMS[severity]
        self.severity = severity

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random degradation(s) to input tensor (N, C, H, W).

        Returns degraded copy (does not modify input in-place).
        """
        x = x.clone()

        # Randomly select 1-2 degradation types
        n_degrad = random.randint(1, min(2, len(self.policy_list)))
        selected = random.sample(self.policy_list, n_degrad)

        for policy in selected:
            if policy == "blur":
                ks = random.choice(self.params["blur_kernels"])
                sigma = random.uniform(*self.params["blur_sigma"])
                x = gaussian_blur_auto(x, kernel_size=ks, sigma=sigma)

            elif policy == "noise":
                std = random.uniform(*self.params["noise_std"])
                x = _gaussian_noise(x, std=std)

            elif policy == "downsample":
                scale = random.choice(self.params["downsample_scales"])
                x = _downsample_upsample(x, scale=scale)

        # Clamp to valid range [-1, 1] (noise can push values outside)
        x = x.clamp(-1.0, 1.0)
        return x

    def __repr__(self):
        return (f"QualityDegradation(policies={self.policy_list}, "
                f"severity={self.severity})")
