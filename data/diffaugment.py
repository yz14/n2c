"""
Differentiable Augmentation for Data-Efficient GAN Training.

Applies lightweight, differentiable augmentations to both real and fake images
before feeding them to the discriminator. This prevents D from overfitting to
the training set and improves GAN stability with limited data.

All augmentations are differentiable (gradients flow through them), so they
can be applied inside the training loop without breaking backpropagation.

Supported policies:
  - "color":       random brightness, saturation, contrast shifts
  - "translation": random spatial translation with zero-padding
  - "cutout":      random rectangular cutout (erased to zero)

Usage:
    augment = DiffAugment(policy="color,translation,cutout")
    augmented_img = augment(img)  # (N, C, H, W) tensor

Reference:
  Zhao et al., "Differentiable Augmentation for Data-Efficient GAN Training",
  NeurIPS 2020.
"""

import torch
import torch.nn.functional as F


class DiffAugment:
    """
    Differentiable augmentation module.

    Args:
        policy: comma-separated list of augmentation types.
            Options: "color", "translation", "cutout".
            Empty string disables augmentation.
    """

    VALID_POLICIES = {"color", "translation", "cutout"}

    def __init__(self, policy: str = "color,translation,cutout"):
        self.policies = []
        if policy:
            for p in policy.split(","):
                p = p.strip()
                if p and p in self.VALID_POLICIES:
                    self.policies.append(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations sequentially.

        Args:
            x: (N, C, H, W) tensor

        Returns:
            Augmented (N, C, H, W) tensor (same shape, differentiable).
        """
        for p in self.policies:
            if p == "color":
                x = _aug_color(x)
            elif p == "translation":
                x = _aug_translation(x)
            elif p == "cutout":
                x = _aug_cutout(x)
        return x

    def __repr__(self):
        return f"DiffAugment(policy={','.join(self.policies)})"


def _aug_color(x: torch.Tensor) -> torch.Tensor:
    """Random color jittering: brightness, saturation, contrast.

    Each transform is applied with 50% probability per batch.
    The same random shift is applied to all samples in the batch.
    """
    # Random brightness shift
    if torch.rand(1).item() < 0.5:
        shift = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 0.2
        x = x + shift

    # Random saturation (only meaningful for multi-channel)
    if x.shape[1] > 1 and torch.rand(1).item() < 0.5:
        mean = x.mean(dim=1, keepdim=True)
        factor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2.0
        x = mean + (x - mean) * factor

    # Random contrast
    if torch.rand(1).item() < 0.5:
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        factor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 0.5 + 0.5
        x = mean + (x - mean) * factor

    return x


def _aug_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Random spatial translation with zero-padding.

    Shifts the image by up to `ratio` fraction of its size in each direction.
    """
    N, C, H, W = x.shape
    shift_h = int(H * ratio + 0.5)
    shift_w = int(W * ratio + 0.5)

    # Random translation offsets per sample
    th = torch.randint(-shift_h, shift_h + 1, (N, 1, 1), device=x.device)
    tw = torch.randint(-shift_w, shift_w + 1, (N, 1, 1), device=x.device)

    # Create sampling grid
    grid_h = torch.arange(H, device=x.device).float()
    grid_w = torch.arange(W, device=x.device).float()
    grid_h = (grid_h[None, :, None] + th.float()) / (H / 2) - 1  # (N, H, 1)
    grid_w = (grid_w[None, None, :] + tw.float()) / (W / 2) - 1  # (N, 1, W)

    grid = torch.cat([
        grid_w.expand(N, H, W).unsqueeze(-1),
        grid_h.expand(N, H, W).unsqueeze(-1),
    ], dim=-1)  # (N, H, W, 2)

    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)


def _aug_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Random rectangular cutout.

    Erases a random rectangular region (up to `ratio` of image size) to zero.
    The same cutout region is used for all channels but differs per sample.
    """
    N, C, H, W = x.shape
    cut_h = int(H * ratio + 0.5)
    cut_w = int(W * ratio + 0.5)

    # Random cutout size per sample
    h_size = torch.randint(1, cut_h + 1, (N,), device=x.device)
    w_size = torch.randint(1, cut_w + 1, (N,), device=x.device)

    # Random cutout position
    h_start = torch.randint(0, H, (N,), device=x.device)
    w_start = torch.randint(0, W, (N,), device=x.device)

    # Build mask (1 = keep, 0 = cutout)
    mask = torch.ones_like(x)
    for i in range(N):
        h1 = h_start[i].item()
        w1 = w_start[i].item()
        h2 = min(h1 + h_size[i].item(), H)
        w2 = min(w1 + w_size[i].item(), W)
        mask[i, :, h1:h2, w1:w2] = 0.0

    return x * mask
