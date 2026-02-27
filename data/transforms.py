"""
Data augmentation transforms for 2.5D medical image slices.

Two categories:
  - Spatial transforms: applied identically to all inputs (ncct, cta, mask)
  - Pixel transforms: applied only to the input image (ncct)

All transforms operate on tensors of shape (C, H, W).
"""

import random
import math
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any


def sample_spatial_params(
    max_angle: float = 15.0,
    scale_range: float = 0.1,
    translate_frac: float = 0.05,
) -> Dict[str, Any]:
    """
    Sample random spatial augmentation parameters.

    Returns:
        dict with keys: flip_h, angle, scale, translate_x, translate_y
    """
    return {
        "flip_h": random.random() < 0.5,
        "angle": random.uniform(-max_angle, max_angle),
        "scale": random.uniform(1.0 - scale_range, 1.0 + scale_range),
        "translate_x": random.uniform(-translate_frac, translate_frac),
        "translate_y": random.uniform(-translate_frac, translate_frac),
    }


def apply_spatial_transform(
    tensor: torch.Tensor,
    params: Dict[str, Any],
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Apply spatial augmentation to a (C, H, W) tensor using pre-sampled params.

    Args:
        tensor:  (C, H, W) tensor
        params:  dict from sample_spatial_params()
        mode:    interpolation mode ("bilinear" for images, "nearest" for masks)

    Returns:
        Transformed (C, H, W) tensor
    """
    if params["flip_h"]:
        tensor = tensor.flip(-1)  # horizontal flip

    angle = params["angle"]
    scale = params["scale"]
    tx = params["translate_x"]
    ty = params["translate_y"]

    # Build 2x3 affine matrix
    theta_rad = math.radians(angle)
    cos_a = math.cos(theta_rad) * scale
    sin_a = math.sin(theta_rad) * scale

    # Affine matrix: rotation + scale + translation
    theta = torch.tensor([
        [cos_a, -sin_a, tx],
        [sin_a,  cos_a, ty],
    ], dtype=tensor.dtype, device=tensor.device)

    # grid_sample expects (N, C, H, W), theta (N, 2, 3)
    x = tensor.unsqueeze(0)  # (1, C, H, W)
    theta = theta.unsqueeze(0)  # (1, 2, 3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    padding_mode = "zeros" if mode == "nearest" else "reflection"
    x = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode,
                       align_corners=False)
    return x.squeeze(0)  # (C, H, W)


def apply_spatial_augmentation(
    ncct: torch.Tensor,
    cta: torch.Tensor,
    mask: torch.Tensor,
    params: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply identical spatial transform to all inputs.

    Args:
        ncct:   (C, H, W) float tensor
        cta:    (C, H, W) float tensor
        mask:   (C, H, W) float tensor (binary mask as float)
        params: augmentation parameters

    Returns:
        (ncct, cta, mask) after spatial augmentation
    """
    ncct = apply_spatial_transform(ncct, params, mode="bilinear")
    cta = apply_spatial_transform(cta, params, mode="bilinear")
    mask = apply_spatial_transform(mask, params, mode="nearest")
    return ncct, cta, mask


def sample_pixel_params(
    brightness_range: float = 0.1,
    contrast_range: float = 0.1,
    noise_std: float = 0.02,
) -> Dict[str, Any]:
    """
    Sample random pixel-space augmentation parameters.

    Returns:
        dict with keys: brightness, contrast, noise_std, apply_noise
    """
    return {
        "brightness": random.uniform(-brightness_range, brightness_range),
        "contrast": random.uniform(1.0 - contrast_range, 1.0 + contrast_range),
        "noise_std": noise_std,
        "apply_noise": random.random() < 0.5,
    }


def apply_pixel_augmentation(
    tensor: torch.Tensor,
    params: Dict[str, Any],
) -> torch.Tensor:
    """
    Apply pixel-space augmentation to input image only.
    Operates in normalized [-1, 1] space.

    Args:
        tensor:  (C, H, W) normalized float tensor
        params:  dict from sample_pixel_params()

    Returns:
        Augmented (C, H, W) tensor
    """
    # Contrast adjustment (multiply around mean)
    mean_val = tensor.mean()
    tensor = (tensor - mean_val) * params["contrast"] + mean_val

    # Brightness adjustment (additive shift)
    tensor = tensor + params["brightness"]

    # Gaussian noise
    if params["apply_noise"] and params["noise_std"] > 0:
        noise = torch.randn_like(tensor) * params["noise_std"]
        tensor = tensor + noise

    # Clamp to valid range
    tensor = tensor.clamp(-1.0, 1.0)
    return tensor
