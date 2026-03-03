"""
Shared utility functions for GPU-based data augmentation.

These are used across multiple augmentation modules (transforms, quality_augment,
cta_degrade) to avoid code duplication.

For 2.5D data (C > 1 adjacent slices), use ``gaussian_blur_auto`` which
automatically selects 2D or 3D blurring based on channel count.
"""

import torch
import torch.nn.functional as F

# Channel count threshold: C >= this uses 3D blur (treats C as depth)
_3D_CHANNEL_THRESHOLD = 4


def gaussian_blur_2d(
    x: torch.Tensor, kernel_size: int, sigma: float
) -> torch.Tensor:
    """Apply 2D Gaussian blur to (N, C, H, W) using separable depthwise conv.

    Each channel is blurred independently with the same Gaussian kernel.
    Uses reflection padding to avoid border artifacts.

    Args:
        x: input tensor (N, C, H, W)
        kernel_size: size of the Gaussian kernel (odd integer)
        sigma: standard deviation of the Gaussian

    Returns:
        Blurred tensor with same shape as input.
    """
    C = x.shape[1]
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
    coords -= kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g /= g.sum()

    # Separable kernels for depthwise conv
    kernel_h = g.view(1, 1, -1, 1).expand(C, -1, -1, -1)  # (C, 1, K, 1)
    kernel_w = g.view(1, 1, 1, -1).expand(C, -1, -1, -1)  # (C, 1, 1, K)

    pad = kernel_size // 2
    # Horizontal blur
    x = F.pad(x, [pad, pad, 0, 0], mode="reflect")
    x = F.conv2d(x, kernel_w, groups=C)
    # Vertical blur
    x = F.pad(x, [0, 0, pad, pad], mode="reflect")
    x = F.conv2d(x, kernel_h, groups=C)
    return x


def gaussian_blur_3d(
    x: torch.Tensor, kernel_size: int, sigma: float
) -> torch.Tensor:
    """Apply 3D Gaussian blur to (N, C, H, W) treating C as depth.

    Reshapes to (N, 1, D=C, H, W), applies separable 3D Gaussian, reshapes back.
    The depth kernel size is clamped to min(kernel_size, D) and forced odd.
    Uses replicate padding in all dimensions.

    Args:
        x: input tensor (N, C, H, W) where C represents adjacent slices
        kernel_size: spatial kernel size (H, W dimensions)
        sigma: standard deviation of the Gaussian

    Returns:
        Blurred tensor with same shape as input.
    """
    N, C, H, W = x.shape

    # Depth kernel: clamp to C, ensure odd
    ks_d = min(kernel_size, C)
    if ks_d % 2 == 0:
        ks_d = max(ks_d - 1, 1)
    ks_hw = kernel_size

    # Build 1D Gaussian kernels
    def _gauss_1d(size, s):
        coords = torch.arange(size, dtype=torch.float32, device=x.device) - size // 2
        g = torch.exp(-0.5 * (coords / s) ** 2)
        return g / g.sum()

    g_hw = _gauss_1d(ks_hw, sigma)
    g_d = _gauss_1d(ks_d, sigma)

    # Reshape to 5D: (N, 1, D, H, W)
    x5 = x.unsqueeze(1)

    pad_hw = ks_hw // 2
    pad_d = ks_d // 2

    # Separable 3D convolution: D → H → W
    # Depth pass
    k_d = g_d.view(1, 1, -1, 1, 1)
    x5 = F.pad(x5, [0, 0, 0, 0, pad_d, pad_d], mode="replicate")
    x5 = F.conv3d(x5, k_d)

    # Height pass
    k_h = g_hw.view(1, 1, 1, -1, 1)
    x5 = F.pad(x5, [0, 0, pad_hw, pad_hw, 0, 0], mode="replicate")
    x5 = F.conv3d(x5, k_h)

    # Width pass
    k_w = g_hw.view(1, 1, 1, 1, -1)
    x5 = F.pad(x5, [pad_hw, pad_hw, 0, 0, 0, 0], mode="replicate")
    x5 = F.conv3d(x5, k_w)

    # Back to 4D: (N, C, H, W)
    return x5.squeeze(1)


def gaussian_blur_auto(
    x: torch.Tensor, kernel_size: int, sigma: float
) -> torch.Tensor:
    """Auto-select 2D or 3D Gaussian blur based on channel count.

    For C >= _3D_CHANNEL_THRESHOLD (default 4), treats channels as depth
    and applies 3D Gaussian blur for more realistic degradation of 2.5D data.
    For C < threshold, applies standard 2D per-channel blur.

    Args:
        x: input tensor (N, C, H, W)
        kernel_size: Gaussian kernel size (odd integer)
        sigma: Gaussian standard deviation

    Returns:
        Blurred tensor with same shape as input.
    """
    if x.shape[1] >= _3D_CHANNEL_THRESHOLD:
        return gaussian_blur_3d(x, kernel_size, sigma)
    return gaussian_blur_2d(x, kernel_size, sigma)


def downsample_upsample_auto(
    x: torch.Tensor, scale: int
) -> torch.Tensor:
    """Downsample then upsample to create resolution artifacts.

    For C >= _3D_CHANNEL_THRESHOLD, uses 3D trilinear interpolation
    (treats C as depth) for realistic resolution degradation.
    Otherwise uses 2D bilinear per channel.

    Args:
        x: input tensor (N, C, H, W)
        scale: downsample factor

    Returns:
        Degraded tensor with same shape as input.
    """
    C, H, W = x.shape[1], x.shape[2], x.shape[3]

    if C >= _3D_CHANNEL_THRESHOLD:
        # 3D: (N, 1, D=C, H, W)
        x5 = x.unsqueeze(1)
        D_small = max(1, C // max(1, scale // 2))  # milder depth downscale
        H_small = max(1, H // scale)
        W_small = max(1, W // scale)
        small = F.interpolate(x5, size=(D_small, H_small, W_small),
                              mode="trilinear", align_corners=False)
        big = F.interpolate(small, size=(C, H, W),
                            mode="trilinear", align_corners=False)
        return big.squeeze(1)
    else:
        # 2D per-channel
        small = F.interpolate(x, size=(max(1, H // scale), max(1, W // scale)),
                              mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
