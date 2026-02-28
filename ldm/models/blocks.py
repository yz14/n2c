"""
Core building blocks for LDM models.

Provides reusable components for both the VAE (Encoder/Decoder) and the
diffusion UNet:
  - GroupNorm32: GroupNorm computed in float32 for stability
  - Normalize: standard GroupNorm(32, channels)
  - nonlinearity: SiLU (swish) activation
  - ResnetBlock: residual block with optional timestep embedding projection
  - AttnBlock: single-head self-attention (for VAE bottleneck)
  - MultiHeadAttnBlock: multi-head self-attention (for diffusion UNet)
  - Downsample: stride-2 convolution with asymmetric padding
  - Upsample: nearest-neighbor interpolation + convolution
  - timestep_embedding: sinusoidal positional embedding for diffusion timesteps
  - zero_module: zero-initialize all parameters of a module
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class GroupNorm32(nn.GroupNorm):
    """GroupNorm that always computes in float32 for numerical stability."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def Normalize(in_channels: int, num_groups: int = 32) -> nn.Module:
    """Standard GroupNorm normalization layer."""
    return GroupNorm32(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """SiLU / Swish activation: x * sigmoid(x)."""
    return F.silu(x)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out all parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: (N,) tensor of integer timestep indices.
        dim: embedding dimension.
        max_period: controls the minimum frequency.

    Returns:
        (N, dim) tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


# ---------------------------------------------------------------------------
# Downsample / Upsample
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """
    Spatial downsampling by factor 2 using a stride-2 convolution.

    Uses asymmetric padding (pad right and bottom by 1) to match the
    reference latent-diffusion implementation.
    """

    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            # Asymmetric padding: (left=0, right=1, top=0, bottom=1)
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    """
    Spatial upsampling by factor 2 using nearest interpolation + convolution.
    """

    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResnetBlock(nn.Module):
    """
    Residual block with optional timestep embedding injection.

    Architecture:
        norm1 → SiLU → conv1 → (+temb_proj if temb) → norm2 → SiLU → dropout → conv2
        + skip connection (1×1 conv if channel mismatch)

    When temb_channels=0, no timestep projection is created (used in VAE).
    When use_scale_shift_norm=True, the timestep embedding produces
    (scale, shift) for FiLM conditioning instead of a simple additive bias.

    Args:
        in_channels: input feature channels.
        out_channels: output feature channels (defaults to in_channels).
        temb_channels: timestep embedding channels (0 = no timestep input).
        dropout: dropout rate.
        use_scale_shift_norm: if True, use FiLM (scale+shift) conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        temb_channels: int = 0,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

        if temb_channels > 0:
            if use_scale_shift_norm:
                self.temb_proj = nn.Linear(temb_channels, 2 * out_channels)
            else:
                self.temb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.temb_proj = None

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if self.temb_proj is not None and temb is not None:
            temb_out = nonlinearity(temb)
            temb_out = self.temb_proj(temb_out)[:, :, None, None]

            if self.use_scale_shift_norm:
                scale, shift = torch.chunk(temb_out, 2, dim=1)
                h = self.norm2(h) * (1 + scale) + shift
                h = nonlinearity(h)
            else:
                h = h + temb_out
                h = self.norm2(h)
                h = nonlinearity(h)
        else:
            h = self.norm2(h)
            h = nonlinearity(h)

        h = self.dropout(h)
        h = self.conv2(h)

        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)

        return x + h


# ---------------------------------------------------------------------------
# Attention Blocks
# ---------------------------------------------------------------------------

class AttnBlock(nn.Module):
    """
    Single-head self-attention block (used in VAE encoder/decoder bottleneck).

    Projects input to Q, K, V via 1×1 convolutions, computes scaled
    dot-product attention, and applies a residual connection.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        B, C, H, W = q.shape

        # Reshape to (B, HW, C) for attention
        q = q.reshape(B, C, H * W).permute(0, 2, 1)    # (B, HW, C)
        k = k.reshape(B, C, H * W)                       # (B, C, HW)

        # Scaled dot-product attention
        attn = torch.bmm(q, k) * (C ** -0.5)             # (B, HW, HW)
        attn = F.softmax(attn, dim=2)

        # Apply attention to values
        v = v.reshape(B, C, H * W)                        # (B, C, HW)
        attn = attn.permute(0, 2, 1)                      # (B, HW, HW)
        h_ = torch.bmm(v, attn)                           # (B, C, HW)
        h_ = h_.reshape(B, C, H, W)

        h_ = self.proj_out(h_)
        return x + h_


class MultiHeadAttnBlock(nn.Module):
    """
    Multi-head self-attention block (used in diffusion UNet).

    Uses F.scaled_dot_product_attention for efficiency when available
    (PyTorch 2.0+), falls back to manual computation otherwise.

    Args:
        channels: number of input/output channels.
        num_heads: number of attention heads.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = Normalize(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)                                     # (B, 3C, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)                     # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, heads, HW, HW)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                            # (B, heads, HW, head_dim)

        out = out.permute(0, 2, 1, 3).reshape(B, C, H, W)     # (B, C, H, W)
        out = self.proj_out(out)
        return x + out
