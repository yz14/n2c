"""
Lightweight residual refinement network (G2) for NCCT→CTA pipeline.

Takes an intermediate CTA estimate (e.g. ncct * |G(ncct)|) and applies
small residual corrections to match the true CTA values.

Architecture:
    conv_in → [ResBlock × N] → norm → act → conv_out → + input (global residual)

Design choices:
    - Global residual: output = input + learned_correction (stable start)
    - Zero-initialized final conv: initial output = input (identity mapping)
    - GroupNorm + SiLU: consistent with the main UNet style
    - No downsampling/upsampling: operates at full resolution (corrections are local)
"""

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """Pre-activation residual block: norm → act → conv → norm → act → conv."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, dim), dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, dim), dim)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        return x + h


class RefineNet(nn.Module):
    """
    Lightweight residual refinement network (G2).

    Args:
        in_channels: number of input/output channels (= num_slices)
        hidden_dim:  hidden channel dimension in ResBlocks (default 64)
        num_blocks:  number of residual blocks (default 6)
        enabled:     ON/OFF switch (stored for serialization, not used in forward)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_blocks: int = 6,
        enabled: bool = True,
    ):
        super().__init__()
        self.enabled = enabled
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.conv_in = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.blocks = nn.ModuleList(
            [_ResBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.norm_out = nn.GroupNorm(min(32, hidden_dim), hidden_dim)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(hidden_dim, in_channels, 3, padding=1)

        # Zero-initialize final conv so initial output = input (identity)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refine the intermediate CTA estimate.

        Args:
            x: (N, C, H, W) intermediate CTA estimate

        Returns:
            (N, C, H, W) refined CTA estimate, clamped to [-1, 1]
        """
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        # Global residual + clamp to valid range
        return torch.clamp(x + h, -1.0, 1.0)
