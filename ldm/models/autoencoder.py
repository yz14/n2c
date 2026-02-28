"""
AutoencoderKL for Latent Diffusion Model.

Compresses 2.5D medical images (C×H×W) into a compact latent representation
(z_channels×H/f×W/f) using a KL-regularized variational autoencoder.

Architecture follows the latent-diffusion reference:
  Encoder: conv_in → [ResBlock + AttnBlock]×levels with Downsample → mid → conv_out
  Decoder: conv_in → mid → [ResBlock + AttnBlock]×levels with Upsample → conv_out
  Latent:  quant_conv (2*z → 2*embed) → DiagonalGaussian → post_quant_conv (embed → z)

The downsampling factor f = 2^(num_levels - 1). For ch_mult=(1,2,4,4) and
resolution=256, the latent is 32×32 (f=8).
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

from .blocks import (
    Normalize, nonlinearity, ResnetBlock, AttnBlock, Downsample, Upsample,
)
from .distributions import DiagonalGaussianDistribution

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    VAE Encoder: maps (B, in_ch, H, W) → (B, 2*z_ch, H/f, W/f).

    Structure per resolution level:
      - num_res_blocks × (ResnetBlock + optional AttnBlock)
      - Downsample (except at the last level)
    Followed by a mid block (ResBlock + Attn + ResBlock) and output conv.
    """

    def __init__(
        self,
        ch: int,
        ch_mult: Tuple[int, ...],
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...],
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, ch, 3, stride=1, padding=1)

        # Downsampling path
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout)
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=True)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # Output
        self.norm_out = Normalize(block_in)
        out_ch = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], None)
                if len(self.down[i_level].attn) > i_block:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """
    VAE Decoder: maps (B, z_ch, H/f, W/f) → (B, out_ch, H, W).

    Mirrors the Encoder structure in reverse: conv_in → mid → up path → conv_out.
    """

    def __init__(
        self,
        ch: int,
        ch_mult: Tuple[int, ...],
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...],
        out_channels: int,
        resolution: int,
        z_channels: int,
        dropout: float = 0.0,
        tanh_out: bool = False,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.tanh_out = tanh_out

        # Compute block_in at the lowest resolution
        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        # Input convolution (from latent)
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, stride=1, padding=1)

        # Middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # Upsampling path
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for _ in range(num_res_blocks + 1):
                block.append(
                    ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout)
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=True)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend for consistent ordering

        # Output
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Input
        h = self.conv_in(z)

        # Middle
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, None)
                if len(self.up[i_level].attn) > i_block:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class AutoencoderKL(nn.Module):
    """
    KL-regularized Autoencoder for Latent Diffusion.

    Encodes input images into a continuous latent space via a diagonal Gaussian
    posterior, with a KL penalty toward N(0, I). The latent is further projected
    through quant_conv / post_quant_conv 1×1 convolutions to decouple the
    encoder/decoder channel count from the latent embedding dimension.

    Usage:
        vae = AutoencoderKL(cfg)
        # Encode
        posterior = vae.encode(x)      # DiagonalGaussianDistribution
        z = posterior.sample()          # (B, embed_dim, H/f, W/f)
        # Decode
        x_recon = vae.decode(z)        # (B, out_ch, H, W)
        # Full forward
        x_recon, posterior = vae(x)

    Args:
        cfg: VAEConfig dataclass with architecture hyperparameters.
    """

    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(
            ch=cfg.ch,
            ch_mult=tuple(cfg.ch_mult),
            num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=tuple(cfg.attn_resolutions),
            in_channels=cfg.in_channels,
            resolution=cfg.resolution,
            z_channels=cfg.z_channels,
            double_z=cfg.double_z,
            dropout=cfg.dropout,
        )
        self.decoder = Decoder(
            ch=cfg.ch,
            ch_mult=tuple(cfg.ch_mult),
            num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=tuple(cfg.attn_resolutions),
            out_channels=cfg.out_channels,
            resolution=cfg.resolution,
            z_channels=cfg.z_channels,
            dropout=cfg.dropout,
        )
        assert cfg.double_z, "AutoencoderKL requires double_z=True"
        self.quant_conv = nn.Conv2d(2 * cfg.z_channels, 2 * cfg.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(cfg.embed_dim, cfg.z_channels, 1)
        self.embed_dim = cfg.embed_dim

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"AutoencoderKL: {n_params / 1e6:.2f}M parameters")

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """Encode input to latent posterior distribution."""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent sample to image space."""
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, sample_posterior: bool = True,
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """
        Full forward pass: encode → sample/mode → decode.

        Returns:
            (reconstruction, posterior)
        """
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_last_layer(self) -> torch.Tensor:
        """Return decoder's final conv weight (for adaptive loss weighting)."""
        return self.decoder.conv_out.weight
