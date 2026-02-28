"""
Conditional Diffusion UNet for Latent Diffusion Model.

Operates in the latent space of the VAE. The condition (NCCT latent) is
concatenated with the noisy target latent (CTA) along the channel dimension,
so in_channels = 2 * z_channels.

Architecture (following openaimodel.py from latent-diffusion):
  - Sinusoidal timestep embedding → MLP → time_embed_dim
  - Encoder: conv_in → [ResBlock(+temb) + optional MultiHeadAttn] per level,
             with Downsample between levels
  - Middle:  ResBlock + MultiHeadAttn + ResBlock
  - Decoder: [ResBlock(+skip+temb) + optional MultiHeadAttn] per level,
             with Upsample between levels
  - Output:  norm → SiLU → conv_out

Skip connections between encoder and decoder (standard UNet pattern).
The UNet predicts the noise ε added to the latent.
"""

import logging
from typing import Tuple, List

import torch
import torch.nn as nn

from .blocks import (
    Normalize,
    nonlinearity,
    zero_module,
    timestep_embedding,
    ResnetBlock,
    MultiHeadAttnBlock,
    Downsample,
    Upsample,
)

logger = logging.getLogger(__name__)


class DiffusionUNet(nn.Module):
    """
    Conditional UNet for latent diffusion.

    The model takes:
      - x: (B, in_channels, H, W) — concatenation of [noisy_z_cta, z_ncct]
      - t: (B,) — integer timestep indices

    And predicts the noise ε of shape (B, out_channels, H, W).

    Args:
        in_channels:  input channels (= 2 * z_channels for concat conditioning).
        out_channels: output channels (= z_channels, the noise dimension).
        model_channels: base channel count.
        channel_mult: per-level channel multipliers.
        num_res_blocks: residual blocks per level.
        attention_resolutions: set of downsample rates at which attention is used.
                               E.g., {2, 4} means attention at 2× and 4× downsampled.
        dropout: dropout rate.
        num_heads: number of attention heads.
        use_scale_shift_norm: use FiLM conditioning for timestep embedding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (4, 2),
        dropout: float = 0.0,
        num_heads: int = 4,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.attention_resolutions = set(attention_resolutions)
        self.num_heads = num_heads

        time_embed_dim = model_channels * 4

        # --- Timestep embedding MLP ---
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # --- Encoder (input blocks) ---
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        )

        input_block_channels = [model_channels]
        ch = model_channels
        ds = 1  # current downsample rate

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_ch = mult * model_channels
                layers: List[nn.Module] = [
                    ResnetBlock(
                        in_channels=ch,
                        out_channels=out_ch,
                        temb_channels=time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = out_ch
                if ds in self.attention_resolutions:
                    layers.append(MultiHeadAttnBlock(ch, num_heads=num_heads))
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_channels.append(ch)

            # Downsample (except at the last level)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    nn.ModuleList([Downsample(ch, with_conv=True)])
                )
                input_block_channels.append(ch)
                ds *= 2

        # --- Middle block ---
        self.middle_block = nn.ModuleList([
            ResnetBlock(ch, ch, temb_channels=time_embed_dim, dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm),
            MultiHeadAttnBlock(ch, num_heads=num_heads),
            ResnetBlock(ch, ch, temb_channels=time_embed_dim, dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm),
        ])

        # --- Decoder (output blocks) ---
        self.output_blocks = nn.ModuleList()

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                skip_ch = input_block_channels.pop()
                out_ch = mult * model_channels
                layers: List[nn.Module] = [
                    ResnetBlock(
                        in_channels=ch + skip_ch,
                        out_channels=out_ch,
                        temb_channels=time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = out_ch
                if ds in self.attention_resolutions:
                    layers.append(MultiHeadAttnBlock(ch, num_heads=num_heads))

                # Upsample at the last block of each level (except level 0)
                if level != 0 and i == num_res_blocks:
                    layers.append(Upsample(ch, with_conv=True))
                    ds //= 2

                self.output_blocks.append(nn.ModuleList(layers))

        # --- Output ---
        self.out_norm = Normalize(ch)
        self.out_conv = zero_module(
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"DiffusionUNet: {n_params / 1e6:.2f}M parameters")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, in_channels, H, W) — concat of [noisy_z, condition_z].
            t: (B,) — integer timestep indices.

        Returns:
            (B, out_channels, H, W) — predicted noise.
        """
        # Timestep embedding
        temb = timestep_embedding(t, self.model_channels)
        temb = self.time_embed(temb)

        # Encoder
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Conv2d):
                # First conv_in
                h = module(h)
            else:
                # ModuleList of [ResBlock, (optional Attn), (optional Downsample)]
                for layer in module:
                    if isinstance(layer, ResnetBlock):
                        h = layer(h, temb)
                    elif isinstance(layer, (MultiHeadAttnBlock, Downsample)):
                        h = layer(h)
                    else:
                        h = layer(h)
            hs.append(h)

        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResnetBlock):
                h = layer(h, temb)
            else:
                h = layer(h)

        # Decoder
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResnetBlock):
                    h = layer(h, temb)
                elif isinstance(layer, (MultiHeadAttnBlock, Upsample)):
                    h = layer(h)
                else:
                    h = layer(h)

        # Output
        h = self.out_norm(h)
        h = nonlinearity(h)
        h = self.out_conv(h)
        return h

    @classmethod
    def from_config(cls, unet_cfg, z_channels: int) -> "DiffusionUNet":
        """
        Build DiffusionUNet from a DiffusionUNetConfig.

        The in_channels is 2 * z_channels (noisy target + condition),
        and out_channels is z_channels (predicted noise).
        """
        return cls(
            in_channels=2 * z_channels,
            out_channels=z_channels,
            model_channels=unet_cfg.model_channels,
            channel_mult=tuple(unet_cfg.channel_mult),
            num_res_blocks=unet_cfg.num_res_blocks,
            attention_resolutions=tuple(unet_cfg.attention_resolutions),
            dropout=unet_cfg.dropout,
            num_heads=unet_cfg.num_heads,
            use_scale_shift_norm=unet_cfg.use_scale_shift_norm,
        )
