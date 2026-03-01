"""
ResBlock-based Multi-Scale Discriminator with Self-Attention.

Improvements over the standard PatchGAN (discriminator.py):
  - Residual connections for better gradient flow through deep D
  - Learned downsampling (strided convolution, not external AvgPool2d)
  - Optional self-attention at intermediate resolution
  - Spectral normalization on all layers
  - Compatible with the same interface as MultiscaleDiscriminator:
    returns List[List[Tensor]] for feature matching loss

Architecture per sub-discriminator:
  conv_in → [ResBlock(downsample) × N] → [SelfAttn] → conv_out(1)

Reference:
  - Miyato et al., "Spectral Normalization for GANs", ICLR 2018
  - Zhang et al., "Self-Attention GAN", ICML 2019
  - Karras et al., "Analyzing and Improving StyleGAN", CVPR 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _maybe_sn(module: nn.Module, use_sn: bool) -> nn.Module:
    return spectral_norm(module) if use_sn else module


class _ResBlock(nn.Module):
    """
    Residual block for discriminator.

    Architecture: conv(stride) → LeakyReLU → conv → + skip → LeakyReLU
    Skip connection uses 1x1 conv when channels change or downsampling.
    """

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False,
                 use_sn: bool = True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = _maybe_sn(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1), use_sn
        )
        self.conv2 = _maybe_sn(
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1), use_sn
        )
        self.act = nn.LeakyReLU(0.2, inplace=False)

        if in_ch != out_ch or downsample:
            self.skip = _maybe_sn(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride), use_sn
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + self.skip(x))


class SelfAttention(nn.Module):
    """
    Self-attention layer (SAGAN-style).

    Captures long-range spatial dependencies that convolutions miss.
    Uses spectral-normalized 1x1 convolutions for Q, K, V projections.
    """

    def __init__(self, in_ch: int, use_sn: bool = True):
        super().__init__()
        mid_ch = max(in_ch // 8, 1)
        self.query = _maybe_sn(nn.Conv2d(in_ch, mid_ch, 1), use_sn)
        self.key = _maybe_sn(nn.Conv2d(in_ch, mid_ch, 1), use_sn)
        self.value = _maybe_sn(nn.Conv2d(in_ch, in_ch, 1), use_sn)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.query(x).flatten(2)        # (B, mid, H*W)
        k = self.key(x).flatten(2)          # (B, mid, H*W)
        v = self.value(x).flatten(2)        # (B, C, H*W)

        attn = torch.bmm(q.transpose(1, 2), k)  # (B, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.transpose(1, 2))  # (B, C, H*W)
        out = out.view(B, C, H, W)
        return x + self.gamma * out


class ResBlockSubDiscriminator(nn.Module):
    """
    Single-scale ResBlock discriminator.

    Architecture:
      conv_in(input_nc → ndf) → [ResBlock(ndf_i → ndf_{i+1}, downsample)] × n_blocks
      → optional SelfAttention → conv_out(ndf_final → 1)

    Returns list of intermediate features (for feature matching) if get_interm_feat=True.

    Args:
        input_nc:         number of input channels
        ndf:              base number of filters
        n_blocks:         number of ResBlocks (each downsamples 2x)
        use_spectral_norm: apply SN to all conv layers
        use_attention:    add self-attention after middle block
        get_interm_feat:  return intermediate features for feature matching
    """

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_blocks: int = 4,
        use_spectral_norm: bool = True,
        use_attention: bool = True,
        get_interm_feat: bool = True,
    ):
        super().__init__()
        self.get_interm_feat = get_interm_feat
        self.n_blocks = n_blocks
        use_sn = use_spectral_norm

        # Input conv (no downsampling)
        self.conv_in = nn.Sequential(
            _maybe_sn(nn.Conv2d(input_nc, ndf, 3, padding=1), use_sn),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ResBlocks with progressive channel increase
        blocks = []
        ch = ndf
        attn_idx = n_blocks // 2  # insert attention after middle block
        for i in range(n_blocks):
            ch_next = min(ch * 2, 512)
            blocks.append(_ResBlock(ch, ch_next, downsample=True, use_sn=use_sn))
            ch = ch_next
            if use_attention and i == attn_idx:
                blocks.append(SelfAttention(ch, use_sn=use_sn))
        self.blocks = nn.ModuleList(blocks)

        # Final prediction layer (1 channel, spatial output like PatchGAN)
        self.conv_out = _maybe_sn(nn.Conv2d(ch, 1, 3, padding=1), use_sn)

    def forward(self, x: torch.Tensor):
        features = []
        h = self.conv_in(x)
        features.append(h)

        for block in self.blocks:
            h = block(h)
            if isinstance(block, _ResBlock):
                features.append(h)

        out = self.conv_out(h)
        features.append(out)

        if self.get_interm_feat:
            return features
        return out


class MultiscaleResBlockDiscriminator(nn.Module):
    """
    Multi-scale ResBlock discriminator.

    Like MultiscaleDiscriminator but uses ResBlockSubDiscriminator instead
    of NLayerDiscriminator. Each scale still operates at a different
    resolution via input downsampling.

    The inter-scale downsampling uses bilinear interpolation (better than
    AvgPool2d for preserving subtle features).

    Interface is identical to MultiscaleDiscriminator:
      forward(x) → List[List[Tensor]]

    Args:
        input_nc:         total input channels
        ndf:              base filters per sub-discriminator
        n_blocks:         ResBlocks per sub-discriminator
        num_D:            number of discriminator scales
        use_spectral_norm: apply SN to all layers
        use_attention:    self-attention (only on full-res scale)
        get_interm_feat:  return features for feature matching
        enabled:          ON/OFF switch
    """

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_blocks: int = 4,
        num_D: int = 2,
        use_spectral_norm: bool = True,
        use_attention: bool = True,
        get_interm_feat: bool = True,
        enabled: bool = True,
    ):
        super().__init__()
        self.num_D = num_D
        self.get_interm_feat = get_interm_feat
        self.enabled = enabled

        for i in range(num_D):
            netD = ResBlockSubDiscriminator(
                input_nc=input_nc,
                ndf=ndf,
                n_blocks=n_blocks,
                use_spectral_norm=use_spectral_norm,
                use_attention=(use_attention and i == 0),  # attention only on full-res
                get_interm_feat=get_interm_feat,
            )
            setattr(self, f"discriminator_{i}", netD)

    def forward(self, x: torch.Tensor):
        """
        Multi-scale forward pass.

        Returns:
            List of num_D results, each is a list of feature tensors.
        """
        results = []
        input_downsampled = x
        for i in range(self.num_D):
            netD = getattr(self, f"discriminator_{i}")
            out = netD(input_downsampled)
            if not self.get_interm_feat:
                out = [out]
            results.append(out)
            if i < self.num_D - 1:
                # Bilinear interpolation — better than AvgPool2d for subtle features
                input_downsampled = F.interpolate(
                    input_downsampled, scale_factor=0.5, mode="bilinear",
                    align_corners=False,
                )
        return results
