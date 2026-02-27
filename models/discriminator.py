"""
Multi-Scale PatchGAN Discriminator with Spectral Normalization.

Adapted from pix2pixHD (Wang et al., 2018) with modern improvements:
  - Spectral Normalization (SN) on all conv layers for training stability
  - InstanceNorm instead of BatchNorm (better for image synthesis)
  - Intermediate feature extraction for feature matching loss
  - ON/OFF switch for staged training

Architecture:
  MultiscaleDiscriminator contains num_D NLayerDiscriminators operating at
  different image scales (original, 2x downsampled, 4x downsampled, ...).
  Each sub-discriminator is a PatchGAN that outputs a spatial map of
  real/fake predictions.

Why Spectral Normalization:
  SN constrains the Lipschitz constant of the discriminator, providing:
  - More stable GAN training (prevents mode collapse)
  - Better gradient flow to the generator
  - Removes the need for careful learning rate balancing
  - Compatible with LSGAN/hinge loss
  In practice, SN + InstanceNorm is the recommended combination for
  conditional image synthesis tasks (SPADE, pix2pixHD v2, etc.).

Reference:
  Wang et al., "High-Resolution Image Synthesis and Semantic Manipulation
  with Conditional GANs", CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator with spectral normalization.

    Architecture: C64 → C128 → C256 → C512 → 1
    Each layer: SpectralNorm(Conv) → InstanceNorm → LeakyReLU(0.2)
    First layer has no normalization, last layer outputs single channel.

    Args:
        input_nc:         number of input channels
        ndf:              base number of filters (default: 64)
        n_layers:         number of intermediate conv layers (default: 3)
        use_spectral_norm: apply spectral normalization (default: True)
        get_interm_feat:  return intermediate features for feature matching
    """

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
        get_interm_feat: bool = True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.get_interm_feat = get_interm_feat

        def maybe_sn(module):
            return spectral_norm(module) if use_spectral_norm else module

        kw = 4
        padw = 2  # ceil((kw-1)/2), same as pix2pixHD

        # First layer: no normalization
        layers = [
            [maybe_sn(nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw)),
             nn.LeakyReLU(0.2, inplace=True)]
        ]

        # Intermediate layers
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers.append([
                maybe_sn(nn.Conv2d(nf_prev, nf, kw, stride=2, padding=padw)),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # Penultimate layer: stride=1
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers.append([
            maybe_sn(nn.Conv2d(nf_prev, nf, kw, stride=1, padding=padw)),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        # Final layer: single channel output (no activation for LSGAN)
        layers.append([
            maybe_sn(nn.Conv2d(nf, 1, kw, stride=1, padding=padw)),
        ])

        # Register each group as a separate Sequential for feature extraction
        if get_interm_feat:
            for n in range(len(layers)):
                setattr(self, f"layer{n}", nn.Sequential(*layers[n]))
        else:
            flat = []
            for layer_group in layers:
                flat.extend(layer_group)
            self.model = nn.Sequential(*flat)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: (N, input_nc, H, W) input tensor

        Returns:
            If get_interm_feat: list of intermediate feature tensors
            Otherwise: single output tensor (N, 1, H', W')
        """
        if self.get_interm_feat:
            features = []
            for n in range(self.n_layers + 2):
                layer = getattr(self, f"layer{n}")
                x = layer(x)
                features.append(x)
            return features
        else:
            return self.model(x)


class MultiscaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator.

    Operates at num_D different scales by progressively downsampling the input.
    Each scale has its own NLayerDiscriminator. This multi-scale approach helps
    capture both local texture and global structure.

    For conditional GAN: input = concat(condition, image) along channel dim.
    The caller is responsible for the concatenation.

    Args:
        input_nc:         total input channels (condition + image channels)
        ndf:              base filters per sub-discriminator (default: 64)
        n_layers:         conv layers per sub-discriminator (default: 3)
        num_D:            number of discriminator scales (default: 3)
        use_spectral_norm: apply SN to all conv layers (default: True)
        get_interm_feat:  return intermediate features (default: True)
        enabled:          ON/OFF switch (default: True)
    """

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        num_D: int = 3,
        use_spectral_norm: bool = True,
        get_interm_feat: bool = True,
        enabled: bool = True,
    ):
        super().__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.get_interm_feat = get_interm_feat
        self.enabled = enabled

        for i in range(num_D):
            netD = NLayerDiscriminator(
                input_nc=input_nc,
                ndf=ndf,
                n_layers=n_layers,
                use_spectral_norm=use_spectral_norm,
                get_interm_feat=get_interm_feat,
            )
            setattr(self, f"discriminator_{i}", netD)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=1, count_include_pad=False
        )

    def forward(self, x: torch.Tensor):
        """
        Multi-scale forward pass.

        Args:
            x: (N, input_nc, H, W) input tensor (already concatenated condition + image)

        Returns:
            List of num_D results. Each result is:
              - If get_interm_feat: list of intermediate feature tensors
              - Otherwise: list containing single output tensor
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
                input_downsampled = self.downsample(input_downsampled)
        return results
