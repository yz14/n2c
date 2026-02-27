"""
Registration network for NCCT→CTA image translation.

Inspired by VoxelMorph, this module provides:
  - SpatialTransformer: warp images using 2D displacement fields via grid_sample
  - VelocityFieldIntegrator: scaling & squaring for diffeomorphic registration
  - RegistrationNet: lightweight encoder-decoder predicting a 2D displacement field
    from concatenated (source, target) images

The registration network aligns the generated CTA with the ground truth CTA,
compensating for imperfect spatial correspondence in the training data.
An ON/OFF switch (enabled flag) allows staged training.

Reference:
  Balakrishnan et al., "VoxelMorph: A Learning Framework for Deformable
  Medical Image Registration", IEEE TMI, 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    2D spatial transformer that warps an image by a displacement field.

    Uses PyTorch's grid_sample for differentiable bilinear interpolation.
    The displacement field is in voxel coordinates (not normalized).
    """

    def __init__(self, mode: str = "bilinear", padding_mode: str = "zeros"):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(
        self, image: torch.Tensor, displacement: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp image by displacement field.

        Args:
            image:        (N, C, H, W) tensor to warp
            displacement: (N, 2, H, W) displacement field in voxel units
                          displacement[:, 0] = dy, displacement[:, 1] = dx

        Returns:
            Warped image (N, C, H, W)
        """
        N, _, H, W = image.shape

        # Build identity grid: (H, W, 2) with values in [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=image.device, dtype=image.dtype),
            torch.linspace(-1, 1, W, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        identity = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2) — x, y order
        identity = identity.unsqueeze(0).expand(N, -1, -1, -1)  # (N, H, W, 2)

        # Convert displacement from voxel units to normalized [-1, 1] units
        # displacement[:, 0] is dy (row), displacement[:, 1] is dx (col)
        disp_norm = torch.zeros_like(identity)
        disp_norm[..., 0] = displacement[:, 1] / (W - 1) * 2  # dx → normalized x
        disp_norm[..., 1] = displacement[:, 0] / (H - 1) * 2  # dy → normalized y

        grid = identity + disp_norm

        return F.grid_sample(
            image, grid, mode=self.mode, padding_mode=self.padding_mode,
            align_corners=True,
        )


class VelocityFieldIntegrator(nn.Module):
    """
    Integrate a stationary velocity field via scaling and squaring.

    Produces a diffeomorphic displacement field that is smooth and invertible.

    Args:
        steps: number of squaring steps (higher = more accurate, typical: 5-7)
    """

    def __init__(self, steps: int = 7):
        super().__init__()
        assert steps >= 0
        self.steps = steps
        self.scale = 1.0 / (2 ** steps)
        self.transformer = SpatialTransformer()

    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Integrate velocity field → displacement field.

        Args:
            velocity: (N, 2, H, W) stationary velocity field

        Returns:
            displacement: (N, 2, H, W) integrated displacement field
        """
        disp = velocity * self.scale
        for _ in range(self.steps):
            disp = disp + self.transformer(disp, disp)
        return disp


class _ConvBlock(nn.Module):
    """Conv + InstanceNorm + LeakyReLU block for the registration encoder/decoder."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RegistrationNet(nn.Module):
    """
    Lightweight encoder-decoder network for 2D image registration.

    Takes concatenated (source, target) images and predicts a 2D displacement
    field. Architecture follows VoxelMorph: small UNet with skip connections.

    Features:
      - 4-level encoder-decoder with skip connections
      - InstanceNorm + LeakyReLU activations
      - Flow layer initialized with near-zero weights for stable training start
      - Optional diffeomorphic integration via scaling & squaring
      - ON/OFF switch: when disabled, acts as identity (returns zero displacement)

    Args:
        in_channels:       number of channels per image (C)
        nb_features:       list of feature counts per encoder level
        integration_steps: steps for velocity field integration (0 = direct displacement)
        enabled:           if False, network is bypassed (identity mode)
    """

    def __init__(
        self,
        in_channels: int = 3,
        nb_features: tuple = (16, 32, 32, 32),
        integration_steps: int = 7,
        enabled: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.enabled = enabled
        self.integration_steps = integration_steps

        # --- Encoder ---
        enc_channels = [in_channels * 2] + list(nb_features)
        self.encoders = nn.ModuleList()
        for i in range(len(nb_features)):
            self.encoders.append(
                _ConvBlock(enc_channels[i], enc_channels[i + 1], stride=2)
            )

        # --- Decoder (with skip connections) ---
        dec_channels = list(reversed(nb_features))
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(len(dec_channels) - 1):
            # After upsample + skip concat: dec_channels[i] + enc_skip_channels
            skip_ch = nb_features[len(nb_features) - 2 - i]
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            )
            self.decoders.append(
                _ConvBlock(dec_channels[i] + skip_ch, dec_channels[i + 1])
            )

        # Final upsample to input resolution
        self.final_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.final_conv = _ConvBlock(
            dec_channels[-1] + in_channels * 2, 16
        )

        # --- Flow layer (initialized near-zero for stable start) ---
        self.flow = nn.Conv2d(16, 2, kernel_size=3, padding=1)
        nn.init.normal_(self.flow.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow.bias)

        # --- Integration & Spatial Transformer ---
        if integration_steps > 0:
            self.integrator = VelocityFieldIntegrator(steps=integration_steps)
        else:
            self.integrator = None
        self.spatial_transformer = SpatialTransformer()

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """
        Register source to target.

        Args:
            source: (N, C, H, W) generated CTA (from UNet)
            target: (N, C, H, W) ground truth CTA

        Returns:
            dict with keys:
              - warped:       (N, C, H, W) warped source image
              - displacement: (N, 2, H, W) displacement field
              - velocity:     (N, 2, H, W) velocity field (if integration_steps > 0)
        """
        if not self.enabled:
            # Identity mode: no registration
            return {
                "warped": source,
                "displacement": torch.zeros(
                    source.shape[0], 2, source.shape[2], source.shape[3],
                    device=source.device, dtype=source.dtype,
                ),
                "velocity": None,
            }

        # Concatenate source and target
        x = torch.cat([source, target], dim=1)  # (N, 2C, H, W)

        # Encoder with skip connections
        skips = [x]
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # Decoder with skip connections (reverse order, skip the last encoder output)
        x = skips[-1]
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoders)):
            x = up(x)
            skip = skips[-(i + 2)]
            # Handle size mismatch from odd dimensions
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # Final upsample to input resolution + skip from input
        x = self.final_upsample(x)
        input_concat = skips[0]
        if x.shape != input_concat.shape:
            x = F.interpolate(x, size=input_concat.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, input_concat], dim=1)
        x = self.final_conv(x)

        # Predict velocity/displacement field
        velocity = self.flow(x)  # (N, 2, H, W)

        if self.integrator is not None:
            displacement = self.integrator(velocity)
        else:
            displacement = velocity
            velocity = None

        # Warp source image
        warped = self.spatial_transformer(source, displacement)

        return {
            "warped": warped,
            "displacement": displacement,
            "velocity": velocity,
        }
