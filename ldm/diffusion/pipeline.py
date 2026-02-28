"""
Conditional Latent Diffusion Pipeline for NCCT→CTA translation.

Orchestrates the full inference process:
  1. Encode NCCT input with the VAE encoder → z_ncct (condition latent)
  2. Start from random noise in latent space
  3. Iteratively denoise using the diffusion UNet conditioned on z_ncct
  4. Decode the denoised latent with the VAE decoder → predicted CTA

Supports both DDPM (slow, T steps) and DDIM (fast, configurable steps) sampling.
"""

import logging
from typing import Optional

import torch
from tqdm import tqdm

from ..models.autoencoder import AutoencoderKL
from ..models.unet import DiffusionUNet
from .scheduler import DDPMScheduler, DDIMScheduler

logger = logging.getLogger(__name__)


class ConditionalLDMPipeline:
    """
    Inference pipeline for conditional latent diffusion.

    Usage:
        pipeline = ConditionalLDMPipeline(vae, unet, ddpm_scheduler)
        cta_pred = pipeline.sample(ncct_input, num_inference_steps=50)

    Args:
        vae: trained AutoencoderKL.
        unet: trained conditional DiffusionUNet.
        scheduler: DDPMScheduler with the noise schedule used during training.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: DiffusionUNet,
        scheduler: DDPMScheduler,
    ):
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @torch.no_grad()
    def encode_condition(self, ncct: torch.Tensor) -> torch.Tensor:
        """Encode NCCT input to latent space (deterministic, uses posterior mode)."""
        posterior = self.vae.encode(ncct)
        return posterior.mode()

    @torch.no_grad()
    def sample(
        self,
        ncct: torch.Tensor,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        return_intermediates: bool = False,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Generate CTA prediction from NCCT input via DDIM sampling.

        Args:
            ncct: (B, C, H, W) NCCT input on device.
            num_inference_steps: number of DDIM denoising steps.
            eta: DDIM stochasticity (0 = deterministic).
            generator: optional random generator for reproducibility.
            return_intermediates: if True, return list of intermediate predictions.
            verbose: show progress bar.

        Returns:
            (B, C, H, W) predicted CTA image (in normalized space).
            If return_intermediates, returns (final, list_of_intermediates).
        """
        device = self.device

        # 1. Encode condition
        z_ncct = self.encode_condition(ncct)

        # 2. Initialize from random noise
        z_shape = z_ncct.shape  # (B, z_ch, H_lat, W_lat)
        z_noisy = torch.randn(
            z_shape, generator=generator, device=device, dtype=z_ncct.dtype,
        )

        # 3. Setup DDIM scheduler
        ddim = DDIMScheduler(
            self.scheduler, num_inference_steps=num_inference_steps, eta=eta,
        )

        intermediates = []
        iterator = range(len(ddim.timesteps))
        if verbose:
            iterator = tqdm(iterator, desc="DDIM Sampling", leave=False)

        # 4. Iterative denoising
        for i in iterator:
            t_val = ddim.timesteps[i]
            t = torch.full((z_noisy.shape[0],), t_val, device=device, dtype=torch.long)

            # Concat condition: [noisy_z_cta, z_ncct]
            model_input = torch.cat([z_noisy, z_ncct], dim=1)
            noise_pred = self.unet(model_input, t)

            z_noisy, pred_x0 = ddim.step(noise_pred, i, z_noisy, generator=generator)

            if return_intermediates:
                intermediates.append(pred_x0.clone())

        # 5. Decode latent to image space
        cta_pred = self.vae.decode(z_noisy)

        if return_intermediates:
            return cta_pred, intermediates
        return cta_pred

    @torch.no_grad()
    def sample_ddpm(
        self,
        ncct: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Generate CTA prediction using full DDPM sampling (slow, T steps).

        Mainly useful for debugging or quality comparison with DDIM.
        """
        device = self.device

        z_ncct = self.encode_condition(ncct)
        z_noisy = torch.randn(
            z_ncct.shape, generator=generator, device=device, dtype=z_ncct.dtype,
        )

        T = self.scheduler.num_train_timesteps
        iterator = reversed(range(T))
        if verbose:
            iterator = tqdm(iterator, desc="DDPM Sampling", total=T, leave=False)

        for t_val in iterator:
            t = torch.full((z_noisy.shape[0],), t_val, device=device, dtype=torch.long)
            model_input = torch.cat([z_noisy, z_ncct], dim=1)
            noise_pred = self.unet(model_input, t)
            z_noisy = self.scheduler.ddpm_step(noise_pred, t_val, z_noisy, generator=generator)

        return self.vae.decode(z_noisy)
