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
        latent_scale_factor: scaling factor for latent normalization.
        cfg_scale: Classifier-Free Guidance scale. 1.0 = disabled.
            Requires the UNet to have been trained with condition dropping.
        dynamic_threshold_percentile: percentile for dynamic thresholding
            of pred_x0 during DDIM sampling. 0.0 = disabled.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: DiffusionUNet,
        scheduler: DDPMScheduler,
        latent_scale_factor: float = 1.0,
        cfg_scale: float = 1.0,
        dynamic_threshold_percentile: float = 0.0,
    ):
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.latent_scale_factor = latent_scale_factor
        self.cfg_scale = cfg_scale
        self.dynamic_threshold_percentile = dynamic_threshold_percentile

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @torch.no_grad()
    def encode_condition(self, ncct: torch.Tensor) -> torch.Tensor:
        """Encode NCCT input to latent space (deterministic, uses posterior mode).

        Applies latent_scale_factor for consistency with diffusion training.
        """
        posterior = self.vae.encode(ncct)
        return posterior.mode() * self.latent_scale_factor

    @staticmethod
    def _dynamic_threshold(pred_x0: torch.Tensor, percentile: float) -> torch.Tensor:
        """Apply dynamic thresholding to pred_x0 (adapted for latent space).

        Clips pred_x0 at the given percentile of absolute values per sample.
        Unlike Imagen's pixel-space version, we do NOT rescale by /s because
        latent values have a natural scale (std ≈ 1.0 after latent_scale_factor)
        that the VAE decoder expects. Rescaling would shrink the latent and
        produce washed-out / distorted images.
        """
        B = pred_x0.shape[0]
        flat = pred_x0.reshape(B, -1).abs()
        s = torch.quantile(flat, percentile, dim=1)  # (B,)
        s = torch.clamp(s, min=1.0)  # never shrink below 1.0
        s = s.reshape(B, 1, 1, 1)
        return pred_x0.clamp(-s, s)

    @torch.no_grad()
    def sample(
        self,
        ncct: torch.Tensor,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        return_intermediates: bool = False,
        verbose: bool = True,
        cfg_scale: Optional[float] = None,
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
            cfg_scale: override self.cfg_scale if provided. 1.0 = no guidance.

        Returns:
            (B, C, H, W) predicted CTA image (in normalized space).
            If return_intermediates, returns (final, list_of_intermediates).
        """
        device = self.device
        w = cfg_scale if cfg_scale is not None else self.cfg_scale
        use_cfg = w > 1.0

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
        dtp = self.dynamic_threshold_percentile
        for i in iterator:
            t_val = ddim.timesteps[i]

            if use_cfg:
                # Classifier-Free Guidance: batched conditional + unconditional
                z_doubled = torch.cat([z_noisy, z_noisy], dim=0)
                z_cond = torch.cat([z_ncct, torch.zeros_like(z_ncct)], dim=0)
                t = torch.full((z_doubled.shape[0],), t_val, device=device, dtype=torch.long)
                model_input = torch.cat([z_doubled, z_cond], dim=1)
                noise_pred_both = self.unet(model_input, t)
                noise_pred_cond, noise_pred_uncond = noise_pred_both.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + w * (noise_pred_cond - noise_pred_uncond)
            else:
                t = torch.full((z_noisy.shape[0],), t_val, device=device, dtype=torch.long)
                model_input = torch.cat([z_noisy, z_ncct], dim=1)
                noise_pred = self.unet(model_input, t)

            # Save z_noisy before step for dynamic thresholding recomputation
            z_before = z_noisy

            z_noisy, pred_x0 = ddim.step(noise_pred, i, z_before, generator=generator)

            # Dynamic thresholding on pred_x0 (reduces artifacts from extreme values)
            if dtp > 0:
                pred_x0 = self._dynamic_threshold(pred_x0, dtp)
                # Re-derive epsilon from thresholded pred_x0
                alpha_t = float(ddim.ddim_alphas[i])
                alpha_prev = float(ddim.ddim_alphas_prev[i])
                sigma_t = float(ddim.ddim_sigmas[i])
                sqrt_alpha_t = alpha_t ** 0.5
                sqrt_1m_alpha_t = max((1.0 - alpha_t) ** 0.5, 1e-8)
                pred_eps = (z_before - sqrt_alpha_t * pred_x0) / sqrt_1m_alpha_t
                # Recompute x_prev from thresholded pred_x0 and corrected epsilon
                dir_xt = ((1.0 - alpha_prev - sigma_t ** 2) ** 0.5) * pred_eps
                z_noisy = (alpha_prev ** 0.5) * pred_x0 + dir_xt

            if return_intermediates:
                intermediates.append(pred_x0.clone())

        # 5. Decode latent to image space (undo scaling before decode)
        cta_pred = self.vae.decode(z_noisy / self.latent_scale_factor)

        if return_intermediates:
            return cta_pred, intermediates
        return cta_pred

    @torch.no_grad()
    def sample_ddpm(
        self,
        ncct: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        verbose: bool = True,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate CTA prediction using full DDPM sampling (slow, T steps).

        Mainly useful for debugging or quality comparison with DDIM.
        """
        device = self.device
        w = cfg_scale if cfg_scale is not None else self.cfg_scale
        use_cfg = w > 1.0

        z_ncct = self.encode_condition(ncct)
        z_noisy = torch.randn(
            z_ncct.shape, generator=generator, device=device, dtype=z_ncct.dtype,
        )

        T = self.scheduler.num_train_timesteps
        iterator = reversed(range(T))
        if verbose:
            iterator = tqdm(iterator, desc="DDPM Sampling", total=T, leave=False)

        for t_val in iterator:
            if use_cfg:
                z_doubled = torch.cat([z_noisy, z_noisy], dim=0)
                z_cond = torch.cat([z_ncct, torch.zeros_like(z_ncct)], dim=0)
                t = torch.full((z_doubled.shape[0],), t_val, device=device, dtype=torch.long)
                model_input = torch.cat([z_doubled, z_cond], dim=1)
                noise_pred_both = self.unet(model_input, t)
                noise_pred_cond, noise_pred_uncond = noise_pred_both.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + w * (noise_pred_cond - noise_pred_uncond)
            else:
                t = torch.full((z_noisy.shape[0],), t_val, device=device, dtype=torch.long)
                model_input = torch.cat([z_noisy, z_ncct], dim=1)
                noise_pred = self.unet(model_input, t)

            z_noisy = self.scheduler.ddpm_step(noise_pred, t_val, z_noisy, generator=generator)

        return self.vae.decode(z_noisy / self.latent_scale_factor)
