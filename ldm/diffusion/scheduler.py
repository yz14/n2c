"""
Noise schedulers for Latent Diffusion Model.

Implements:
  - DDPMScheduler: full DDPM forward process (adding noise) and single-step
    reverse (for training loss computation and slow sampling).
  - DDIMScheduler: accelerated deterministic/stochastic sampling via DDIM
    (Denoising Diffusion Implicit Models, Song et al. 2020).

Both schedulers use a pre-computed beta schedule (linear or cosine) and
store all derived quantities (alphas, cumulative products, etc.) as buffers.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch


def make_beta_schedule(
    schedule: str,
    n_timesteps: int,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    cosine_s: float = 8e-3,
) -> np.ndarray:
    """
    Create a beta noise schedule.

    Args:
        schedule: "linear" or "cosine".
        n_timesteps: number of diffusion timesteps.
        linear_start: start value for linear schedule.
        linear_end: end value for linear schedule.
        cosine_s: offset for cosine schedule.

    Returns:
        (n_timesteps,) array of beta values.
    """
    if schedule == "linear":
        betas = np.linspace(
            linear_start ** 0.5, linear_end ** 0.5, n_timesteps, dtype=np.float64
        ) ** 2
    elif schedule == "cosine":
        steps = np.arange(n_timesteps + 1, dtype=np.float64) / n_timesteps
        alphas_bar = np.cos((steps + cosine_s) / (1 + cosine_s) * np.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")
    return betas


class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Model scheduler.

    Handles:
      - Forward process: q(x_t | x_0) = N(sqrt(ᾱ_t) x_0, (1-ᾱ_t) I)
      - Training: add noise at random timestep, compute loss on predicted noise.
      - Single-step reverse: p(x_{t-1} | x_t) for slow (T-step) sampling.

    All schedule tensors are stored on CPU and moved to the appropriate device
    when needed via the `to()` method.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        prediction_type: str = "epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        betas = make_beta_schedule(
            beta_schedule, num_train_timesteps,
            linear_start=beta_start, linear_end=beta_end,
        )
        self.betas = torch.from_numpy(betas).float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )

        # Pre-compute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # Posterior q(x_{t-1} | x_t, x_0) parameters
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> "DDPMScheduler":
        """Move all schedule tensors to the specified device."""
        self._device = device
        for attr in [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod",
            "posterior_variance", "posterior_log_variance_clipped",
            "posterior_mean_coef1", "posterior_mean_coef2",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Gather values from `a` at indices `t` and reshape for broadcasting."""
        B = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(B, *((1,) * (len(x_shape) - 1)))

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0) = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε.

        Args:
            x_start: (B, C, H, W) clean latent.
            noise: (B, C, H, W) Gaussian noise.
            timesteps: (B,) integer timestep indices.

        Returns:
            (B, C, H, W) noisy latent at timestep t.
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x_0 from x_t and predicted noise: x_0 = (x_t - sqrt(1-ᾱ) ε) / sqrt(ᾱ)."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def ddpm_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Single DDPM reverse step: p(x_{t-1} | x_t).

        Args:
            model_output: predicted noise ε_θ(x_t, t).
            timestep: current timestep index.
            sample: current noisy sample x_t.

        Returns:
            x_{t-1}
        """
        t = torch.tensor([timestep], device=sample.device, dtype=torch.long)

        # Predict x_0
        pred_x0 = self.predict_start_from_noise(sample, t, model_output)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Posterior mean
        coef1 = self._extract(self.posterior_mean_coef1, t, sample.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, sample.shape)
        posterior_mean = coef1 * pred_x0 + coef2 * sample

        # Posterior variance
        if timestep == 0:
            return posterior_mean
        else:
            log_var = self._extract(self.posterior_log_variance_clipped, t, sample.shape)
            noise = torch.randn(
                sample.shape, generator=generator,
                device=sample.device, dtype=sample.dtype,
            )
            return posterior_mean + torch.exp(0.5 * log_var) * noise


class DDIMScheduler:
    """
    DDIM (Denoising Diffusion Implicit Models) scheduler for accelerated sampling.

    Uses a subset of DDPM timesteps for fewer denoising steps (e.g., 50 instead
    of 1000), with an eta parameter controlling stochasticity (eta=0 is fully
    deterministic).
    """

    def __init__(
        self,
        ddpm_scheduler: DDPMScheduler,
        num_inference_steps: int = 50,
        eta: float = 0.0,
    ):
        self.ddpm = ddpm_scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        # Compute DDIM timestep subset (uniform spacing)
        step_ratio = ddpm_scheduler.num_train_timesteps // num_inference_steps
        self.timesteps = np.arange(0, ddpm_scheduler.num_train_timesteps, step_ratio)
        self.timesteps = np.flip(self.timesteps).copy()  # reverse for sampling

        # Pre-compute DDIM parameters for each step
        alphas_cumprod = ddpm_scheduler.alphas_cumprod.cpu().numpy()
        self.ddim_alphas = alphas_cumprod[self.timesteps]
        self.ddim_alphas_prev = np.concatenate(
            [[1.0], alphas_cumprod[self.timesteps[:-1]]]
        )
        # Clamp argument to avoid sqrt of negative values (numerical edge cases)
        sigma_arg = np.clip(
            (1 - self.ddim_alphas_prev)
            / np.clip(1 - self.ddim_alphas, 1e-20, None)
            * (1 - self.ddim_alphas / np.clip(self.ddim_alphas_prev, 1e-20, None)),
            0.0, None,
        )
        self.ddim_sigmas = eta * np.sqrt(sigma_arg)

    @property
    def device(self) -> torch.device:
        return self.ddpm.device

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        step_index: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single DDIM reverse step.

        Args:
            model_output: predicted noise ε_θ(x_t, t).
            step_index: index into self.timesteps (0 = highest noise).
            sample: current noisy sample x_t.

        Returns:
            (x_{t-1}, pred_x0)
        """
        device = sample.device
        alpha_t = torch.tensor(self.ddim_alphas[step_index], device=device, dtype=sample.dtype)
        alpha_prev = torch.tensor(self.ddim_alphas_prev[step_index], device=device, dtype=sample.dtype)
        sigma_t = torch.tensor(self.ddim_sigmas[step_index], device=device, dtype=sample.dtype)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

        # Predict x_0
        pred_x0 = (sample - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1.0 - alpha_prev - sigma_t ** 2) * model_output

        # Noise (stochastic component)
        if sigma_t > 0 and step_index > 0:
            noise = torch.randn(
                sample.shape, generator=generator,
                device=device, dtype=sample.dtype,
            )
        else:
            noise = torch.zeros_like(sample)

        x_prev = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma_t * noise
        return x_prev, pred_x0
