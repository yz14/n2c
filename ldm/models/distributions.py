"""
Diagonal Gaussian distribution for VAE latent space.

Used by AutoencoderKL to parameterize the posterior q(z|x)
with a diagonal covariance Gaussian. Supports sampling, KL divergence,
and mode (deterministic) access.
"""

import torch
import numpy as np


class DiagonalGaussianDistribution:
    """
    Gaussian distribution with diagonal covariance, parameterized by
    concatenated [mean, logvar] along channel dimension.

    Args:
        parameters: (B, 2*C, H, W) tensor — first half is mean, second half is logvar.
        deterministic: if True, std=0 (acts as a delta distribution at the mean).
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device
            )

    def sample(self, generator: torch.Generator = None) -> torch.Tensor:
        """Sample z ~ N(mean, std^2)."""
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * noise

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        """
        Compute KL divergence KL(self || other).

        If other is None, computes KL(self || N(0, I)).

        Returns:
            (B,) tensor of per-sample KL values.
        """
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device)

        if other is None:
            # KL(N(mu, sigma^2) || N(0, I))
            return 0.5 * torch.sum(
                self.mean.pow(2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        else:
            # KL(self || other) for two diagonal Gaussians
            return 0.5 * torch.sum(
                (self.mean - other.mean).pow(2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3],
            )

    def nll(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood of a sample under this distribution.

        Returns:
            (B,) tensor of per-sample NLL values.
        """
        if self.deterministic:
            return torch.zeros(sample.shape[0], device=sample.device)

        log2pi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            log2pi + self.logvar + (sample - self.mean).pow(2) / self.var,
            dim=[1, 2, 3],
        )

    def mode(self) -> torch.Tensor:
        """Return the mode (= mean) of the distribution."""
        return self.mean
