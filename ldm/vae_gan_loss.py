"""
VAE-GAN Loss Module for VQGAN-style VAE Training.

Implements the adversarial training component for AutoencoderKL, following
the Taming Transformers (VQGAN) paper (Esser et al., CVPR 2021).

Key components:
  - PatchGAN discriminator (reused from Scheme 1)
  - Adaptive weight balancing: |∂L_rec/∂θ_last| / |∂L_GAN/∂θ_last|
  - Hinge or LSGAN loss
  - Optional: VGG perceptual loss, feature matching, R1 penalty

Training protocol:
  1. VAE update: freeze D, compute recon + KL + perceptual + GAN_g loss
  2. D update: freeze VAE, compute real/fake classification loss

The adaptive weight automatically balances reconstruction and adversarial
losses by comparing their gradient magnitudes at the decoder's last layer,
preventing either loss from dominating training.

Usage:
    vae_gan = VAEGANLoss(cfg.vae_gan, vae, device)
    # In training loop:
    g_loss_dict = vae_gan.compute_g_loss(recon, target, posterior, epoch)
    d_loss_dict = vae_gan.compute_d_loss(recon.detach(), target, epoch)
    vae_gan.step_d(d_loss_dict["d_loss"])
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ldm.config import VAEGANConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptive weight computation (VQGAN core)
# ---------------------------------------------------------------------------

def compute_adaptive_weight(
    nll_loss: torch.Tensor,
    g_loss: torch.Tensor,
    last_layer_weight: torch.Tensor,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """Compute VQGAN adaptive weight for balancing reconstruction vs GAN loss.

    Balances the two losses so they contribute equally at the decoder's
    final layer, preventing either from dominating:

        λ = |∂L_rec / ∂θ_last| / |∂L_GAN / ∂θ_last|

    Args:
        nll_loss:           reconstruction loss (scalar, with grad)
        g_loss:             generator adversarial loss (scalar, with grad)
        last_layer_weight:  decoder's final conv layer weight tensor
        max_weight:         clamp upper bound for stability

    Returns:
        adaptive weight λ (scalar tensor)
    """
    nll_grads = torch.autograd.grad(
        nll_loss, last_layer_weight, retain_graph=True
    )[0]
    g_grads = torch.autograd.grad(
        g_loss, last_layer_weight, retain_graph=True
    )[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, max_weight).detach()
    return d_weight


# ---------------------------------------------------------------------------
# GAN loss functions (lightweight, for single-discriminator output)
# ---------------------------------------------------------------------------

def _hinge_d_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator."""
    return 0.5 * (F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean())


def _hinge_g_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator."""
    return -fake_pred.mean()


def _lsgan_d_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """LSGAN loss for discriminator."""
    return 0.5 * (
        F.mse_loss(real_pred, torch.ones_like(real_pred)) +
        F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
    )


def _lsgan_g_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    """LSGAN loss for generator."""
    return F.mse_loss(fake_pred, torch.ones_like(fake_pred))


# ---------------------------------------------------------------------------
# Feature matching loss
# ---------------------------------------------------------------------------

def _feature_matching_loss(
    real_features: list,
    fake_features: list,
) -> torch.Tensor:
    """L1 feature matching loss across discriminator layers."""
    loss = 0.0
    n = 0
    for rf, ff in zip(real_features[:-1], fake_features[:-1]):
        loss = loss + F.l1_loss(ff, rf.detach())
        n += 1
    return loss / max(n, 1)


# ---------------------------------------------------------------------------
# R1 gradient penalty
# ---------------------------------------------------------------------------

def _r1_penalty(
    real_pred: torch.Tensor,
    real_input: torch.Tensor,
) -> torch.Tensor:
    """R1 gradient penalty (Mescheder et al., 2018).

    real_input must have requires_grad=True before D forward.
    """
    gradients = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_input,
        create_graph=True,
        only_inputs=True,
    )[0]
    return gradients.pow(2).flatten(1).sum(1).mean()


# ---------------------------------------------------------------------------
# VAEGANLoss
# ---------------------------------------------------------------------------

class VAEGANLoss(nn.Module):
    """
    VQGAN-style adversarial loss manager for VAE training.

    Encapsulates the discriminator, its optimizer/scheduler, and all
    GAN-related loss computations. The trainer only needs to call
    compute_g_loss() and compute_d_loss().

    Architecture:
      - Single NLayerDiscriminator (PatchGAN) operating on pixel-space images
      - D sees the reconstructed image directly (unconditional)
      - Intermediate features extracted for optional feature matching

    Args:
        cfg:    VAEGANConfig with all discriminator hyperparameters
        vae:    AutoencoderKL model (for get_last_layer() in adaptive weight)
        device: torch device
        total_steps: total training steps (for LR scheduler)
    """

    def __init__(
        self,
        cfg: VAEGANConfig,
        vae: nn.Module,
        device: torch.device,
        total_steps: int = 100000,
    ):
        super().__init__()
        self.cfg = cfg
        self.vae = vae
        self.device = device
        self.d_step_count = 0

        if not cfg.enabled:
            logger.info("  VAE-GAN: DISABLED")
            self.discriminator = None
            self.perceptual_loss = None
            return

        # Import and create discriminator (reuse from Scheme 1)
        from models.discriminator import NLayerDiscriminator, init_d_weights

        input_nc = vae.decoder.conv_out.out_channels  # = VAE output channels
        self.discriminator = NLayerDiscriminator(
            input_nc=input_nc,
            ndf=cfg.ndf,
            n_layers=cfg.n_layers,
            use_spectral_norm=cfg.use_spectral_norm,
            get_interm_feat=(cfg.feat_match_weight > 0),
        ).to(device)
        init_d_weights(self.discriminator)

        # GAN loss functions
        if cfg.gan_loss_type == "hinge":
            self._d_loss_fn = _hinge_d_loss
            self._g_loss_fn = _hinge_g_loss
        elif cfg.gan_loss_type == "lsgan":
            self._d_loss_fn = _lsgan_d_loss
            self._g_loss_fn = _lsgan_g_loss
        else:
            raise ValueError(f"Unknown gan_loss_type: {cfg.gan_loss_type}")

        # Perceptual loss (VGG-based)
        self.perceptual_loss = None
        if cfg.perceptual_weight > 0:
            from models.perceptual_loss import VGGPerceptualLoss
            self.perceptual_loss = VGGPerceptualLoss().to(device)
            self.perceptual_loss.eval()
            logger.info(f"  VAE-GAN perceptual loss: weight={cfg.perceptual_weight}")

        # Negative sample augmentation for D (degraded images as extra fakes)
        self.neg_augment = None
        if cfg.d_neg_augment:
            from data.cta_degrade import CTADegradation
            self.neg_augment = CTADegradation(
                direct_prob=0.7,  # mostly direct CTA degradation
                blur_prob=0.8,
                noise_prob=0.6,
            )
            logger.info("  VAE-GAN: D negative sample augmentation ENABLED")

        # D optimizer
        self.optimizer_d = AdamW(
            self.discriminator.parameters(),
            lr=cfg.lr,
            betas=(0.5, 0.9),
        )

        # D LR scheduler (warmup + cosine, matching VAE)
        warmup_steps = 500
        self.scheduler_d = LambdaLR(
            self.optimizer_d,
            lr_lambda=self._warmup_cosine_lambda(warmup_steps, total_steps),
        )

        # Load pretrained D weights if specified
        if cfg.disc_pretrained_path:
            self._load_pretrained_d(cfg.disc_pretrained_path)

        # Log
        n_params = sum(p.numel() for p in self.discriminator.parameters()) / 1e6
        logger.info(f"  VAE-GAN: ENABLED")
        logger.info(f"    D params:         {n_params:.2f}M")
        logger.info(f"    D input channels: {input_nc}")
        logger.info(f"    GAN loss:         {cfg.gan_loss_type}")
        logger.info(f"    GAN weight:       {cfg.gan_weight}")
        logger.info(f"    Adaptive weight:  {cfg.adaptive_weight} (max={cfg.adaptive_weight_max})")
        logger.info(f"    Perceptual weight:{cfg.perceptual_weight}")
        logger.info(f"    Feat match weight:{cfg.feat_match_weight}")
        logger.info(f"    D start epoch:    {cfg.disc_start_epoch}")
        logger.info(f"    D LR:             {cfg.lr}")
        logger.info(f"    R1 gamma:         {cfg.r1_gamma} (interval={cfg.r1_interval})")

    @staticmethod
    def _warmup_cosine_lambda(warmup_steps: int, total_steps: int):
        """Warmup + cosine decay LR lambda for D."""
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr_lambda

    def is_active(self, epoch: int) -> bool:
        """Check if D should be active at this epoch."""
        if not self.cfg.enabled or self.discriminator is None:
            return False
        return epoch >= self.cfg.disc_start_epoch

    # ------------------------------------------------------------------
    # Generator (VAE) loss
    # ------------------------------------------------------------------

    def compute_g_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        posterior,
        epoch: int,
        nll_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all generator-side losses (GAN + perceptual + adaptive weight).

        Call this AFTER computing the base reconstruction loss. The discriminator
        parameters are frozen during this computation.

        Args:
            recon:      (N, C, H, W) VAE reconstruction
            target:     (N, C, H, W) ground truth image
            posterior:  DiagonalGaussianDistribution from VAE encoder
            epoch:      current epoch (for disc_start check)
            nll_loss:   pre-computed reconstruction loss (for adaptive weight).
                        If None, L1 loss is computed internally.

        Returns:
            dict with: g_loss, perceptual, adaptive_w, disc_factor, gan_g
                       (+ feat_match if enabled)
        """
        result = {
            "g_loss": torch.tensor(0.0, device=self.device),
            "perceptual": torch.tensor(0.0, device=self.device),
            "gan_g": torch.tensor(0.0, device=self.device),
            "adaptive_w": torch.tensor(0.0, device=self.device),
            "disc_factor": torch.tensor(0.0, device=self.device),
        }

        if not self.cfg.enabled:
            return result

        # Perceptual loss (always active if weight > 0, independent of D)
        p_loss = torch.tensor(0.0, device=self.device)
        if self.perceptual_loss is not None and self.cfg.perceptual_weight > 0:
            p_loss = self.perceptual_loss(recon, target)
            result["perceptual"] = p_loss.detach()

        # GAN loss (only active after disc_start_epoch)
        disc_factor = 1.0 if self.is_active(epoch) else 0.0
        result["disc_factor"] = torch.tensor(disc_factor, device=self.device)

        g_adv_loss = torch.tensor(0.0, device=self.device)
        feat_match_loss = torch.tensor(0.0, device=self.device)

        if disc_factor > 0:
            # Freeze D for G update
            for p in self.discriminator.parameters():
                p.requires_grad = False

            if self.cfg.feat_match_weight > 0:
                # Need features for feature matching
                fake_features = self.discriminator(recon)
                g_adv_loss = self._g_loss_fn(fake_features[-1])

                with torch.no_grad():
                    real_features = self.discriminator(target)
                feat_match_loss = _feature_matching_loss(real_features, fake_features)
                result["feat_match"] = feat_match_loss.detach()
            else:
                fake_pred = self.discriminator(recon)
                if isinstance(fake_pred, list):
                    fake_pred = fake_pred[-1]
                g_adv_loss = self._g_loss_fn(fake_pred)

            result["gan_g"] = g_adv_loss.detach()

            # Unfreeze D
            for p in self.discriminator.parameters():
                p.requires_grad = True

        # Adaptive weight computation
        adaptive_w = torch.tensor(self.cfg.gan_weight, device=self.device)
        if disc_factor > 0 and self.cfg.adaptive_weight:
            # nll_loss = reconstruction + perceptual (the "content" losses)
            if nll_loss is None:
                nll_loss = F.l1_loss(recon, target)
            nll_with_perceptual = nll_loss + self.cfg.perceptual_weight * p_loss

            try:
                last_layer = self.vae.get_last_layer()
                adaptive_w = compute_adaptive_weight(
                    nll_with_perceptual, g_adv_loss, last_layer,
                    max_weight=self.cfg.adaptive_weight_max,
                )
            except RuntimeError:
                # Fallback if grad computation fails (e.g., no grad path)
                adaptive_w = torch.tensor(self.cfg.gan_weight, device=self.device)

        result["adaptive_w"] = adaptive_w.detach() if isinstance(adaptive_w, torch.Tensor) else torch.tensor(adaptive_w)

        # Total G loss = perceptual + adaptive_w * disc_factor * gan_g + feat_match
        total_g = (
            self.cfg.perceptual_weight * p_loss
            + adaptive_w * disc_factor * g_adv_loss
            + self.cfg.feat_match_weight * feat_match_loss
        )
        result["g_loss"] = total_g

        return result

    # ------------------------------------------------------------------
    # Discriminator loss
    # ------------------------------------------------------------------

    def compute_d_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        neg_samples: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute discriminator loss on real and fake images.

        Call this AFTER the VAE update. The recon tensor should be detached
        to prevent gradients flowing back to VAE.

        Args:
            recon:       (N, C, H, W) VAE reconstruction (DETACHED)
            target:      (N, C, H, W) ground truth image
            epoch:       current epoch
            neg_samples: (N, C, H, W) optional degraded images as extra fakes

        Returns:
            dict with: d_loss, d_real, d_fake (+ r1_penalty, d_neg if applicable)
        """
        result = {
            "d_loss": torch.tensor(0.0, device=self.device),
            "d_real": torch.tensor(0.0, device=self.device),
            "d_fake": torch.tensor(0.0, device=self.device),
        }

        if not self.is_active(epoch):
            return result

        # Real
        if self.cfg.r1_gamma > 0 and self.d_step_count % self.cfg.r1_interval == 0:
            target_r1 = target.detach().requires_grad_(True)
            real_out = self.discriminator(target_r1)
            real_pred = real_out[-1] if isinstance(real_out, list) else real_out
            r1 = _r1_penalty(real_pred, target_r1)
            result["r1_penalty"] = r1.detach()
        else:
            real_out = self.discriminator(target)
            real_pred = real_out[-1] if isinstance(real_out, list) else real_out
            r1 = torch.tensor(0.0, device=self.device)

        # Fake (must be detached)
        fake_out = self.discriminator(recon.detach())
        fake_pred = fake_out[-1] if isinstance(fake_out, list) else fake_out

        # D loss
        d_loss = self._d_loss_fn(real_pred, fake_pred)

        # Negative sample loss (degraded images as extra fakes)
        if neg_samples is not None:
            neg_out = self.discriminator(neg_samples.detach())
            neg_pred = neg_out[-1] if isinstance(neg_out, list) else neg_out
            # Treat degraded images as fake — D should reject them too
            neg_loss = self._d_loss_fn(real_pred.detach(), neg_pred)
            d_loss = d_loss + 0.5 * neg_loss
            result["d_neg"] = neg_pred.mean().detach()

        # Add R1 penalty
        if self.cfg.r1_gamma > 0 and self.d_step_count % self.cfg.r1_interval == 0:
            d_loss = d_loss + 0.5 * self.cfg.r1_gamma * r1

        result["d_loss"] = d_loss
        result["d_real"] = real_pred.mean().detach()
        result["d_fake"] = fake_pred.mean().detach()

        self.d_step_count += 1
        return result

    # ------------------------------------------------------------------
    # D optimizer step
    # ------------------------------------------------------------------

    def step_d(self, d_loss: torch.Tensor):
        """Backward + optimizer step for discriminator.

        Args:
            d_loss: discriminator loss (from compute_d_loss)
        """
        if self.discriminator is None:
            return

        self.optimizer_d.zero_grad()
        d_loss.backward()

        if self.cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.cfg.grad_clip_norm
            )

        self.optimizer_d.step()
        self.scheduler_d.step()

    # ------------------------------------------------------------------
    # State dict for checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serializable state for checkpointing."""
        if self.discriminator is None:
            return {}
        return {
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "scheduler_d_state_dict": self.scheduler_d.state_dict(),
            "d_step_count": self.d_step_count,
        }

    def load_state_dict_from_checkpoint(self, state: dict):
        """Restore state from checkpoint."""
        if self.discriminator is None or not state:
            return
        if "discriminator_state_dict" in state:
            self.discriminator.load_state_dict(state["discriminator_state_dict"])
            logger.info("  Loaded discriminator weights")
        if "optimizer_d_state_dict" in state:
            self.optimizer_d.load_state_dict(state["optimizer_d_state_dict"])
        if "scheduler_d_state_dict" in state:
            self.scheduler_d.load_state_dict(state["scheduler_d_state_dict"])
        if "d_step_count" in state:
            self.d_step_count = state["d_step_count"]

    def _load_pretrained_d(self, path: str):
        """Load pretrained discriminator weights.

        Supports loading from:
          - A VAE+GAN checkpoint (has 'vae_gan_loss' key with D state inside)
          - A standalone D state dict
        """
        logger.info(f"  Loading pretrained D from: {path}")
        state = torch.load(path, map_location=self.device)

        # Try VAE checkpoint format first
        if "vae_gan_loss" in state and "discriminator_state_dict" in state["vae_gan_loss"]:
            d_state = state["vae_gan_loss"]["discriminator_state_dict"]
        elif "discriminator_state_dict" in state:
            d_state = state["discriminator_state_dict"]
        else:
            d_state = state  # assume raw state dict

        missing, unexpected = self.discriminator.load_state_dict(d_state, strict=False)
        if missing:
            logger.warning(f"  Missing D keys: {missing[:5]}...")
        if unexpected:
            logger.warning(f"  Unexpected D keys: {unexpected[:5]}...")
        logger.info(f"  Loaded pretrained D weights ({len(d_state)} tensors)")

    def generate_negative_samples(
        self,
        ncct: torch.Tensor,
        cta: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Generate degraded images as extra negative samples for D.

        Returns None if d_neg_augment is disabled.
        """
        if self.neg_augment is None:
            return None
        with torch.no_grad():
            return self.neg_augment(ncct, cta)

    @property
    def d_lr(self) -> float:
        """Current discriminator learning rate."""
        if self.discriminator is None:
            return 0.0
        return self.optimizer_d.param_groups[0]["lr"]
