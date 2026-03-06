"""
VAE (AutoencoderKL) Trainer for Latent Diffusion Model — Stage 1.

Trains the VAE to reconstruct 2.5D medical images with:
  - L1 reconstruction loss (primary)
  - KL divergence regularization (weighted, typically 1e-6)
  - Optional VQGAN-style adversarial training (PatchGAN discriminator)
  - Optional VGG perceptual loss
  - Adaptive weight balancing (VQGAN: |∂L_rec/∂θ_last| / |∂L_GAN/∂θ_last|)
  - EMA of model weights
  - Cosine LR schedule with warmup
  - Gradient clipping
  - Checkpoint saving/resuming
  - Visualization grids (input | reconstruction | ground truth)

The trained VAE provides the latent space for Stage 2 (diffusion UNet).

Usage:
    python -m ldm.train_vae --config configs/ldm_default.yaml
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm import tqdm

from ldm.config import LDMConfig
from ldm.models.autoencoder import AutoencoderKL
from ldm.utils import MetricTracker, warmup_cosine_schedule as _warmup_cosine_schedule
from models.nn_utils import update_ema, load_pretrained_weights
from utils.visualization import save_sample_grid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VAE Trainer
# ---------------------------------------------------------------------------

class VAETrainer:
    """
    Training manager for AutoencoderKL (LDM Stage 1).

    Training flow per step:
      1. Augment batch on GPU → extract middle C slices
      2. VAE forward: encode → sample → decode
      3. Compute L1 reconstruction loss + KL divergence loss
      4. Backward + gradient clipping + optimizer step
      5. EMA update

    The VAE is trained on both NCCT and CTA images to learn a general
    latent representation for medical CT images. Each batch processes
    both modalities to maximize data utilization.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        augmentor,
        config: LDMConfig,
        device: torch.device,
        vae_gan_loss=None,
    ):
        self.vae = vae.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.augmentor = augmentor
        self.config = config
        self.device = device
        self.vae_gan_loss = vae_gan_loss  # VAEGANLoss instance (optional)

        tcfg = config.vae_train

        # Optimizer
        self.optimizer = AdamW(
            self.vae.parameters(),
            lr=tcfg.lr,
            weight_decay=tcfg.weight_decay,
        )

        # LR scheduler
        steps_per_epoch = len(train_loader)
        total_steps = tcfg.num_epochs * steps_per_epoch
        self.scheduler = self._build_scheduler(
            self.optimizer, tcfg.lr_scheduler, tcfg.warmup_steps, total_steps,
        )

        # Loss weights
        self.l1_weight = tcfg.l1_weight
        self.kl_weight = tcfg.kl_weight

        # EMA
        self.ema_rate = tcfg.ema_rate
        self.ema_params = [p.clone().detach() for p in self.vae.parameters()]

        # Gradient clipping
        self.grad_clip_norm = tcfg.grad_clip_norm

        # State
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        # Output
        self.output_dir = Path(tcfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Resume
        if tcfg.resume_checkpoint:
            self._load_checkpoint(tcfg.resume_checkpoint)

        # Log
        n_params = sum(p.numel() for p in self.vae.parameters()) / 1e6
        logger.info(f"VAETrainer initialized:")
        logger.info(f"  VAE params:       {n_params:.2f}M")
        logger.info(f"  L1 weight:        {self.l1_weight}")
        logger.info(f"  KL weight:        {self.kl_weight}")
        logger.info(f"  LR:               {tcfg.lr}")
        logger.info(f"  Grad clip norm:   {self.grad_clip_norm}")
        logger.info(f"  EMA rate:         {self.ema_rate}")
        if self.vae_gan_loss is not None and self.vae_gan_loss.cfg.enabled:
            logger.info(f"  VAE-GAN:          ENABLED (start epoch {self.vae_gan_loss.cfg.disc_start_epoch})")
        else:
            logger.info(f"  VAE-GAN:          DISABLED")

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    @staticmethod
    def _build_scheduler(optimizer, scheduler_type, warmup_steps, total_steps):
        if scheduler_type == "cosine":
            return LambdaLR(optimizer, _warmup_cosine_schedule(warmup_steps, total_steps))
        elif scheduler_type == "step":
            return StepLR(optimizer, step_size=50, gamma=0.5)
        return None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Run the full VAE training loop."""
        tcfg = self.config.vae_train
        logger.info(f"Starting VAE training for {tcfg.num_epochs} epochs")

        for epoch in range(self.start_epoch, tcfg.num_epochs):
            train_metrics = self._train_epoch(epoch)
            logger.info(f"Epoch {epoch+1}/{tcfg.num_epochs} [Train] {train_metrics}")

            # Validation
            if self.val_loader and (epoch + 1) % tcfg.val_interval == 0:
                val_metrics, val_vis = self._validate(epoch)
                logger.info(f"Epoch {epoch+1}/{tcfg.num_epochs} [Val]   {val_metrics}")
                val_loss = val_metrics.result()["loss"]
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f"  New best val loss: {val_loss:.6f}")
                self._save_visualizations(epoch, val_vis)

            if (epoch + 1) % tcfg.save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Epoch-level scheduler stepping (for "step" type)
            if self.scheduler and tcfg.lr_scheduler == "step":
                self.scheduler.step()

        self._save_checkpoint(tcfg.num_epochs - 1, is_best=False,
                              filename="checkpoint_final.pt")
        logger.info("VAE training complete.")

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> MetricTracker:
        """Run one training epoch.

        When VAE-GAN is enabled, each step has two phases:
          Phase 1 (VAE/G update): freeze D → compute recon + KL + GAN_g + perceptual
          Phase 2 (D update):     freeze VAE → compute D real/fake loss
        Gradient isolation ensures no leakage between VAE and D.
        """
        self.vae.train()
        tracker = MetricTracker()
        tcfg = self.config.vae_train
        train_vis_samples: List[Dict[str, torch.Tensor]] = []
        gan_active = (self.vae_gan_loss is not None and
                      self.vae_gan_loss.is_active(epoch))

        pbar = tqdm(self.train_loader, desc=f"VAE Epoch {epoch+1}", leave=False)
        for batch in pbar:
            ncct_3c = batch["ncct"].to(self.device)
            cta_3c = batch["cta"].to(self.device)
            mask_3c = batch["ncct_lung"].to(self.device)

            # Augment and extract middle C slices
            ncct, cta, mask = self.augmentor(ncct_3c, cta_3c, mask_3c, training=True)

            # ============================================================
            # Phase 1: VAE (Generator) update
            # ============================================================

            # Train on CTA images (primary target for reconstruction quality)
            recon_cta, posterior_cta = self.vae(cta)
            loss_cta, loss_dict_cta = self._compute_loss(cta, recon_cta, posterior_cta)

            # Also train on NCCT for a shared latent space
            recon_ncct, posterior_ncct = self.vae(ncct)
            loss_ncct, loss_dict_ncct = self._compute_loss(ncct, recon_ncct, posterior_ncct)

            # Combined reconstruction loss (equal weight for both modalities)
            recon_loss = 0.5 * (loss_cta + loss_ncct)

            # GAN + perceptual loss (applied to CTA reconstruction only)
            g_loss_dict = {}
            total_g_loss = torch.tensor(0.0, device=self.device)
            if self.vae_gan_loss is not None and self.vae_gan_loss.cfg.enabled:
                g_loss_dict = self.vae_gan_loss.compute_g_loss(
                    recon=recon_cta,
                    target=cta,
                    posterior=posterior_cta,
                    epoch=epoch,
                    nll_loss=loss_cta,
                )
                total_g_loss = g_loss_dict["g_loss"]

            # Total VAE loss = reconstruction + GAN/perceptual
            loss = recon_loss + total_g_loss

            # Backward for VAE
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = 0.0
            if self.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.vae.parameters(), self.grad_clip_norm
                ).item()

            self.optimizer.step()

            # LR scheduler (step-level for cosine warmup)
            if self.scheduler and tcfg.lr_scheduler == "cosine":
                self.scheduler.step()

            # EMA
            update_ema(self.ema_params, self.vae.parameters(), rate=self.ema_rate)

            # ============================================================
            # Phase 2: Discriminator update (VAE frozen)
            # ============================================================
            d_loss_dict = {}
            if gan_active:
                # Freeze VAE for D update
                for p in self.vae.parameters():
                    p.requires_grad = False

                # Generate negative samples (degraded images) if enabled
                neg_samples = self.vae_gan_loss.generate_negative_samples(ncct, cta)

                d_loss_dict = self.vae_gan_loss.compute_d_loss(
                    recon=recon_cta.detach(),
                    target=cta.detach(),
                    epoch=epoch,
                    neg_samples=neg_samples,
                )
                self.vae_gan_loss.step_d(d_loss_dict["d_loss"])

                # Unfreeze VAE
                for p in self.vae.parameters():
                    p.requires_grad = True

            # ============================================================
            # Metrics
            # ============================================================
            self.global_step += 1
            metrics = {
                "loss": loss.item(),
                "l1_cta": loss_dict_cta["l1"],
                "kl_cta": loss_dict_cta["kl"],
                "l1_ncct": loss_dict_ncct["l1"],
                "kl_ncct": loss_dict_ncct["kl"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "gnorm": grad_norm,
            }
            # GAN metrics
            if g_loss_dict:
                metrics["perceptual"] = g_loss_dict.get("perceptual", torch.tensor(0.0)).item()
                metrics["gan_g"] = g_loss_dict.get("gan_g", torch.tensor(0.0)).item()
                metrics["adaptive_w"] = g_loss_dict.get("adaptive_w", torch.tensor(0.0)).item()
            if d_loss_dict:
                metrics["d_loss"] = d_loss_dict["d_loss"].item()
                metrics["d_real"] = d_loss_dict["d_real"].item()
                metrics["d_fake"] = d_loss_dict["d_fake"].item()
                metrics["lr_d"] = self.vae_gan_loss.d_lr
            tracker.update(metrics)

            if self.global_step % tcfg.log_interval == 0:
                pbar.set_postfix(**{k: f"{v:.4f}" for k, v in tracker.result().items()
                                    if k not in ("lr", "gnorm", "lr_d")})

            # Collect visualization samples (CTA reconstruction)
            if len(train_vis_samples) < 8:
                n_need = 8 - sum(s["input"].shape[0] for s in train_vis_samples)
                if n_need > 0:
                    n_take = min(n_need, cta.shape[0])
                    train_vis_samples.append({
                        "input": cta[:n_take].detach(),
                        "recon": recon_cta[:n_take].detach(),
                        "target": cta[:n_take].detach(),
                    })

        self._train_vis_samples = train_vis_samples
        return tracker

    def _compute_loss(self, target, recon, posterior):
        """Compute reconstruction + KL loss."""
        l1_loss = torch.nn.functional.l1_loss(recon, target)
        kl_loss = posterior.kl().mean()

        total = self.l1_weight * l1_loss + self.kl_weight * kl_loss
        loss_dict = {
            "l1": l1_loss.item(),
            "kl": kl_loss.item(),
        }
        return total, loss_dict

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch: int):
        """Run validation (no augmentation)."""
        self.vae.eval()
        tracker = MetricTracker()
        val_vis_samples: List[Dict[str, torch.Tensor]] = []

        for batch in tqdm(self.val_loader, desc="VAE Validating", leave=False):
            ncct_3c = batch["ncct"].to(self.device)
            cta_3c = batch["cta"].to(self.device)
            mask_3c = batch["ncct_lung"].to(self.device)

            ncct, cta, mask = self.augmentor(ncct_3c, cta_3c, mask_3c, training=False)

            # Validate on CTA reconstruction
            recon_cta, posterior_cta = self.vae(cta)
            loss_cta, loss_dict_cta = self._compute_loss(cta, recon_cta, posterior_cta)

            # Validate on NCCT reconstruction
            recon_ncct, posterior_ncct = self.vae(ncct)
            loss_ncct, loss_dict_ncct = self._compute_loss(ncct, recon_ncct, posterior_ncct)

            loss = 0.5 * (loss_cta + loss_ncct)

            tracker.update({
                "loss": loss.item(),
                "l1_cta": loss_dict_cta["l1"],
                "kl_cta": loss_dict_cta["kl"],
                "l1_ncct": loss_dict_ncct["l1"],
                "kl_ncct": loss_dict_ncct["kl"],
            })

            if len(val_vis_samples) < 8:
                n_need = 8 - sum(s["input"].shape[0] for s in val_vis_samples)
                if n_need > 0:
                    n_take = min(n_need, cta.shape[0])
                    val_vis_samples.append({
                        "input": cta[:n_take].detach(),
                        "recon": recon_cta[:n_take].detach(),
                        "target": cta[:n_take].detach(),
                    })

        return tracker, val_vis_samples

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _save_visualizations(self, epoch: int, val_vis):
        """Save train and validation reconstruction grids."""
        epoch_tag = f"epoch{epoch+1:04d}"

        for tag, samples in [("train", getattr(self, '_train_vis_samples', [])),
                             ("val", val_vis)]:
            if not samples:
                continue
            inputs = torch.cat([s["input"] for s in samples], dim=0)[:8]
            recons = torch.cat([s["recon"] for s in samples], dim=0)[:8]
            targets = torch.cat([s["target"] for s in samples], dim=0)[:8]

            path = self.vis_dir / f"{tag}_{epoch_tag}.png"
            save_sample_grid(inputs, recons, targets, str(path), num_samples=8)

        logger.info(f"  Saved VAE visualizations for epoch {epoch+1}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool = False,
                         filename: Optional[str] = None):
        """Save training checkpoint."""
        state = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "vae_state_dict": self.vae.state_dict(),
            "ema_params": self.ema_params,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": {
                "vae": self.config.vae.__dict__,
                "vae_train": self.config.vae_train.__dict__,
            },
        }
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.vae_gan_loss is not None:
            state["vae_gan_loss"] = self.vae_gan_loss.state_dict()

        if filename is None:
            filename = f"checkpoint_epoch{epoch+1:04d}.pt"
        path = self.output_dir / filename
        torch.save(state, path)
        logger.info(f"  Saved VAE checkpoint: {path}")

        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(state, best_path)

    def _load_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming VAE from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        self.vae.load_state_dict(state["vae_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.start_epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        if "ema_params" in state:
            self.ema_params = state["ema_params"]
        if self.scheduler and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if self.vae_gan_loss is not None and "vae_gan_loss" in state:
            self.vae_gan_loss.load_state_dict_from_checkpoint(state["vae_gan_loss"])
        logger.info(f"  Resumed at epoch {self.start_epoch}, step {self.global_step}")

    def load_ema_weights(self):
        """Swap VAE parameters with EMA parameters for inference."""
        for p, ema_p in zip(self.vae.parameters(), self.ema_params):
            p.data.copy_(ema_p.data)
        logger.info("Loaded EMA weights into VAE")
