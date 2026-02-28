"""
Diffusion UNet Trainer for Latent Diffusion Model — Stage 2.

Trains the conditional diffusion UNet in the frozen VAE's latent space:
  1. Encode NCCT and CTA to latent space using frozen VAE
  2. Sample random timestep t and noise ε
  3. Add noise to CTA latent: z_noisy = sqrt(ᾱ_t) * z_cta + sqrt(1-ᾱ_t) * ε
  4. UNet predicts noise: ε_θ(concat(z_noisy, z_ncct), t)
  5. Loss = MSE(ε_θ, ε)

Features:
  - Frozen VAE (loaded from Stage 1 checkpoint)
  - Simple MSE noise prediction loss
  - EMA of UNet weights
  - Cosine LR schedule with warmup
  - Gradient clipping
  - Periodic DDIM sampling for visual quality monitoring
  - Checkpoint saving/resuming

Usage:
    python -m ldm.train_diffusion --config configs/ldm_default.yaml \\
        --pretrained_vae outputs_vae/checkpoint_best.pt
"""

import logging
import math
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm import tqdm

from ldm.config import LDMConfig
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.unet import DiffusionUNet
from ldm.diffusion.scheduler import DDPMScheduler
from ldm.diffusion.pipeline import ConditionalLDMPipeline
from models.nn_utils import update_ema, load_pretrained_weights
from utils.visualization import save_sample_grid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class MetricTracker:
    """Track running averages of metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = {}
        self._count = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self._sum[k] = self._sum.get(k, 0.0) + v * n
            self._count[k] = self._count.get(k, 0) + n

    def result(self) -> Dict[str, float]:
        return {k: self._sum[k] / self._count[k] for k in self._sum}

    def __str__(self):
        return ", ".join(f"{k}: {v:.6f}" for k, v in self.result().items())


def _warmup_cosine_schedule(warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# ---------------------------------------------------------------------------
# Diffusion Trainer
# ---------------------------------------------------------------------------

class DiffusionTrainer:
    """
    Training manager for the conditional diffusion UNet (LDM Stage 2).

    Requires a pretrained, frozen VAE from Stage 1. The VAE is used only
    for encoding inputs to latent space and decoding predictions for
    visualization.

    Training flow per step:
      1. Augment batch → extract middle C slices
      2. Encode NCCT and CTA with frozen VAE → z_ncct, z_cta
      3. Sample noise ε ~ N(0, I) and timestep t ~ Uniform(0, T)
      4. Compute z_noisy = scheduler.add_noise(z_cta, ε, t)
      5. UNet forward: ε_θ = unet(concat(z_noisy, z_ncct), t)
      6. Loss = MSE(ε_θ, ε)
      7. Backward + grad clip + optimizer step + EMA
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: DiffusionUNet,
        scheduler: DDPMScheduler,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        augmentor,
        config: LDMConfig,
        device: torch.device,
    ):
        self.device = device
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.augmentor = augmentor

        # Frozen VAE — no gradients, eval mode permanently
        self.vae = vae.to(device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # Diffusion UNet — trainable
        self.unet = unet.to(device)

        # Noise scheduler
        self.scheduler = scheduler.to(device)

        tcfg = config.diffusion_train

        # Optimizer
        self.optimizer = AdamW(
            self.unet.parameters(),
            lr=tcfg.lr,
            weight_decay=tcfg.weight_decay,
        )

        # LR scheduler
        steps_per_epoch = len(train_loader)
        total_steps = tcfg.num_epochs * steps_per_epoch
        self.lr_scheduler = self._build_scheduler(
            self.optimizer, tcfg.lr_scheduler, tcfg.warmup_steps, total_steps,
        )

        # EMA
        self.ema_rate = tcfg.ema_rate
        self.ema_params = [p.clone().detach() for p in self.unet.parameters()]

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

        # Sampling pipeline (for visualization — uses EMA weights during eval)
        self._pipeline = None  # built lazily to avoid circular init

        # Log
        n_params = sum(p.numel() for p in self.unet.parameters()) / 1e6
        logger.info(f"DiffusionTrainer initialized:")
        logger.info(f"  UNet params:      {n_params:.2f}M")
        logger.info(f"  Scheduler:        {config.scheduler.beta_schedule}, T={config.scheduler.num_train_timesteps}")
        logger.info(f"  LR:               {tcfg.lr}")
        logger.info(f"  Grad clip norm:   {self.grad_clip_norm}")
        logger.info(f"  EMA rate:         {self.ema_rate}")

    @staticmethod
    def _build_scheduler(optimizer, scheduler_type, warmup_steps, total_steps):
        if scheduler_type == "cosine":
            return LambdaLR(optimizer, _warmup_cosine_schedule(warmup_steps, total_steps))
        elif scheduler_type == "step":
            return StepLR(optimizer, step_size=50, gamma=0.5)
        return None

    # ------------------------------------------------------------------
    # Encoding helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space using frozen VAE (deterministic mode)."""
        return self.vae.encode(x).mode()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Run the full diffusion training loop."""
        tcfg = self.config.diffusion_train
        logger.info(f"Starting diffusion training for {tcfg.num_epochs} epochs")

        for epoch in range(self.start_epoch, tcfg.num_epochs):
            train_metrics = self._train_epoch(epoch)
            logger.info(f"Epoch {epoch+1}/{tcfg.num_epochs} [Train] {train_metrics}")

            # Validation
            if self.val_loader and (epoch + 1) % tcfg.val_interval == 0:
                val_metrics = self._validate(epoch)
                logger.info(f"Epoch {epoch+1}/{tcfg.num_epochs} [Val]   {val_metrics}")
                val_loss = val_metrics.result()["loss"]
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f"  New best val loss: {val_loss:.6f}")

                # Generate sample visualizations (every val_interval)
                self._generate_samples(epoch)

            if (epoch + 1) % tcfg.save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            if self.lr_scheduler and tcfg.lr_scheduler == "step":
                self.lr_scheduler.step()

        self._save_checkpoint(tcfg.num_epochs - 1, is_best=False,
                              filename="checkpoint_final.pt")
        logger.info("Diffusion training complete.")

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> MetricTracker:
        self.unet.train()
        tracker = MetricTracker()
        tcfg = self.config.diffusion_train
        T = self.config.scheduler.num_train_timesteps

        pbar = tqdm(self.train_loader, desc=f"Diff Epoch {epoch+1}", leave=False)
        for batch in pbar:
            ncct_3c = batch["ncct"].to(self.device)
            cta_3c = batch["cta"].to(self.device)
            mask_3c = batch["ncct_lung"].to(self.device)

            # Augment and extract middle C slices
            ncct, cta, _ = self.augmentor(ncct_3c, cta_3c, mask_3c, training=True)

            # Encode to latent space (frozen VAE, no grad)
            z_ncct = self._encode_to_latent(ncct)
            z_cta = self._encode_to_latent(cta)

            # Sample noise and timesteps
            noise = torch.randn_like(z_cta)
            t = torch.randint(0, T, (z_cta.shape[0],), device=self.device, dtype=torch.long)

            # Forward diffusion: add noise to CTA latent
            z_noisy = self.scheduler.add_noise(z_cta, noise, t)

            # UNet predicts noise from concat(z_noisy, z_ncct)
            model_input = torch.cat([z_noisy, z_ncct], dim=1)
            noise_pred = self.unet(model_input, t)

            # Simple MSE loss on noise prediction
            loss = F.mse_loss(noise_pred, noise)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            grad_norm = 0.0
            if self.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.unet.parameters(), self.grad_clip_norm
                ).item()

            self.optimizer.step()

            if self.lr_scheduler and tcfg.lr_scheduler == "cosine":
                self.lr_scheduler.step()

            # EMA
            update_ema(self.ema_params, self.unet.parameters(), rate=self.ema_rate)

            # Metrics
            self.global_step += 1
            metrics = {
                "loss": loss.item(),
                "lr": self.optimizer.param_groups[0]["lr"],
                "gnorm": grad_norm,
            }
            tracker.update(metrics)

            if self.global_step % tcfg.log_interval == 0:
                pbar.set_postfix(loss=f"{tracker.result()['loss']:.4f}")

        return tracker

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch: int) -> MetricTracker:
        """Validate noise prediction MSE on the validation set."""
        self.unet.eval()
        tracker = MetricTracker()
        T = self.config.scheduler.num_train_timesteps

        for batch in tqdm(self.val_loader, desc="Diff Validating", leave=False):
            ncct_3c = batch["ncct"].to(self.device)
            cta_3c = batch["cta"].to(self.device)
            mask_3c = batch["ncct_lung"].to(self.device)

            ncct, cta, _ = self.augmentor(ncct_3c, cta_3c, mask_3c, training=False)

            z_ncct = self._encode_to_latent(ncct)
            z_cta = self._encode_to_latent(cta)

            noise = torch.randn_like(z_cta)
            t = torch.randint(0, T, (z_cta.shape[0],), device=self.device, dtype=torch.long)
            z_noisy = self.scheduler.add_noise(z_cta, noise, t)

            model_input = torch.cat([z_noisy, z_ncct], dim=1)
            noise_pred = self.unet(model_input, t)

            loss = F.mse_loss(noise_pred, noise)
            tracker.update({"loss": loss.item()})

        return tracker

    # ------------------------------------------------------------------
    # Sample generation (for visualization)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_samples(self, epoch: int, num_samples: int = 4):
        """Generate DDIM samples from validation set for visual monitoring."""
        if self.val_loader is None:
            return

        # Swap to EMA weights for sampling
        orig_params = [p.clone() for p in self.unet.parameters()]
        for p, ema_p in zip(self.unet.parameters(), self.ema_params):
            p.data.copy_(ema_p.data)
        self.unet.eval()

        # Build pipeline if needed
        if self._pipeline is None:
            self._pipeline = ConditionalLDMPipeline(
                self.vae, self.unet, self.scheduler,
            )

        # Get a validation batch
        batch = next(iter(self.val_loader))
        ncct_3c = batch["ncct"][:num_samples].to(self.device)
        cta_3c = batch["cta"][:num_samples].to(self.device)
        mask_3c = batch["ncct_lung"][:num_samples].to(self.device)

        ncct, cta, _ = self.augmentor(ncct_3c, cta_3c, mask_3c, training=False)

        # Generate
        ddim_steps = min(20, self.config.scheduler.num_inference_steps)
        cta_pred = self._pipeline.sample(
            ncct, num_inference_steps=ddim_steps, verbose=False,
        )

        # Save visualization
        epoch_tag = f"epoch{epoch+1:04d}"
        path = self.vis_dir / f"samples_{epoch_tag}.png"
        save_sample_grid(ncct, cta_pred, cta, str(path), num_samples=num_samples)
        logger.info(f"  Saved diffusion samples for epoch {epoch+1}")

        # Restore original weights
        for p, orig_p in zip(self.unet.parameters(), orig_params):
            p.data.copy_(orig_p.data)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool = False,
                         filename: Optional[str] = None):
        state = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "unet_state_dict": self.unet.state_dict(),
            "ema_params": self.ema_params,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": {
                "unet": self.config.unet.__dict__,
                "scheduler": self.config.scheduler.__dict__,
                "diffusion_train": self.config.diffusion_train.__dict__,
            },
        }
        if self.lr_scheduler:
            state["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if filename is None:
            filename = f"checkpoint_epoch{epoch+1:04d}.pt"
        path = self.output_dir / filename
        torch.save(state, path)
        logger.info(f"  Saved diffusion checkpoint: {path}")

        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(state, best_path)

    def _load_checkpoint(self, checkpoint_path: str):
        logger.info(f"Resuming diffusion from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        self.unet.load_state_dict(state["unet_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.start_epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        if "ema_params" in state:
            self.ema_params = state["ema_params"]
        if self.lr_scheduler and "scheduler_state_dict" in state:
            self.lr_scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info(f"  Resumed at epoch {self.start_epoch}, step {self.global_step}")

    def load_ema_weights(self):
        """Swap UNet parameters with EMA parameters for inference."""
        for p, ema_p in zip(self.unet.parameters(), self.ema_params):
            p.data.copy_(ema_p.data)
        logger.info("Loaded EMA weights into UNet")
