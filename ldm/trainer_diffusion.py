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
from ldm.utils import MetricTracker, warmup_cosine_schedule as _warmup_cosine_schedule
from models.nn_utils import update_ema, load_pretrained_weights
from utils.visualization import save_sample_grid

logger = logging.getLogger(__name__)


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

        # Latent scaling factor (Stable Diffusion style: z_scaled = z * factor)
        # Normalizes latent std to ~1.0 for proper noise schedule matching.
        self.latent_scale_factor = config.scheduler.latent_scale_factor

        # Min-SNR-γ loss weighting (0 = disabled)
        self.snr_gamma = config.scheduler.snr_gamma

        # Prediction type for loss target selection
        self.prediction_type = config.scheduler.prediction_type

        # Classifier-Free Guidance: condition drop rate during training
        self.cfg_drop_rate = config.diffusion_train.cfg_drop_rate

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
        logger.info(f"  Latent scale:     {self.latent_scale_factor:.6f}")
        logger.info(f"  Prediction type:  {self.prediction_type}")
        logger.info(f"  Min-SNR γ:        {self.snr_gamma} ({'enabled' if self.snr_gamma > 0 else 'disabled'})")
        logger.info(f"  CFG drop rate:    {self.cfg_drop_rate} ({'enabled' if self.cfg_drop_rate > 0 else 'disabled'})")
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
        """Encode images to latent space using frozen VAE (deterministic mode).

        Applies latent_scale_factor to normalize variance for diffusion.
        """
        z = self.vae.encode(x).mode()
        return z * self.latent_scale_factor

    @torch.no_grad()
    def compute_latent_scale_factor(
        self, num_batches: int = 50,
    ) -> float:
        """Auto-compute latent scaling factor from training data.

        Computes scale_factor = 1 / std(z) over a subset of training data
        so that scaled latents have approximately unit variance, matching
        the assumption of the diffusion noise schedule.

        This follows the Stable Diffusion approach (scaling_factor ≈ 0.18215).
        """
        self.vae.eval()
        latent_values = []

        for i, batch in enumerate(self.train_loader):
            if i >= num_batches:
                break
            ncct_3c = batch["ncct"].to(self.device)
            cta_3c = batch["cta"].to(self.device)
            mask_3c = batch["ncct_lung"].to(self.device)

            _, cta, _ = self.augmentor(ncct_3c, cta_3c, mask_3c, training=False)
            z = self.vae.encode(cta).mode()
            latent_values.append(z.flatten().cpu())

        all_latents = torch.cat(latent_values)
        std = all_latents.std().item()
        scale_factor = 1.0 / std

        logger.info(f"  Auto-computed latent scale factor:")
        logger.info(f"    Latent std:     {std:.4f}")
        logger.info(f"    Latent mean:    {all_latents.mean().item():.4f}")
        logger.info(f"    Scale factor:   {scale_factor:.6f}")
        logger.info(f"    Scaled std:     {std * scale_factor:.4f} (target ≈ 1.0)")

        return scale_factor

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

            # Classifier-Free Guidance: randomly drop condition
            if self.cfg_drop_rate > 0:
                drop_mask = (torch.rand(z_ncct.shape[0], 1, 1, 1,
                                        device=self.device) < self.cfg_drop_rate)
                z_ncct = z_ncct * (~drop_mask).float()

            # Sample noise and timesteps
            noise = torch.randn_like(z_cta)
            t = torch.randint(0, T, (z_cta.shape[0],), device=self.device, dtype=torch.long)

            # Forward diffusion: add noise to CTA latent
            z_noisy = self.scheduler.add_noise(z_cta, noise, t)

            # UNet prediction from concat(z_noisy, z_ncct)
            model_input = torch.cat([z_noisy, z_ncct], dim=1)
            model_output = self.unet(model_input, t)

            # Select loss target based on prediction_type
            if self.prediction_type == "v_prediction":
                target = self.scheduler.get_v_target(z_cta, noise, t)
            else:  # epsilon
                target = noise

            # Per-sample MSE loss (before SNR weighting)
            loss_per_sample = F.mse_loss(
                model_output, target, reduction="none",
            ).mean(dim=[1, 2, 3])  # (B,)

            # Min-SNR-γ weighting
            if self.snr_gamma > 0:
                snr_weights = self.scheduler.get_min_snr_weight(t, gamma=self.snr_gamma)
                loss = (loss_per_sample * snr_weights).mean()
            else:
                loss = loss_per_sample.mean()

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

            # Per-timestep-range loss tracking for diagnostics
            with torch.no_grad():
                t_cpu = t.cpu()
                T = self.config.scheduler.num_train_timesteps
                third = T // 3
                low_mask = t_cpu < third
                mid_mask = (t_cpu >= third) & (t_cpu < 2 * third)
                high_mask = t_cpu >= 2 * third
                loss_raw = loss_per_sample.detach().cpu()

            # Metrics
            self.global_step += 1
            metrics = {
                "loss": loss.item(),
                "lr": self.optimizer.param_groups[0]["lr"],
                "gnorm": grad_norm,
            }
            # Per-range losses (only update when samples exist in range)
            if low_mask.any():
                metrics["loss_low_t"] = loss_raw[low_mask].mean().item()
            if mid_mask.any():
                metrics["loss_mid_t"] = loss_raw[mid_mask].mean().item()
            if high_mask.any():
                metrics["loss_high_t"] = loss_raw[high_mask].mean().item()
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
            model_output = self.unet(model_input, t)

            # Same target selection as training
            if self.prediction_type == "v_prediction":
                target = self.scheduler.get_v_target(z_cta, noise, t)
            else:
                target = noise

            loss_per_sample = F.mse_loss(
                model_output, target, reduction="none",
            ).mean(dim=[1, 2, 3])

            if self.snr_gamma > 0:
                snr_weights = self.scheduler.get_min_snr_weight(t, gamma=self.snr_gamma)
                loss = (loss_per_sample * snr_weights).mean()
            else:
                loss = loss_per_sample.mean()

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
        # NOTE: Visualization uses cfg_scale=1.0 (no guidance) to show true
        # model quality. CFG is an inference-time adjustment that should only
        # be applied during final inference, not during training monitoring.
        if self._pipeline is None:
            self._pipeline = ConditionalLDMPipeline(
                self.vae, self.unet, self.scheduler,
                latent_scale_factor=self.latent_scale_factor,
                cfg_scale=1.0,
                dynamic_threshold_percentile=0.0,
            )

        # Get a validation batch
        batch = next(iter(self.val_loader))
        ncct_3c = batch["ncct"][:num_samples].to(self.device)
        cta_3c = batch["cta"][:num_samples].to(self.device)
        mask_3c = batch["ncct_lung"][:num_samples].to(self.device)

        ncct, cta, _ = self.augmentor(ncct_3c, cta_3c, mask_3c, training=False)

        # Generate (use full inference steps for accurate quality assessment)
        ddim_steps = self.config.scheduler.num_inference_steps
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
            "latent_scale_factor": self.latent_scale_factor,
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
        if "latent_scale_factor" in state:
            self.latent_scale_factor = state["latent_scale_factor"]
            logger.info(f"  Restored latent_scale_factor: {self.latent_scale_factor:.6f}")
        logger.info(f"  Resumed at epoch {self.start_epoch}, step {self.global_step}")

    def load_ema_weights(self):
        """Swap UNet parameters with EMA parameters for inference."""
        for p, ema_p in zip(self.unet.parameters(), self.ema_params):
            p.data.copy_(ema_p.data)
        logger.info("Loaded EMA weights into UNet")
