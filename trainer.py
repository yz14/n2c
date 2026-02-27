"""
Training loop for NCCT→CTA image translation.

Features:
  - GPU-based 3D augmentation via GPUAugmentor
  - Generator (G) + optional Registration (R) + optional Discriminator (D)
  - Mask-weighted reconstruction loss (lung 10x)
  - GAN adversarial + feature matching loss
  - Registration smoothness loss
  - Separate optimizers for G+R and D
  - EMA of generator weights
  - Learning rate scheduling (cosine with warmup)
  - Checkpoint saving/resuming
  - Visualization grids after each validation
"""

import logging
import math
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm import tqdm

from models.nn_utils import update_ema
from models.registration import RegistrationNet
from models.discriminator import MultiscaleDiscriminator
from models.losses import (
    CombinedLoss, GANLoss, FeatureMatchingLoss, GradLoss,
)
from data.transforms import GPUAugmentor
from utils.visualization import save_sample_grid

logger = logging.getLogger(__name__)


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
    """Create a warmup + cosine decay LR lambda."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


class Trainer:
    """
    Training manager for the UNet + Registration + Discriminator pipeline.

    Training flow per step:
      1. G(ncct) → pred_cta
      2. R(pred_cta, cta) → warped_pred, displacement  (if R enabled)
      3. Reconstruction loss on (warped_pred, cta) with mask weighting
      4. Smoothness loss on displacement field  (if R enabled)
      5. D(ncct, pred_cta) vs D(ncct, cta) → GAN + feature matching loss  (if D enabled)
      6. Update G+R with combined loss, then update D with adversarial loss
    """

    def __init__(
        self,
        model: nn.Module,
        reg_net: Optional[RegistrationNet],
        discriminator: Optional[MultiscaleDiscriminator],
        criterion: CombinedLoss,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        augmentor: GPUAugmentor,
        config,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.augmentor = augmentor
        self.config = config
        self.device = device

        # --- Registration network ---
        self.use_reg = config.registration.enabled and reg_net is not None
        if self.use_reg:
            self.reg_net = reg_net.to(device)
            self.grad_loss = GradLoss(penalty=config.registration.smoothness_penalty).to(device)
            self.smooth_weight = config.registration.smoothness_weight
        else:
            self.reg_net = None
            self.grad_loss = None
            self.smooth_weight = 0.0

        # --- Discriminator ---
        self.use_disc = config.discriminator.enabled and discriminator is not None
        if self.use_disc:
            self.discriminator = discriminator.to(device)
            self.gan_loss_fn = GANLoss().to(device)
            self.feat_match_fn = FeatureMatchingLoss().to(device)
            self.gan_weight = config.discriminator.gan_weight
            self.feat_match_weight = config.discriminator.feat_match_weight
        else:
            self.discriminator = None

        # --- Optimizer for G (+ R if enabled) ---
        g_params = list(self.model.parameters())
        if self.use_reg:
            g_params += list(self.reg_net.parameters())
        self.optimizer_G = AdamW(
            g_params,
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )

        # --- Optimizer for D ---
        if self.use_disc:
            self.optimizer_D = AdamW(
                self.discriminator.parameters(),
                lr=config.discriminator.lr,
                weight_decay=config.train.weight_decay,
            )
        else:
            self.optimizer_D = None

        # --- LR scheduler (for G) ---
        total_steps = config.train.num_epochs * len(train_loader)
        if config.train.lr_scheduler == "cosine":
            lr_lambda = _warmup_cosine_schedule(config.train.warmup_steps, total_steps)
            self.scheduler_G = LambdaLR(self.optimizer_G, lr_lambda)
        elif config.train.lr_scheduler == "step":
            self.scheduler_G = StepLR(self.optimizer_G, step_size=50, gamma=0.5)
        else:
            self.scheduler_G = None

        # EMA (generator only)
        self.ema_rate = config.train.ema_rate
        self.ema_params = [p.clone().detach() for p in self.model.parameters()]

        # State
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        # Output directory
        self.output_dir = Path(config.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Buffers for visualization
        self._train_vis_samples: List[Dict[str, torch.Tensor]] = []

        # Resume if specified
        if config.train.resume_checkpoint:
            self._load_checkpoint(config.train.resume_checkpoint)

        # Log configuration
        logger.info(f"  Registration: {'ON' if self.use_reg else 'OFF'}")
        logger.info(f"  Discriminator: {'ON' if self.use_disc else 'OFF'}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Run the full training loop."""
        cfg = self.config.train
        logger.info(f"Starting training for {cfg.num_epochs} epochs")
        logger.info(f"  Batch size:    {cfg.batch_size}")
        logger.info(f"  LR (G):        {cfg.lr}")
        logger.info(f"  Output dir:    {self.output_dir}")

        for epoch in range(self.start_epoch, cfg.num_epochs):
            train_metrics = self._train_epoch(epoch)
            logger.info(
                f"Epoch {epoch+1}/{cfg.num_epochs} [Train] {train_metrics}"
            )

            # Validation
            if self.val_loader and (epoch + 1) % cfg.val_interval == 0:
                val_metrics, val_vis = self._validate(epoch)
                logger.info(
                    f"Epoch {epoch+1}/{cfg.num_epochs} [Val]   {val_metrics}"
                )
                val_loss = val_metrics.result()["loss"]
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f"  New best validation loss: {val_loss:.6f}")

                self._save_visualizations(epoch, val_vis)

            if (epoch + 1) % cfg.save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            if self.scheduler_G and cfg.lr_scheduler == "step":
                self.scheduler_G.step()

        self._save_checkpoint(cfg.num_epochs - 1, is_best=False, filename="checkpoint_final.pt")
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> MetricTracker:
        """Run one training epoch."""
        self.model.train()
        if self.use_reg:
            self.reg_net.train()
        if self.use_disc:
            self.discriminator.train()

        tracker = MetricTracker()
        cfg = self.config.train
        self._train_vis_samples = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            ncct_3c = batch["ncct"].to(self.device)
            cta_3c = batch["cta"].to(self.device)
            mask_3c = batch["ncct_lung"].to(self.device)

            ncct, cta, mask = self.augmentor(ncct_3c, cta_3c, mask_3c, training=True)

            # ============================================================
            # Step 1: Update Generator (G) + Registration (R)
            # ============================================================
            pred = self.model(ncct)

            # Registration (if enabled)
            if self.use_reg:
                reg_out = self.reg_net(pred, cta)
                recon_target = reg_out["warped"]  # warped pred for loss
                displacement = reg_out["displacement"]
            else:
                recon_target = pred
                displacement = None

            # Reconstruction loss (mask-weighted L1 + SSIM)
            loss_dict = self.criterion(recon_target, cta, mask)
            g_loss = loss_dict["loss"]

            # Registration smoothness loss
            smooth_loss_val = 0.0
            if self.use_reg and displacement is not None:
                smooth_loss = self.grad_loss(displacement)
                g_loss = g_loss + self.smooth_weight * smooth_loss
                smooth_loss_val = smooth_loss.item()

            # GAN loss (generator side)
            gan_g_val = 0.0
            feat_match_val = 0.0
            if self.use_disc:
                # Conditional input: concat(ncct, pred_cta)
                fake_input = torch.cat([ncct, pred], dim=1)
                real_input = torch.cat([ncct, cta], dim=1)

                fake_features = self.discriminator(fake_input)
                with torch.no_grad():
                    real_features = self.discriminator(real_input)

                # Generator wants D to classify fake as real
                gan_g = self.gan_loss_fn(fake_features, target_is_real=True)
                feat_match = self.feat_match_fn(real_features, fake_features)
                g_loss = g_loss + self.gan_weight * gan_g + self.feat_match_weight * feat_match
                gan_g_val = gan_g.item()
                feat_match_val = feat_match.item()

            # Update G (+R)
            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()

            # ============================================================
            # Step 2: Update Discriminator (D)
            # ============================================================
            d_loss_val = 0.0
            if self.use_disc:
                fake_input = torch.cat([ncct, pred.detach()], dim=1)
                real_input = torch.cat([ncct, cta], dim=1)

                fake_preds = self.discriminator(fake_input)
                real_preds = self.discriminator(real_input)

                d_loss_real = self.gan_loss_fn(real_preds, target_is_real=True)
                d_loss_fake = self.gan_loss_fn(fake_preds, target_is_real=False)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

                self.optimizer_D.zero_grad()
                d_loss.backward()
                self.optimizer_D.step()
                d_loss_val = d_loss.item()

            # LR scheduler (step-level for cosine warmup)
            if self.scheduler_G and cfg.lr_scheduler == "cosine":
                self.scheduler_G.step()

            # EMA update (generator only)
            update_ema(self.ema_params, self.model.parameters(), rate=self.ema_rate)

            # Collect visualization samples
            if len(self._train_vis_samples) < 8:
                n_need = 8 - len(self._train_vis_samples)
                n_take = min(n_need, ncct.shape[0])
                self._train_vis_samples.append({
                    "ncct": ncct[:n_take].detach(),
                    "pred": pred[:n_take].detach(),
                    "cta": cta[:n_take].detach(),
                })

            # Logging
            self.global_step += 1
            metrics = {
                "loss": loss_dict["loss"].item(),
                "l1": loss_dict["l1"].item(),
                "ssim": loss_dict["ssim"].item(),
                "lr": self.optimizer_G.param_groups[0]["lr"],
            }
            if self.use_reg:
                metrics["smooth"] = smooth_loss_val
            if self.use_disc:
                metrics["gan_g"] = gan_g_val
                metrics["fm"] = feat_match_val
                metrics["d_loss"] = d_loss_val
            tracker.update(metrics)

            if self.global_step % cfg.log_interval == 0:
                pbar.set_postfix(**{k: f"{v:.4f}" for k, v in tracker.result().items()})

        return tracker

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch: int):
        """Run validation (no D, no augmentation)."""
        self.model.eval()
        if self.use_reg:
            self.reg_net.eval()
        tracker = MetricTracker()
        val_vis_samples: List[Dict[str, torch.Tensor]] = []

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            ncct_3c = batch["ncct"].to(self.device)
            cta_3c = batch["cta"].to(self.device)
            mask_3c = batch["ncct_lung"].to(self.device)

            ncct, cta, mask = self.augmentor(ncct_3c, cta_3c, mask_3c, training=False)
            pred = self.model(ncct)

            # Validation reconstruction loss (no registration — evaluate raw output)
            loss_dict = self.criterion(pred, cta, mask)

            tracker.update({
                "loss": loss_dict["loss"].item(),
                "l1": loss_dict["l1"].item(),
                "ssim": loss_dict["ssim"].item(),
            })

            if len(val_vis_samples) < 8:
                n_need = 8 - sum(s["ncct"].shape[0] for s in val_vis_samples)
                if n_need > 0:
                    n_take = min(n_need, ncct.shape[0])
                    val_vis_samples.append({
                        "ncct": ncct[:n_take].detach(),
                        "pred": pred[:n_take].detach(),
                        "cta": cta[:n_take].detach(),
                    })

        return tracker, val_vis_samples

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _save_visualizations(
        self, epoch: int, val_vis: List[Dict[str, torch.Tensor]]
    ):
        """Save train and validation visualization grids."""
        epoch_tag = f"epoch{epoch+1:04d}"

        for tag, samples in [("train", self._train_vis_samples), ("val", val_vis)]:
            if not samples:
                continue
            inputs = torch.cat([s["ncct"] for s in samples], dim=0)[:8]
            preds = torch.cat([s["pred"] for s in samples], dim=0)[:8]
            targets = torch.cat([s["cta"] for s in samples], dim=0)[:8]

            path = self.vis_dir / f"{tag}_{epoch_tag}.png"
            save_sample_grid(inputs, preds, targets, str(path), num_samples=8)

        logger.info(f"  Saved visualization grids for epoch {epoch+1}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool = False,
                         filename: Optional[str] = None):
        """Save training checkpoint."""
        state = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "ema_params": self.ema_params,
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler_G:
            state["scheduler_G_state_dict"] = self.scheduler_G.state_dict()
        if self.use_reg and self.reg_net is not None:
            state["reg_net_state_dict"] = self.reg_net.state_dict()
        if self.use_disc and self.discriminator is not None:
            state["discriminator_state_dict"] = self.discriminator.state_dict()
            state["optimizer_D_state_dict"] = self.optimizer_D.state_dict()

        if filename is None:
            filename = f"checkpoint_epoch{epoch+1:04d}.pt"
        path = self.output_dir / filename
        torch.save(state, path)
        logger.info(f"  Saved checkpoint: {path}")

        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(state, best_path)

    def _load_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])

        # Load G optimizer (handle renamed key for backward compatibility)
        if "optimizer_G_state_dict" in state:
            self.optimizer_G.load_state_dict(state["optimizer_G_state_dict"])
        elif "optimizer_state_dict" in state:
            self.optimizer_G.load_state_dict(state["optimizer_state_dict"])

        self.start_epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        if "ema_params" in state:
            self.ema_params = state["ema_params"]
        if self.scheduler_G and "scheduler_G_state_dict" in state:
            self.scheduler_G.load_state_dict(state["scheduler_G_state_dict"])
        elif self.scheduler_G and "scheduler_state_dict" in state:
            self.scheduler_G.load_state_dict(state["scheduler_state_dict"])

        # Load R and D if present
        if self.use_reg and "reg_net_state_dict" in state:
            self.reg_net.load_state_dict(state["reg_net_state_dict"])
        if self.use_disc and "discriminator_state_dict" in state:
            self.discriminator.load_state_dict(state["discriminator_state_dict"])
            if "optimizer_D_state_dict" in state:
                self.optimizer_D.load_state_dict(state["optimizer_D_state_dict"])

        logger.info(f"  Resumed at epoch {self.start_epoch}, step {self.global_step}")
