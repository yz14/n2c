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
  - Learning rate scheduling (cosine with warmup) for both G and D
  - Pretrained weight loading for G, R, D independently
  - Checkpoint saving/resuming
  - Visualization grids after each validation
  - Gradient clipping (configurable max norm)
  - Gradient accumulation (configurable steps)
  - One-sided label smoothing for discriminator stability
  - Gradient norm monitoring for debugging
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

from models.nn_utils import update_ema, load_pretrained_weights
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
            self.label_smoothing = config.discriminator.label_smoothing
        else:
            self.discriminator = None
            self.label_smoothing = 0.0

        # --- Training tricks config ---
        self.grad_clip_norm = config.train.grad_clip_norm
        self.grad_accumulation_steps = max(1, config.train.grad_accumulation_steps)

        # --- Load pretrained weights (before optimizer creation for correct param refs) ---
        self._load_pretrained_if_specified(config)

        # --- Optimizer for G (+ R if enabled) ---
        self._g_param_list = list(self.model.parameters())
        if self.use_reg:
            self._g_param_list += list(self.reg_net.parameters())
        self.optimizer_G = AdamW(
            self._g_param_list,
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
        # Effective steps account for gradient accumulation
        steps_per_epoch = len(train_loader) // self.grad_accumulation_steps
        total_optim_steps = config.train.num_epochs * steps_per_epoch
        self.scheduler_G = self._build_scheduler(
            self.optimizer_G,
            config.train.lr_scheduler,
            config.train.warmup_steps,
            total_optim_steps,
        )

        # --- LR scheduler (for D) ---
        if self.use_disc:
            self.scheduler_D = self._build_scheduler(
                self.optimizer_D,
                config.discriminator.lr_scheduler,
                config.discriminator.warmup_steps,
                total_optim_steps,
            )
        else:
            self.scheduler_D = None

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

        # Resume if specified (after optimizer/scheduler creation)
        if config.train.resume_checkpoint:
            self._load_checkpoint(config.train.resume_checkpoint)

        # Log configuration
        logger.info(f"  Registration:     {'ON' if self.use_reg else 'OFF'}")
        logger.info(f"  Discriminator:    {'ON' if self.use_disc else 'OFF'}")
        logger.info(f"  Grad clipping:    {self.grad_clip_norm if self.grad_clip_norm > 0 else 'OFF'}")
        logger.info(f"  Grad accumulation:{self.grad_accumulation_steps} steps")
        if self.use_disc:
            logger.info(f"  Label smoothing:  {self.label_smoothing}")
            logger.info(f"  D LR scheduler:   {config.discriminator.lr_scheduler}")

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_scheduler(optimizer, scheduler_type: str, warmup_steps: int,
                         total_steps: int):
        """Build LR scheduler for an optimizer."""
        if scheduler_type == "cosine":
            lr_lambda = _warmup_cosine_schedule(warmup_steps, total_steps)
            return LambdaLR(optimizer, lr_lambda)
        elif scheduler_type == "step":
            return StepLR(optimizer, step_size=50, gamma=0.5)
        return None

    def _load_pretrained_if_specified(self, config):
        """Load pretrained weights for G, R, D if paths are specified in config."""
        tcfg = config.train

        if tcfg.pretrained_G:
            load_pretrained_weights(
                self.model, tcfg.pretrained_G,
                component_key="model_state_dict", device=self.device,
            )

        if tcfg.pretrained_R and self.use_reg and self.reg_net is not None:
            load_pretrained_weights(
                self.reg_net, tcfg.pretrained_R,
                component_key="reg_net_state_dict", device=self.device,
            )

        if tcfg.pretrained_D and self.use_disc and self.discriminator is not None:
            load_pretrained_weights(
                self.discriminator, tcfg.pretrained_D,
                component_key="discriminator_state_dict", device=self.device,
            )

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

        # Diagnostic: log data statistics from first batch
        self._log_data_diagnostics()

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

                # Per-epoch diagnostics (debug output stats)
                self._log_epoch_diagnostics(epoch)

            if (epoch + 1) % cfg.save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Epoch-level scheduler stepping (for "step" type schedulers)
            if self.scheduler_G and cfg.lr_scheduler == "step":
                self.scheduler_G.step()
            if self.scheduler_D and self.config.discriminator.lr_scheduler == "step":
                self.scheduler_D.step()

        self._save_checkpoint(cfg.num_epochs - 1, is_best=False, filename="checkpoint_final.pt")
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_data_diagnostics(self):
        """Log data statistics from one batch to diagnose HU range issues."""
        logger.info("=" * 60)
        logger.info("[Diagnostics] Scanning first batch for data statistics...")
        try:
            batch = next(iter(self.train_loader))
            ncct_3c = batch["ncct"].to(self.device)
            cta_3c = batch["cta"].to(self.device)
            mask_3c = batch["ncct_lung"].to(self.device)

            # Raw normalized stats (should be in [-1, 1])
            logger.info(f"  [Raw 3C] ncct: min={ncct_3c.min():.4f}, max={ncct_3c.max():.4f}, "
                        f"mean={ncct_3c.mean():.4f}, std={ncct_3c.std():.4f}")
            logger.info(f"  [Raw 3C] cta:  min={cta_3c.min():.4f}, max={cta_3c.max():.4f}, "
                        f"mean={cta_3c.mean():.4f}, std={cta_3c.std():.4f}")
            logger.info(f"  [Raw 3C] mask: min={mask_3c.min():.4f}, max={mask_3c.max():.4f}, "
                        f"sum={mask_3c.sum():.0f}, frac={mask_3c.mean():.4f}")

            # Check how much data is clipped at boundaries
            ncct_at_min = (ncct_3c <= -0.99).float().mean().item()
            ncct_at_max = (ncct_3c >= 0.99).float().mean().item()
            cta_at_min = (cta_3c <= -0.99).float().mean().item()
            cta_at_max = (cta_3c >= 0.99).float().mean().item()
            logger.info(f"  [Clipping] ncct at min(-1): {ncct_at_min*100:.1f}%, at max(+1): {ncct_at_max*100:.1f}%")
            logger.info(f"  [Clipping] cta  at min(-1): {cta_at_min*100:.1f}%, at max(+1): {cta_at_max*100:.1f}%")

            if cta_at_max > 0.05:
                logger.warning(f"  ⚠ {cta_at_max*100:.1f}% of CTA voxels are clipped at max! "
                               f"HU range may be too narrow — contrast enhancement signal is lost.")
            if ncct_at_min > 0.20:
                logger.warning(f"  ⚠ {ncct_at_min*100:.1f}% of NCCT voxels are clipped at min! "
                               f"HU range may be too narrow — lung detail is lost.")

            # Difference between NCCT and CTA (should be large if HU range is correct)
            diff = (cta_3c - ncct_3c).abs()
            logger.info(f"  [NCCT-CTA diff] mean={diff.mean():.4f}, max={diff.max():.4f}, "
                        f"std={diff.std():.4f}")
            if diff.mean() < 0.02:
                logger.warning(f"  ⚠ NCCT and CTA are nearly identical (mean diff={diff.mean():.4f})! "
                               f"The model has very little signal to learn. Check HU range.")

            # After augmentation (extract middle slices)
            ncct, cta, mask = self.augmentor(ncct_3c, cta_3c, mask_3c, training=False)
            logger.info(f"  [Middle C] ncct shape={list(ncct.shape)}, "
                        f"cta shape={list(cta.shape)}, mask shape={list(mask.shape)}")

            # Run G to check output range
            self.model.eval()
            pred = self.model(ncct)
            residual = pred - ncct
            logger.info(f"  [G output] pred: min={pred.min():.4f}, max={pred.max():.4f}, "
                        f"mean={pred.mean():.4f}")
            logger.info(f"  [G residual] min={residual.min():.4f}, max={residual.max():.4f}, "
                        f"mean_abs={residual.abs().mean():.4f}")
            if residual.abs().mean() < 0.01:
                logger.warning(f"  ⚠ G residual is near zero ({residual.abs().mean():.4f}). "
                               f"Model is learning identity mapping (output ≈ input).")
            self.model.train()

            # HU range info
            hu_min = self.config.data.hu_min
            hu_max = self.config.data.hu_max
            logger.info(f"  [Config] HU range: [{hu_min}, {hu_max}] (width={hu_max - hu_min} HU)")
            if hu_max - hu_min < 1000:
                logger.warning(f"  ⚠ HU range width is only {hu_max - hu_min} HU. "
                               f"For NCCT→CTA with lung, recommend at least [-1024, 600] (1624 HU).")
        except Exception as e:
            logger.warning(f"  [Diagnostics] Failed: {e}")
        logger.info("=" * 60)

    @torch.no_grad()
    def _log_epoch_diagnostics(self, epoch: int):
        """Log model output statistics at end of epoch for debugging."""
        try:
            batch = next(iter(self.val_loader or self.train_loader))
            ncct_3c = batch["ncct"].to(self.device)
            cta_3c = batch["cta"].to(self.device)
            mask_3c = batch["ncct_lung"].to(self.device)
            ncct, cta, mask = self.augmentor(ncct_3c, cta_3c, mask_3c, training=False)

            self.model.eval()
            pred = self.model(ncct)
            residual = pred - ncct

            # Per-pixel error in normalized space
            l1_error = (pred - cta).abs().mean().item()
            # Residual magnitude
            res_mag = residual.abs().mean().item()

            logger.info(f"  [Debug E{epoch+1}] pred range=[{pred.min():.3f}, {pred.max():.3f}], "
                        f"residual_mag={res_mag:.4f}, l1_vs_cta={l1_error:.4f}")

            # Check if lung region has higher error (expected)
            if mask is not None and mask.sum() > 0:
                lung_err = ((pred - cta).abs() * (mask > 0.5).float()).sum() / max(1, (mask > 0.5).sum())
                bg_err = ((pred - cta).abs() * (mask <= 0.5).float()).sum() / max(1, (mask <= 0.5).sum())
                logger.info(f"  [Debug E{epoch+1}] lung_l1={lung_err:.4f}, bg_l1={bg_err:.4f}")

            self.model.train()
        except Exception:
            pass

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
        accum = self.grad_accumulation_steps

        # Zero gradients at the start of epoch
        self.optimizer_G.zero_grad()
        if self.use_disc:
            self.optimizer_D.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for micro_step, batch in enumerate(pbar):
            ncct_3c = batch["ncct"].to(self.device)
            cta_3c = batch["cta"].to(self.device)
            mask_3c = batch["ncct_lung"].to(self.device)

            ncct, cta, mask = self.augmentor(ncct_3c, cta_3c, mask_3c, training=True)

            is_accum_boundary = ((micro_step + 1) % accum == 0) or \
                                (micro_step + 1 == len(self.train_loader))

            # ============================================================
            # Step 1: Forward G + R, compute G loss
            # ============================================================
            pred = self.model(ncct)

            # Registration (if enabled)
            if self.use_reg:
                reg_out = self.reg_net(pred, cta)
                recon_target = reg_out["warped"]
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

            # GAN loss (generator side) — G wants D to classify fake as real
            gan_g_val = 0.0
            feat_match_val = 0.0
            if self.use_disc:
                fake_input = torch.cat([ncct, pred], dim=1)
                real_input = torch.cat([ncct, cta], dim=1)

                fake_features = self.discriminator(fake_input)
                with torch.no_grad():
                    real_features = self.discriminator(real_input)

                gan_g = self.gan_loss_fn(fake_features, target_is_real=True)
                feat_match = self.feat_match_fn(real_features, fake_features)
                g_loss = g_loss + self.gan_weight * gan_g + self.feat_match_weight * feat_match
                gan_g_val = gan_g.item()
                feat_match_val = feat_match.item()

            # Scale loss for gradient accumulation and backward
            (g_loss / accum).backward()

            # ============================================================
            # Step 2: Forward D, compute D loss
            # ============================================================
            d_loss_val = 0.0
            if self.use_disc:
                fake_input = torch.cat([ncct, pred.detach()], dim=1)
                real_input = torch.cat([ncct, cta], dim=1)

                fake_preds = self.discriminator(fake_input)
                real_preds = self.discriminator(real_input)

                # One-sided label smoothing: smooth real target (e.g. 0.9)
                real_target = 1.0 - self.label_smoothing if self.label_smoothing > 0 else None
                d_loss_real = self.gan_loss_fn(
                    real_preds, target_is_real=True, target_value=real_target,
                )
                d_loss_fake = self.gan_loss_fn(fake_preds, target_is_real=False)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

                (d_loss / accum).backward()
                d_loss_val = d_loss.item()

            # ============================================================
            # Step 3: Optimizer step at accumulation boundary
            # ============================================================
            if is_accum_boundary:
                # Gradient clipping
                grad_norm_g = self._clip_and_get_norm(self._g_param_list)
                if self.use_disc:
                    grad_norm_d = self._clip_and_get_norm(
                        list(self.discriminator.parameters())
                    )

                # Optimizer step
                self.optimizer_G.step()
                self.optimizer_G.zero_grad()

                if self.use_disc:
                    self.optimizer_D.step()
                    self.optimizer_D.zero_grad()

                # LR scheduler (step-level for cosine warmup)
                if self.scheduler_G and cfg.lr_scheduler == "cosine":
                    self.scheduler_G.step()
                if self.scheduler_D and self.config.discriminator.lr_scheduler == "cosine":
                    self.scheduler_D.step()

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
                "lr_G": self.optimizer_G.param_groups[0]["lr"],
            }
            if self.use_reg:
                metrics["smooth"] = smooth_loss_val
            if self.use_disc:
                metrics["gan_g"] = gan_g_val
                metrics["fm"] = feat_match_val
                metrics["d_loss"] = d_loss_val
                metrics["lr_D"] = self.optimizer_D.param_groups[0]["lr"]
            # Gradient norms (logged at accumulation boundaries only)
            if is_accum_boundary:
                metrics["gnorm_G"] = grad_norm_g
                if self.use_disc:
                    metrics["gnorm_D"] = grad_norm_d
            tracker.update(metrics)

            if self.global_step % cfg.log_interval == 0:
                pbar.set_postfix(**{k: f"{v:.4f}" for k, v in tracker.result().items()})

        return tracker

    def _clip_and_get_norm(self, params: list) -> float:
        """Clip gradients and return the total gradient norm (before clipping)."""
        if self.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, self.grad_clip_norm
            ).item()
        else:
            # Just compute norm without clipping
            grad_norm = 0.0
            for p in params:
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
        return grad_norm

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
            if self.scheduler_D:
                state["scheduler_D_state_dict"] = self.scheduler_D.state_dict()

        if filename is None:
            filename = f"checkpoint_epoch{epoch+1:04d}.pt"
        path = self.output_dir / filename
        torch.save(state, path)
        logger.info(f"  Saved checkpoint: {path}")

        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(state, best_path)

    def _load_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint (restores full state including optimizers)."""
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

        # Restore G scheduler
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
            # Restore D scheduler
            if self.scheduler_D and "scheduler_D_state_dict" in state:
                self.scheduler_D.load_state_dict(state["scheduler_D_state_dict"])

        logger.info(f"  Resumed at epoch {self.start_epoch}, step {self.global_step}")
