"""
Training loop for NCCT→CTA image translation.

Features:
  - Train/validation loop with logging
  - EMA (Exponential Moving Average) of model weights
  - Learning rate scheduling (cosine with warmup)
  - Checkpoint saving and resuming
  - Metric tracking
"""

import copy
import logging
import time
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR
from tqdm import tqdm

from models.nn_utils import update_ema

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
        return max(0.0, 0.5 * (1.0 + __import__("math").cos(3.141592653589793 * progress)))
    return lr_lambda


class Trainer:
    """
    Training manager for the UNet model.

    Handles the full training lifecycle: data loading, optimization,
    validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )

        # LR scheduler
        total_steps = config.train.num_epochs * len(train_loader)
        if config.train.lr_scheduler == "cosine":
            lr_lambda = _warmup_cosine_schedule(config.train.warmup_steps, total_steps)
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        elif config.train.lr_scheduler == "step":
            self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)
        else:
            self.scheduler = None

        # EMA
        self.ema_rate = config.train.ema_rate
        self.ema_params = [p.clone().detach() for p in self.model.parameters()]

        # State
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        # Output directory
        self.output_dir = Path(config.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resume if specified
        if config.train.resume_checkpoint:
            self._load_checkpoint(config.train.resume_checkpoint)

    def train(self):
        """Run the full training loop."""
        cfg = self.config.train
        logger.info(f"Starting training for {cfg.num_epochs} epochs")
        logger.info(f"  Batch size:    {cfg.batch_size}")
        logger.info(f"  Learning rate: {cfg.lr}")
        logger.info(f"  Output dir:    {self.output_dir}")

        for epoch in range(self.start_epoch, cfg.num_epochs):
            train_metrics = self._train_epoch(epoch)
            logger.info(
                f"Epoch {epoch+1}/{cfg.num_epochs} [Train] {train_metrics}"
            )

            # Validation
            if self.val_loader and (epoch + 1) % cfg.val_interval == 0:
                val_metrics = self._validate(epoch)
                logger.info(
                    f"Epoch {epoch+1}/{cfg.num_epochs} [Val]   {val_metrics}"
                )
                val_loss = val_metrics.result()["loss"]
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f"  New best validation loss: {val_loss:.6f}")

            # Periodic checkpoint
            if (epoch + 1) % cfg.save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Step-based LR scheduler
            if self.scheduler and cfg.lr_scheduler == "step":
                self.scheduler.step()

        # Save final checkpoint
        self._save_checkpoint(cfg.num_epochs - 1, is_best=False, filename="checkpoint_final.pt")
        logger.info("Training complete.")

    def _train_epoch(self, epoch: int) -> MetricTracker:
        """Run one training epoch."""
        self.model.train()
        tracker = MetricTracker()
        cfg = self.config.train

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            ncct = batch["ncct"].to(self.device)
            cta = batch["cta"].to(self.device)

            # Forward
            pred = self.model(ncct)
            loss_dict = self.criterion(pred, cta)

            # Backward
            self.optimizer.zero_grad()
            loss_dict["loss"].backward()
            self.optimizer.step()

            # LR scheduler (step-level for cosine warmup)
            if self.scheduler and self.config.train.lr_scheduler == "cosine":
                self.scheduler.step()

            # EMA update
            update_ema(self.ema_params, self.model.parameters(), rate=self.ema_rate)

            # Logging
            self.global_step += 1
            tracker.update({
                "loss": loss_dict["loss"].item(),
                "l1": loss_dict["l1"].item(),
                "ssim": loss_dict["ssim"].item(),
                "lr": self.optimizer.param_groups[0]["lr"],
            })

            if self.global_step % cfg.log_interval == 0:
                pbar.set_postfix(**{k: f"{v:.4f}" for k, v in tracker.result().items()})

        return tracker

    @torch.no_grad()
    def _validate(self, epoch: int) -> MetricTracker:
        """Run validation."""
        self.model.eval()
        tracker = MetricTracker()

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            ncct = batch["ncct"].to(self.device)
            cta = batch["cta"].to(self.device)

            pred = self.model(ncct)
            loss_dict = self.criterion(pred, cta)

            tracker.update({
                "loss": loss_dict["loss"].item(),
                "l1": loss_dict["l1"].item(),
                "ssim": loss_dict["ssim"].item(),
            })

        return tracker

    def _save_checkpoint(self, epoch: int, is_best: bool = False,
                         filename: Optional[str] = None):
        """Save training checkpoint."""
        state = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "ema_params": self.ema_params,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

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
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.start_epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        if "ema_params" in state:
            self.ema_params = state["ema_params"]
        if self.scheduler and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info(f"  Resumed at epoch {self.start_epoch}, step {self.global_step}")
