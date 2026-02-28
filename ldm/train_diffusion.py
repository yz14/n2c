"""
Diffusion UNet Training Script — LDM Stage 2.

Trains the conditional diffusion UNet in the frozen VAE's latent space.
Requires a pretrained VAE checkpoint from Stage 1.

Usage:
    python -m ldm.train_diffusion --config configs/ldm_default.yaml \
        --pretrained_vae outputs_vae/checkpoint_best.pt
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldm.config import LDMConfig
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.unet import DiffusionUNet
from ldm.diffusion.scheduler import DDPMScheduler
from ldm.trainer_diffusion import DiffusionTrainer
from data.dataset import NCCTDataset
from data.transforms import GPUAugmentor

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / "train_diffusion.log"),
        ],
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: LDMConfig):
    data_cfg = cfg.data
    split_dir = Path(data_cfg.split_dir)

    train_dataset = NCCTDataset(
        data_dir=data_cfg.data_dir,
        split_file=str(split_dir / "train.txt"),
        num_slices=data_cfg.num_slices,
        image_size=data_cfg.image_size,
        hu_min=data_cfg.hu_min,
        hu_max=data_cfg.hu_max,
    )

    val_dataset = NCCTDataset(
        data_dir=data_cfg.data_dir,
        split_file=str(split_dir / "valid.txt"),
        num_slices=data_cfg.num_slices,
        image_size=data_cfg.image_size,
        hu_min=data_cfg.hu_min,
        hu_max=data_cfg.hu_max,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.diffusion_train.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.diffusion_train.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def load_pretrained_vae(cfg: LDMConfig, vae_path: str, device: torch.device) -> AutoencoderKL:
    """Load a pretrained VAE from a Stage 1 checkpoint."""
    vae = AutoencoderKL(cfg.vae)

    logger.info(f"Loading pretrained VAE from: {vae_path}")
    state = torch.load(vae_path, map_location=device)

    # Support both full checkpoint and raw state dict
    if "vae_state_dict" in state:
        vae_sd = state["vae_state_dict"]
    elif "ema_params" in state and isinstance(state["ema_params"], list):
        # Load EMA weights if available (better quality)
        logger.info("  Using EMA weights from VAE checkpoint")
        ema_params = state["ema_params"]
        vae_sd = vae.state_dict()
        for (name, _), ema_p in zip(vae.named_parameters(), ema_params):
            vae_sd[name] = ema_p
    else:
        vae_sd = state

    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
    if missing:
        logger.warning(f"  Missing keys: {missing[:5]}...")
    if unexpected:
        logger.warning(f"  Unexpected keys: {unexpected[:5]}...")

    n_loaded = len(vae_sd) - len(unexpected)
    n_total = len(list(vae.state_dict().keys()))
    logger.info(f"  Loaded {n_loaded}/{n_total} parameter tensors")

    return vae


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion UNet (LDM Stage 2)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to LDM YAML config")
    parser.add_argument("--pretrained_vae", type=str, default=None,
                        help="Path to pretrained VAE checkpoint (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to diffusion checkpoint for resuming")
    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        cfg = LDMConfig.load(args.config)
    else:
        cfg = LDMConfig()
    cfg.sync_channels()

    if args.pretrained_vae:
        cfg.diffusion_train.pretrained_vae = args.pretrained_vae
    if args.resume:
        cfg.diffusion_train.resume_checkpoint = args.resume

    # Validate
    if not cfg.diffusion_train.pretrained_vae:
        raise ValueError(
            "Pretrained VAE is required for Stage 2. "
            "Provide --pretrained_vae or set diffusion_train.pretrained_vae in config."
        )

    # Setup
    setup_logging(cfg.diffusion_train.output_dir)
    set_seed(cfg.diffusion_train.seed)
    cfg.save(str(Path(cfg.diffusion_train.output_dir) / "config.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Load pretrained VAE (frozen)
    vae = load_pretrained_vae(cfg, cfg.diffusion_train.pretrained_vae, device)

    # Build diffusion UNet
    unet = DiffusionUNet.from_config(cfg.unet, z_channels=cfg.vae.embed_dim)

    # Build noise scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.scheduler.num_train_timesteps,
        beta_schedule=cfg.scheduler.beta_schedule,
        beta_start=cfg.scheduler.beta_start,
        beta_end=cfg.scheduler.beta_end,
        prediction_type=cfg.scheduler.prediction_type,
    )

    # Augmentor
    augmentor = GPUAugmentor.from_config(cfg.data)

    # Train
    trainer = DiffusionTrainer(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        augmentor=augmentor,
        config=cfg,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
