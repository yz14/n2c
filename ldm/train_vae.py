"""
VAE Training Script — LDM Stage 1.

Trains the AutoencoderKL to reconstruct 2.5D medical CT images.
Reuses the existing NCCTDataset and GPUAugmentor from Scheme 1.

Usage:
    python -m ldm.train_vae --config configs/ldm_default.yaml
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldm.config import LDMConfig
from ldm.models.autoencoder import AutoencoderKL
from ldm.trainer_vae import VAETrainer
from data.dataset import NCCTDataset
from data.transforms import GPUAugmentor

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str):
    """Configure logging to both console and file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / "train_vae.log"),
        ],
    )


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: LDMConfig):
    """Create train and validation DataLoaders."""
    data_cfg = cfg.data
    split_dir = Path(data_cfg.split_dir)

    train_file = str(split_dir / "train.txt")
    val_file = str(split_dir / "valid.txt")

    train_dataset = NCCTDataset(
        data_dir=data_cfg.data_dir,
        split_file=train_file,
        num_slices=data_cfg.num_slices,
        image_size=data_cfg.image_size,
        hu_min=data_cfg.hu_min,
        hu_max=data_cfg.hu_max,
    )

    val_dataset = NCCTDataset(
        data_dir=data_cfg.data_dir,
        split_file=val_file,
        num_slices=data_cfg.num_slices,
        image_size=data_cfg.image_size,
        hu_min=data_cfg.hu_min,
        hu_max=data_cfg.hu_max,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.vae_train.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.vae_train.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train VAE (LDM Stage 1)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to LDM YAML config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for resuming training")
    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        cfg = LDMConfig.load(args.config)
    else:
        cfg = LDMConfig()
    cfg.sync_channels()

    if args.resume:
        cfg.vae_train.resume_checkpoint = args.resume

    # Setup
    setup_logging(cfg.vae_train.output_dir)
    set_seed(cfg.vae_train.seed)

    # Save config for reproducibility
    cfg.save(str(Path(cfg.vae_train.output_dir) / "config.yaml"))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    vae = AutoencoderKL(cfg.vae)

    # Augmentor
    augmentor = GPUAugmentor.from_config(cfg.data)

    # Train
    trainer = VAETrainer(
        vae=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        augmentor=augmentor,
        config=cfg,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
