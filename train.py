"""
Entry point for NCCT→CTA training.

Usage:
    python train.py                          # default config
    python train.py --config configs/my.yaml # custom config
    python train.py --split_only             # only run data splitting
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data.split import split_dataset
from data.dataset import NCCTDataset
from data.transforms import GPUAugmentor
from models.unet import UNet
from models.losses import CombinedLoss
from trainer import Trainer

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str):
    """Configure logging to console and file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(output_dir) / "train.log"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_splits(cfg: Config) -> tuple:
    """Ensure train/valid/test split files exist."""
    split_dir = Path(cfg.data.split_dir)
    train_file = split_dir / "train.txt"
    valid_file = split_dir / "valid.txt"
    test_file = split_dir / "test.txt"

    if not all(f.exists() for f in [train_file, valid_file, test_file]):
        logger.info("Split files not found, generating...")
        split_dataset(
            data_dir=cfg.data.data_dir,
            output_dir=str(split_dir),
            seed=cfg.train.seed,
        )
    else:
        logger.info("Using existing split files")

    return str(train_file), str(valid_file), str(test_file)


def build_dataloaders(cfg: Config, train_file: str, valid_file: str):
    """Create train and validation DataLoaders."""
    data_cfg = cfg.data

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
        split_file=valid_file,
        num_slices=data_cfg.num_slices,
        image_size=data_cfg.image_size,
        hu_min=data_cfg.hu_min,
        hu_max=data_cfg.hu_max,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_model(cfg: Config) -> UNet:
    """Create UNet model from config."""
    mcfg = cfg.model
    return UNet(
        image_size=cfg.data.image_size,
        in_channels=mcfg.in_channels,
        model_channels=mcfg.model_channels,
        out_channels=mcfg.out_channels,
        num_res_blocks=mcfg.num_res_blocks,
        attention_resolutions=tuple(mcfg.attention_resolutions),
        dropout=mcfg.dropout,
        channel_mult=tuple(mcfg.channel_mult),
        conv_resample=mcfg.conv_resample,
        use_checkpoint=mcfg.use_checkpoint,
        use_fp16=mcfg.use_fp16,
        num_heads=mcfg.num_heads,
        num_head_channels=mcfg.num_head_channels,
        resblock_updown=mcfg.resblock_updown,
        residual_output=mcfg.residual_output,
    )


def main():
    parser = argparse.ArgumentParser(description="Train NCCT→CTA model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--split_only", action="store_true", help="Only run data splitting")
    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        cfg = Config.load(args.config)
    else:
        cfg = Config()
    cfg.sync_channels()

    # Setup
    setup_logging(cfg.train.output_dir)
    set_seed(cfg.train.seed)

    # Save config for reproducibility
    cfg.save(str(Path(cfg.train.output_dir) / "config.yaml"))

    # Data splitting
    train_file, valid_file, test_file = ensure_splits(cfg)
    if args.split_only:
        logger.info("Split only mode — done.")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build components
    train_loader, val_loader = build_dataloaders(cfg, train_file, valid_file)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(cfg)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {param_count:.2f}M")

    criterion = CombinedLoss(
        l1_weight=cfg.train.l1_weight,
        ssim_weight=cfg.train.ssim_weight,
        use_3d_ssim=cfg.train.use_3d_ssim,
    )

    # GPU augmentor
    augmentor = GPUAugmentor.from_config(cfg.data)

    # Train
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        augmentor=augmentor,
        config=cfg,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
