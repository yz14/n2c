"""Quick test script to validate the full pipeline."""

import logging
import os
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

C = 3  # num_slices


def test_config():
    logger.info("=== Testing Config ===")
    from config import Config

    cfg = Config()
    cfg.sync_channels()
    logger.info(f"Data config:  slices={cfg.data.num_slices}, size={cfg.data.image_size}")
    logger.info(f"Model config: channels={cfg.model.model_channels}, "
                f"mult={cfg.model.channel_mult}")
    logger.info(f"Train config: lr={cfg.train.lr}, use_3d_ssim={cfg.train.use_3d_ssim}")

    # Test save/load round-trip
    cfg.save("./test_config.yaml")
    cfg2 = Config.load("./test_config.yaml")
    assert cfg2.data.num_slices == cfg.data.num_slices
    assert cfg2.train.use_3d_ssim == cfg.train.use_3d_ssim
    os.remove("./test_config.yaml")
    logger.info("Config test PASSED\n")


def test_dataset():
    logger.info("=== Testing Dataset (returns 3C slices) ===")
    from data.dataset import NCCTDataset

    ds = NCCTDataset(
        data_dir="D:/codes/data/ncct_tiny",
        split_file="./splits/train.txt",
        num_slices=C,
        image_size=256,
    )
    logger.info(f"Dataset size: {len(ds)}")

    sample = ds[0]
    ncct = sample["ncct"]
    cta = sample["cta"]
    mask = sample["ncct_lung"]
    logger.info(f"ncct:  shape={ncct.shape}, range=[{ncct.min():.3f}, {ncct.max():.3f}]")
    logger.info(f"cta:   shape={cta.shape}, range=[{cta.min():.3f}, {cta.max():.3f}]")
    logger.info(f"mask:  shape={mask.shape}, range=[{mask.min():.3f}, {mask.max():.3f}]")

    assert ncct.shape == (3 * C, 256, 256), f"Unexpected shape: {ncct.shape}"
    assert cta.shape == (3 * C, 256, 256)
    assert mask.shape == (3 * C, 256, 256)
    logger.info("Dataset test PASSED\n")
    return ds


def test_dataloader(ds):
    logger.info("=== Testing DataLoader ===")
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    logger.info(f"Batch ncct:  {batch['ncct'].shape}")
    logger.info(f"Batch cta:   {batch['cta'].shape}")
    assert batch["ncct"].shape[1] == 3 * C
    logger.info("DataLoader test PASSED\n")
    return batch


def test_gpu_augmentor(batch):
    logger.info("=== Testing GPUAugmentor ===")
    from data.transforms import GPUAugmentor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augmentor = GPUAugmentor(num_slices=C, aug_prob=1.0)

    ncct = batch["ncct"].to(device)
    cta = batch["cta"].to(device)
    mask = batch["ncct_lung"].to(device)
    logger.info(f"Input shapes: ncct={ncct.shape}, cta={cta.shape}, mask={mask.shape}")

    # Training mode (with augmentation)
    ncct_aug, cta_aug, mask_aug = augmentor(ncct, cta, mask, training=True)
    logger.info(f"Output (train): ncct={ncct_aug.shape}, cta={cta_aug.shape}")
    assert ncct_aug.shape[1] == C, f"Expected C={C} channels, got {ncct_aug.shape[1]}"

    # Eval mode (no augmentation, just extract middle slices)
    ncct_val, cta_val, mask_val = augmentor(ncct, cta, mask, training=False)
    logger.info(f"Output (val):   ncct={ncct_val.shape}")
    assert ncct_val.shape[1] == C

    logger.info("GPUAugmentor test PASSED\n")
    return ncct_aug, cta_aug


def test_model():
    logger.info("=== Testing UNet Model ===")
    from models.unet import UNet

    model = UNet(
        image_size=256,
        in_channels=C,
        model_channels=64,
        out_channels=C,
        num_res_blocks=2,
        attention_resolutions=(4, 8),
        channel_mult=(1, 2, 4, 8),
        num_heads=4,
        residual_output=True,
    )
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {params:.2f}M")

    x = torch.randn(2, C, 256, 256)
    y = model(x)
    logger.info(f"Input: {x.shape} -> Output: {y.shape}")
    assert x.shape == y.shape
    logger.info("Model test PASSED\n")


def test_loss():
    logger.info("=== Testing Loss Functions (2D + 3D SSIM) ===")
    from models.losses import CombinedLoss

    # 2D SSIM
    criterion_2d = CombinedLoss(use_3d_ssim=False)
    pred = torch.randn(2, C, 64, 64)
    target = torch.randn(2, C, 64, 64)
    loss_2d = criterion_2d(pred, target)
    logger.info(f"2D SSIM - total: {loss_2d['loss'].item():.4f}, "
                f"l1: {loss_2d['l1'].item():.4f}, ssim: {loss_2d['ssim'].item():.4f}")

    # 3D SSIM
    criterion_3d = CombinedLoss(use_3d_ssim=True)
    loss_3d = criterion_3d(pred, target)
    logger.info(f"3D SSIM - total: {loss_3d['loss'].item():.4f}, "
                f"l1: {loss_3d['l1'].item():.4f}, ssim: {loss_3d['ssim'].item():.4f}")

    # Same image should have ~0 loss
    same = torch.randn(2, C, 64, 64)
    loss_same = criterion_3d(same, same)
    logger.info(f"Same-image loss (3D): {loss_same['loss'].item():.6f}")
    assert loss_same["loss"].item() < 0.01
    logger.info("Loss test PASSED\n")


def test_visualization():
    logger.info("=== Testing Visualization ===")
    from utils.visualization import save_sample_grid

    inputs = torch.randn(8, C, 64, 64)
    preds = torch.randn(8, C, 64, 64)
    targets = torch.randn(8, C, 64, 64)

    path = "./test_vis_grid.png"
    save_sample_grid(inputs, preds, targets, path, num_samples=8)
    exists = os.path.exists(path) or os.path.exists(path.replace(".png", ".pgm"))
    if exists:
        logger.info(f"Grid saved successfully")
        for ext in [".png", ".pgm"]:
            p = path.replace(".png", ext)
            if os.path.exists(p):
                os.remove(p)
    else:
        logger.warning("Grid file not found — check torchvision availability")
    logger.info("Visualization test PASSED\n")


def test_forward_backward():
    logger.info("=== Testing Forward+Backward with GPU Augmentor ===")
    from models.unet import UNet
    from models.losses import CombinedLoss
    from data.transforms import GPUAugmentor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = UNet(
        image_size=128,
        in_channels=C,
        model_channels=32,
        out_channels=C,
        num_res_blocks=1,
        attention_resolutions=(4,),
        channel_mult=(1, 2, 4),
        num_heads=2,
        residual_output=True,
    ).to(device)

    criterion = CombinedLoss(use_3d_ssim=True).to(device)
    augmentor = GPUAugmentor(num_slices=C, aug_prob=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Simulate batch: (N, 3C, H, W) as from dataset
    ncct_3c = torch.randn(2, 3 * C, 128, 128, device=device)
    cta_3c = torch.randn(2, 3 * C, 128, 128, device=device)
    mask_3c = torch.zeros(2, 3 * C, 128, 128, device=device)

    # GPU augmentation
    ncct, cta, mask = augmentor(ncct_3c, cta_3c, mask_3c, training=True)
    logger.info(f"After augmentor: ncct={ncct.shape}, cta={cta.shape}")

    # Forward
    pred = model(ncct)
    loss_dict = criterion(pred, cta)
    logger.info(f"Loss: {loss_dict['loss'].item():.4f}")

    # Backward
    optimizer.zero_grad()
    loss_dict["loss"].backward()
    optimizer.step()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    logger.info(f"Gradient norm: {grad_norm:.4f}")
    logger.info("Forward+Backward test PASSED\n")


if __name__ == "__main__":
    test_config()
    ds = test_dataset()
    batch = test_dataloader(ds)
    test_gpu_augmentor(batch)
    test_model()
    test_loss()
    test_visualization()
    test_forward_backward()
    logger.info("=" * 40)
    logger.info("ALL TESTS PASSED!")
