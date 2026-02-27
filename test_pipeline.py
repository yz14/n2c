"""Quick test script to validate the full pipeline."""

import logging
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_dataset():
    logger.info("=== Testing Dataset ===")
    from data.dataset import NCCTDataset

    ds = NCCTDataset(
        data_dir="D:/codes/data/ncct_tiny",
        split_file="./splits/train.txt",
        num_slices=3,
        image_size=256,
        augment=True,
    )
    logger.info(f"Dataset size: {len(ds)}")

    sample = ds[0]
    ncct = sample["ncct"]
    cta = sample["cta"]
    mask = sample["ncct_lung"]
    logger.info(f"ncct:  shape={ncct.shape}, dtype={ncct.dtype}, "
                f"range=[{ncct.min():.3f}, {ncct.max():.3f}]")
    logger.info(f"cta:   shape={cta.shape}, dtype={cta.dtype}, "
                f"range=[{cta.min():.3f}, {cta.max():.3f}]")
    logger.info(f"mask:  shape={mask.shape}, dtype={mask.dtype}, "
                f"range=[{mask.min():.3f}, {mask.max():.3f}]")
    logger.info(f"file:  {sample['filename']}")

    assert ncct.shape == (3, 256, 256), f"Unexpected ncct shape: {ncct.shape}"
    assert cta.shape == (3, 256, 256), f"Unexpected cta shape: {cta.shape}"
    assert mask.shape == (3, 256, 256), f"Unexpected mask shape: {mask.shape}"
    logger.info("Dataset test PASSED\n")
    return ds


def test_dataloader(ds):
    logger.info("=== Testing DataLoader ===")
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    logger.info(f"Batch ncct:  {batch['ncct'].shape}")
    logger.info(f"Batch cta:   {batch['cta'].shape}")
    logger.info(f"Batch mask:  {batch['ncct_lung'].shape}")
    logger.info("DataLoader test PASSED\n")
    return batch


def test_model():
    logger.info("=== Testing UNet Model ===")
    from models.unet import UNet

    model = UNet(
        image_size=256,
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(4, 8),
        channel_mult=(1, 2, 4, 8),
        num_heads=4,
        residual_output=True,
    )
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {params:.2f}M")

    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    logger.info(f"Input: {x.shape} → Output: {y.shape}")
    assert x.shape == y.shape, "Input/output shape mismatch"
    logger.info("Model test PASSED\n")
    return model


def test_loss():
    logger.info("=== Testing Loss Functions ===")
    from models.losses import CombinedLoss

    criterion = CombinedLoss(l1_weight=1.0, ssim_weight=1.0)
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    loss_dict = criterion(pred, target)
    logger.info(f"Total loss: {loss_dict['loss'].item():.4f}")
    logger.info(f"L1 loss:    {loss_dict['l1'].item():.4f}")
    logger.info(f"SSIM loss:  {loss_dict['ssim'].item():.4f}")

    # Test that identical images give low loss
    same = torch.randn(2, 3, 64, 64)
    loss_same = criterion(same, same)
    logger.info(f"Same-image loss: {loss_same['loss'].item():.6f} (should be ~0)")
    assert loss_same["loss"].item() < 0.01, "Same-image loss too high"
    logger.info("Loss test PASSED\n")


def test_forward_backward():
    logger.info("=== Testing Forward+Backward ===")
    from models.unet import UNet
    from models.losses import CombinedLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = UNet(
        image_size=256,
        in_channels=3,
        model_channels=32,  # smaller for speed
        out_channels=3,
        num_res_blocks=1,
        attention_resolutions=(4,),
        channel_mult=(1, 2, 4),
        num_heads=2,
        residual_output=True,
    ).to(device)

    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ncct = torch.randn(2, 3, 128, 128, device=device)
    cta = torch.randn(2, 3, 128, 128, device=device)

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


def test_config():
    logger.info("=== Testing Config ===")
    from config import Config

    cfg = Config()
    cfg.sync_channels()
    logger.info(f"Data config:  slices={cfg.data.num_slices}, size={cfg.data.image_size}")
    logger.info(f"Model config: channels={cfg.model.model_channels}, "
                f"mult={cfg.model.channel_mult}")
    logger.info(f"Train config: lr={cfg.train.lr}, epochs={cfg.train.num_epochs}")

    # Test save/load round-trip
    cfg.save("./test_config.yaml")
    cfg2 = Config.load("./test_config.yaml")
    assert cfg2.data.num_slices == cfg.data.num_slices
    assert cfg2.model.model_channels == cfg.model.model_channels
    import os
    os.remove("./test_config.yaml")
    logger.info("Config test PASSED\n")


if __name__ == "__main__":
    test_config()
    ds = test_dataset()
    test_dataloader(ds)
    test_model()
    test_loss()
    test_forward_backward()
    logger.info("=" * 40)
    logger.info("ALL TESTS PASSED!")
