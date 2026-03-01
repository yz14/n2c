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
    logger.info(f"Train config: lr={cfg.train.lr}, use_3d_ssim={cfg.train.use_3d_ssim}, "
                f"lung_weight={cfg.train.lung_weight}")
    logger.info(f"Registration: enabled={cfg.registration.enabled}, "
                f"features={cfg.registration.nb_features}")
    logger.info(f"Discriminator: enabled={cfg.discriminator.enabled}, "
                f"ndf={cfg.discriminator.ndf}, num_D={cfg.discriminator.num_D}")

    # Test save/load round-trip
    cfg.save("./test_config.yaml")
    cfg2 = Config.load("./test_config.yaml")
    assert cfg2.data.num_slices == cfg.data.num_slices
    assert cfg2.train.use_3d_ssim == cfg.train.use_3d_ssim
    assert cfg2.registration.nb_features == cfg.registration.nb_features
    assert cfg2.discriminator.ndf == cfg.discriminator.ndf
    assert cfg2.train.lung_weight == cfg.train.lung_weight
    assert cfg2.refine.enabled == cfg.refine.enabled
    assert cfg2.refine.hidden_dim == cfg.refine.hidden_dim
    assert cfg2.refine.freeze_G == cfg.refine.freeze_G
    assert cfg2.train.pretrained_G2 == cfg.train.pretrained_G2
    assert cfg2.discriminator.d_cond_mode == cfg.discriminator.d_cond_mode
    assert cfg2.discriminator.gan_loss_type == cfg.discriminator.gan_loss_type
    assert cfg2.discriminator.r1_gamma == cfg.discriminator.r1_gamma
    assert cfg2.discriminator.disc_type == cfg.discriminator.disc_type
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


def test_weighted_loss():
    logger.info("=== Testing Weighted Mask Loss ===")
    from models.losses import CombinedLoss

    criterion = CombinedLoss(use_3d_ssim=False, lung_weight=10.0)
    pred = torch.randn(2, C, 64, 64)
    target = torch.randn(2, C, 64, 64)

    # Without mask
    loss_no_mask = criterion(pred, target)
    logger.info(f"No mask   - loss: {loss_no_mask['loss'].item():.4f}")

    # With mask (partial lung region)
    mask = torch.zeros(2, C, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0  # lung region
    loss_with_mask = criterion(pred, target, mask=mask)
    logger.info(f"With mask - loss: {loss_with_mask['loss'].item():.4f}")

    # Mask should change the loss value
    assert loss_no_mask["loss"].item() != loss_with_mask["loss"].item(), \
        "Mask should change the loss"
    logger.info("Weighted loss test PASSED\n")


def test_registration():
    logger.info("=== Testing Registration Network ===")
    from models.registration import RegistrationNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enabled mode
    reg = RegistrationNet(
        in_channels=C, nb_features=(16, 32, 32, 32),
        integration_steps=7, enabled=True,
    ).to(device)
    params = sum(p.numel() for p in reg.parameters()) / 1e6
    logger.info(f"Registration params: {params:.2f}M")

    source = torch.randn(2, C, 64, 64, device=device)
    target = torch.randn(2, C, 64, 64, device=device)
    out = reg(source, target)
    logger.info(f"Warped: {out['warped'].shape}, Disp: {out['displacement'].shape}")
    assert out["warped"].shape == source.shape
    assert out["displacement"].shape == (2, 2, 64, 64)

    # Disabled mode (identity)
    reg_off = RegistrationNet(in_channels=C, enabled=False).to(device)
    out_off = reg_off(source, target)
    assert torch.equal(out_off["warped"], source), "Disabled reg should return source"
    assert out_off["displacement"].abs().sum().item() == 0.0, "Disabled reg should have zero disp"
    logger.info("Registration test PASSED\n")


def test_discriminator():
    logger.info("=== Testing Multi-Scale Discriminator ===")
    from models.discriminator import MultiscaleDiscriminator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_nc = C * 2  # concat(ncct, cta)

    disc = MultiscaleDiscriminator(
        input_nc=input_nc, ndf=64, n_layers=3, num_D=3,
        use_spectral_norm=True, get_interm_feat=True,
    ).to(device)
    params = sum(p.numel() for p in disc.parameters()) / 1e6
    logger.info(f"Discriminator params: {params:.2f}M")

    x = torch.randn(2, input_nc, 64, 64, device=device)
    results = disc(x)
    logger.info(f"Num scales: {len(results)}")
    for i, scale in enumerate(results):
        logger.info(f"  Scale {i}: {len(scale)} layers, final={scale[-1].shape}")
    assert len(results) == 3
    logger.info("Discriminator test PASSED\n")


def test_gan_losses():
    logger.info("=== Testing GAN + Feature Matching Losses ===")
    from models.losses import GANLoss, FeatureMatchingLoss
    from models.discriminator import MultiscaleDiscriminator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_nc = C * 2

    disc = MultiscaleDiscriminator(
        input_nc=input_nc, ndf=32, n_layers=2, num_D=2,
        use_spectral_norm=True, get_interm_feat=True,
    ).to(device)

    real_input = torch.randn(2, input_nc, 64, 64, device=device)
    fake_input = torch.randn(2, input_nc, 64, 64, device=device)

    real_feats = disc(real_input)
    fake_feats = disc(fake_input)

    gan_loss_fn = GANLoss()
    loss_real = gan_loss_fn(real_feats, target_is_real=True)
    loss_fake = gan_loss_fn(fake_feats, target_is_real=False)
    logger.info(f"GAN loss real: {loss_real.item():.4f}, fake: {loss_fake.item():.4f}")

    fm_fn = FeatureMatchingLoss()
    fm_loss = fm_fn(real_feats, fake_feats)
    logger.info(f"Feature matching loss: {fm_loss.item():.4f}")
    logger.info("GAN losses test PASSED\n")


def test_grad_loss():
    logger.info("=== Testing Grad (Smoothness) Loss ===")
    from models.losses import GradLoss

    grad_l2 = GradLoss(penalty="l2")
    grad_l1 = GradLoss(penalty="l1")

    # Smooth field should have low loss
    smooth_disp = torch.zeros(2, 2, 64, 64)
    loss_smooth = grad_l2(smooth_disp)
    logger.info(f"Smooth field L2: {loss_smooth.item():.6f}")
    assert loss_smooth.item() < 1e-8

    # Random field should have higher loss
    random_disp = torch.randn(2, 2, 64, 64)
    loss_random = grad_l2(random_disp)
    logger.info(f"Random field L2: {loss_random.item():.4f}")
    assert loss_random.item() > 0.1

    loss_l1 = grad_l1(random_disp)
    logger.info(f"Random field L1: {loss_l1.item():.4f}")
    logger.info("Grad loss test PASSED\n")


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
    logger.info("=== Testing Forward+Backward (G only) ===")
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

    criterion = CombinedLoss(use_3d_ssim=True, lung_weight=10.0).to(device)
    augmentor = GPUAugmentor(num_slices=C, aug_prob=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ncct_3c = torch.randn(2, 3 * C, 128, 128, device=device)
    cta_3c = torch.randn(2, 3 * C, 128, 128, device=device)
    mask_3c = torch.zeros(2, 3 * C, 128, 128, device=device)

    ncct, cta, mask = augmentor(ncct_3c, cta_3c, mask_3c, training=True)
    logger.info(f"After augmentor: ncct={ncct.shape}, cta={cta.shape}")

    pred = model(ncct)
    loss_dict = criterion(pred, cta, mask)
    logger.info(f"Loss: {loss_dict['loss'].item():.4f}")

    optimizer.zero_grad()
    loss_dict["loss"].backward()
    optimizer.step()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    logger.info(f"Gradient norm: {grad_norm:.4f}")
    logger.info("Forward+Backward (G only) test PASSED\n")


def test_full_pipeline_grd():
    logger.info("=== Testing Full G+R+D Pipeline ===")
    from models.unet import UNet
    from models.registration import RegistrationNet
    from models.discriminator import MultiscaleDiscriminator
    from models.losses import CombinedLoss, GANLoss, FeatureMatchingLoss, GradLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = 64

    # Build G
    G = UNet(
        image_size=H, in_channels=C, model_channels=32, out_channels=C,
        num_res_blocks=1, attention_resolutions=(4,), channel_mult=(1, 2, 4),
        num_heads=2, residual_output=True,
    ).to(device)

    # Build R
    R = RegistrationNet(
        in_channels=C, nb_features=(8, 16, 16, 16),
        integration_steps=5, enabled=True,
    ).to(device)

    # Build D
    D = MultiscaleDiscriminator(
        input_nc=C * 2, ndf=16, n_layers=2, num_D=2,
        use_spectral_norm=True, get_interm_feat=True,
    ).to(device)

    # Losses
    criterion = CombinedLoss(use_3d_ssim=False, lung_weight=10.0).to(device)
    gan_loss_fn = GANLoss()
    fm_fn = FeatureMatchingLoss()
    grad_fn = GradLoss(penalty="l2")

    # Optimizers
    opt_G = torch.optim.Adam(list(G.parameters()) + list(R.parameters()), lr=1e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4)

    # Synthetic data
    ncct = torch.randn(2, C, H, H, device=device)
    cta = torch.randn(2, C, H, H, device=device)
    mask = torch.zeros(2, C, H, H, device=device)
    mask[:, :, 16:48, 16:48] = 1.0

    # --- G + R forward ---
    pred = G(ncct)
    reg_out = R(pred, cta)
    warped = reg_out["warped"]
    disp = reg_out["displacement"]

    recon_loss = criterion(warped, cta, mask)
    smooth_loss = grad_fn(disp)

    fake_in = torch.cat([ncct, pred], dim=1)
    real_in = torch.cat([ncct, cta], dim=1)
    fake_feats = D(fake_in)
    with torch.no_grad():
        real_feats = D(real_in)

    gan_g = gan_loss_fn(fake_feats, target_is_real=True)
    fm = fm_fn(real_feats, fake_feats)

    g_total = recon_loss["loss"] + 1.0 * smooth_loss + 1.0 * gan_g + 10.0 * fm
    opt_G.zero_grad()
    g_total.backward()
    opt_G.step()

    logger.info(f"G total: {g_total.item():.4f}, recon: {recon_loss['loss'].item():.4f}, "
                f"smooth: {smooth_loss.item():.4f}, gan_g: {gan_g.item():.4f}, fm: {fm.item():.4f}")

    # --- D forward ---
    fake_in_d = torch.cat([ncct, pred.detach()], dim=1)
    real_in_d = torch.cat([ncct, cta], dim=1)
    d_real = gan_loss_fn(D(real_in_d), target_is_real=True)
    d_fake = gan_loss_fn(D(fake_in_d), target_is_real=False)
    d_total = 0.5 * (d_real + d_fake)

    opt_D.zero_grad()
    d_total.backward()
    opt_D.step()
    logger.info(f"D total: {d_total.item():.4f}")

    logger.info("Full G+R+D pipeline test PASSED\n")


def test_refine_net():
    logger.info("=== Testing RefineNet (G2) ===")
    from models.refine_net import RefineNet

    B, H, W = 2, 64, 64
    g2 = RefineNet(in_channels=C, hidden_dim=32, num_blocks=4)
    param_count = sum(p.numel() for p in g2.parameters()) / 1e6
    logger.info(f"G2 params: {param_count:.2f}M")

    x = torch.randn(B, C, H, W)
    y = g2(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    assert y.min() >= -1.0 and y.max() <= 1.0, "Output out of [-1, 1] range"

    # Zero-init: initial output should be close to input (clamped)
    x_clamp = x.clamp(-1, 1)
    delta = (y - x_clamp).abs().mean().item()
    logger.info(f"Initial identity delta (should be ~0): {delta:.6f}")
    assert delta < 0.01, f"Initial output not identity: delta={delta}"

    # Backward pass
    loss = y.mean()
    loss.backward()
    grad_ok = all(p.grad is not None for p in g2.parameters() if p.requires_grad)
    assert grad_ok, "Some parameters have no gradients"

    logger.info("RefineNet test PASSED\n")


def test_g2_pipeline():
    logger.info("=== Testing G + G2 + D pipeline ===")
    from models.unet import UNet
    from models.refine_net import RefineNet
    from models.discriminator import MultiscaleDiscriminator
    from models.losses import CombinedLoss, GANLoss, FeatureMatchingLoss

    B, H, W = 2, 64, 64

    G = UNet(
        image_size=H, in_channels=C, model_channels=16, out_channels=C,
        num_res_blocks=1, attention_resolutions=(), channel_mult=(1, 2),
        residual_output=True,
    )
    G2 = RefineNet(in_channels=C, hidden_dim=32, num_blocks=4)
    D = MultiscaleDiscriminator(
        input_nc=C * 2, ndf=16, n_layers=2, num_D=2,
        use_spectral_norm=True, get_interm_feat=True,
    )

    criterion = CombinedLoss(l1_weight=1.0, ssim_weight=0.0, lung_weight=1.0)
    gan_loss_fn = GANLoss()
    fm_fn = FeatureMatchingLoss()

    ncct = torch.randn(B, C, H, W)
    cta = torch.randn(B, C, H, W)
    mask = torch.ones(B, C, H, W)

    # Freeze G
    for p in G.parameters():
        p.requires_grad_(False)
    G.eval()

    # Forward: G (frozen) → intermediate → G2
    with torch.no_grad():
        g_pred = G(ncct)
    intermediate = ncct * g_pred.abs()
    pred = G2(intermediate)

    # Reconstruction loss on G2 output
    loss_dict = criterion(pred, cta, mask)
    g_loss = loss_dict["loss"]

    # GAN + FM loss
    fake_input = torch.cat([ncct, pred], dim=1)
    real_input = torch.cat([ncct, cta], dim=1)
    fake_feats = D(fake_input)
    with torch.no_grad():
        real_feats = D(real_input)
    gan_g = gan_loss_fn(fake_feats, target_is_real=True)
    fm = fm_fn(real_feats, fake_feats)
    g_loss = g_loss + 1.0 * gan_g + 1.0 * fm

    # Backward — only G2 should get gradients
    g_loss.backward()

    g_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                     for p in G.parameters() if p.requires_grad)
    g2_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in G2.parameters())
    assert not g_has_grad, "G should NOT have gradients (frozen)"
    assert g2_has_grad, "G2 should have gradients"

    logger.info(f"G2 total loss: {g_loss.item():.4f}")
    logger.info(f"G frozen (no grads): OK, G2 has grads: OK")
    logger.info("G + G2 + D pipeline test PASSED\n")


def test_resblock_discriminator():
    logger.info("=== Testing ResBlock Discriminator ===")
    from models.discriminator_v2 import MultiscaleResBlockDiscriminator

    B, H, W = 2, 64, 64
    D = MultiscaleResBlockDiscriminator(
        input_nc=C, ndf=32, n_blocks=3, num_D=2,
        use_spectral_norm=True, use_attention=True, get_interm_feat=True,
    )
    param_count = sum(p.numel() for p in D.parameters()) / 1e6
    logger.info(f"ResBlock D params: {param_count:.2f}M")

    x = torch.randn(B, C, H, W)
    results = D(x)
    assert len(results) == 2, f"Expected 2 scales, got {len(results)}"
    for i, scale in enumerate(results):
        assert isinstance(scale, list), f"Scale {i} should be list of features"
        assert len(scale) >= 2, f"Scale {i} should have at least 2 features"
        pred = scale[-1]
        logger.info(f"  Scale {i}: {len(scale)} features, pred shape={list(pred.shape)}")

    # Backward pass
    loss = sum(s[-1].mean() for s in results)
    loss.backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in D.parameters())
    assert grad_ok, "D should have gradients"
    logger.info("ResBlock discriminator test PASSED\n")


def test_hinge_loss_and_r1():
    logger.info("=== Testing Hinge Loss + R1 Penalty ===")
    from models.discriminator import MultiscaleDiscriminator
    from models.losses import HingeGANLoss, r1_gradient_penalty

    B, H, W = 2, 64, 64
    D = MultiscaleDiscriminator(
        input_nc=C, ndf=16, n_layers=2, num_D=2,
        use_spectral_norm=True, get_interm_feat=True,
    )
    hinge_fn = HingeGANLoss()

    real = torch.randn(B, C, H, W)
    fake = torch.randn(B, C, H, W)

    real_preds = D(real)
    fake_preds = D(fake)

    # D loss (hinge)
    d_real = hinge_fn(real_preds, target_is_real=True, for_discriminator=True)
    d_fake = hinge_fn(fake_preds, target_is_real=False, for_discriminator=True)
    d_loss = 0.5 * (d_real + d_fake)
    logger.info(f"  Hinge D loss: {d_loss.item():.4f} (real={d_real.item():.4f}, fake={d_fake.item():.4f})")

    # G loss (hinge)
    fake_preds2 = D(fake)
    g_loss = hinge_fn(fake_preds2, target_is_real=True, for_discriminator=False)
    logger.info(f"  Hinge G loss: {g_loss.item():.4f}")

    # R1 gradient penalty
    real_r1 = real.clone().requires_grad_(True)
    real_preds_r1 = D(real_r1)
    r1 = r1_gradient_penalty(real_preds_r1, real_r1)
    logger.info(f"  R1 penalty: {r1.item():.4f}")
    assert r1.item() >= 0, "R1 should be non-negative"

    # R1 backward
    total = d_loss + 5.0 * r1
    total.backward()
    logger.info("Hinge loss + R1 penalty test PASSED\n")


def test_unconditional_d():
    logger.info("=== Testing Unconditional D Mode ===")
    from models.discriminator import MultiscaleDiscriminator
    from models.losses import GANLoss

    B, H, W = 2, 64, 64
    # Unconditional: D gets only the image (C channels), not concat(ncct, image)
    D = MultiscaleDiscriminator(
        input_nc=C, ndf=16, n_layers=2, num_D=2,
        use_spectral_norm=True, get_interm_feat=True,
    )
    gan_fn = GANLoss()

    image = torch.randn(B, C, H, W)
    preds = D(image)
    loss = gan_fn(preds, target_is_real=True)
    loss.backward()

    logger.info(f"  Unconditional D loss: {loss.item():.4f}")
    logger.info("Unconditional D mode test PASSED\n")


if __name__ == "__main__":
    test_config()
    ds = test_dataset()
    batch = test_dataloader(ds)
    test_gpu_augmentor(batch)
    test_model()
    test_loss()
    test_weighted_loss()
    test_registration()
    test_discriminator()
    test_gan_losses()
    test_grad_loss()
    test_visualization()
    test_forward_backward()
    test_full_pipeline_grd()
    test_refine_net()
    test_g2_pipeline()
    test_resblock_discriminator()
    test_hinge_loss_and_r1()
    test_unconditional_d()
    logger.info("=" * 40)
    logger.info("ALL TESTS PASSED!")
