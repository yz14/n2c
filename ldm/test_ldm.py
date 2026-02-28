"""
Smoke tests for all LDM modules.

Tests each component with dummy data to verify shapes, forward/backward
passes, and module interactions. Run with:
    python -m ldm.test_ldm
"""

import logging
import sys
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Use smaller sizes for testing to keep it fast
TEST_BATCH = 2
TEST_SLICES = 3       # num_slices (C) for 2.5D
TEST_SIZE = 64        # spatial resolution (smaller for speed)
TEST_Z_CH = 4         # latent channels
TEST_EMBED_DIM = 4    # embedding dim


def test_config():
    """Test LDM configuration save/load."""
    logger.info("=== Testing LDM Config ===")
    from ldm.config import LDMConfig

    cfg = LDMConfig()
    cfg.data.num_slices = TEST_SLICES
    cfg.sync_channels()
    assert cfg.vae.in_channels == TEST_SLICES
    assert cfg.vae.out_channels == TEST_SLICES

    # Save and reload
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        tmp_path = f.name
    try:
        cfg.save(tmp_path)
        cfg2 = LDMConfig.load(tmp_path)
        assert cfg2.vae.in_channels == TEST_SLICES
        assert cfg2.scheduler.num_train_timesteps == 1000
        assert cfg2.data.num_slices == TEST_SLICES
    finally:
        os.unlink(tmp_path)

    logger.info("Config test PASSED\n")


def test_distributions():
    """Test DiagonalGaussianDistribution."""
    logger.info("=== Testing DiagonalGaussianDistribution ===")
    from ldm.models.distributions import DiagonalGaussianDistribution

    B, C, H, W = TEST_BATCH, TEST_Z_CH, 8, 8
    params = torch.randn(B, 2 * C, H, W)
    dist = DiagonalGaussianDistribution(params)

    assert dist.mean.shape == (B, C, H, W)
    assert dist.logvar.shape == (B, C, H, W)

    z = dist.sample()
    assert z.shape == (B, C, H, W)

    kl = dist.kl()
    assert kl.shape == (B,)
    assert (kl >= 0).all(), "KL should be non-negative"

    nll = dist.nll(z)
    assert nll.shape == (B,)

    mode = dist.mode()
    assert torch.allclose(mode, dist.mean)

    # Deterministic mode
    det_dist = DiagonalGaussianDistribution(params, deterministic=True)
    z_det = det_dist.sample()
    assert torch.allclose(z_det, det_dist.mean)
    assert det_dist.kl().sum() == 0

    logger.info("DiagonalGaussianDistribution test PASSED\n")


def test_blocks():
    """Test building blocks."""
    logger.info("=== Testing Building Blocks ===")
    from ldm.models.blocks import (
        ResnetBlock, AttnBlock, MultiHeadAttnBlock,
        Downsample, Upsample, timestep_embedding,
    )

    B, C, H, W = TEST_BATCH, 64, 16, 16

    # TimestepEmbedding
    t = torch.randint(0, 1000, (B,))
    temb = timestep_embedding(t, 128)
    assert temb.shape == (B, 128), f"Expected (B, 128), got {temb.shape}"
    logger.info(f"  timestep_embedding: {temb.shape}")

    # ResnetBlock (no temb)
    x = torch.randn(B, C, H, W)
    resblock = ResnetBlock(in_channels=C, out_channels=C * 2)
    out = resblock(x)
    assert out.shape == (B, C * 2, H, W)
    logger.info(f"  ResnetBlock (no temb): {x.shape} -> {out.shape}")

    # ResnetBlock (with temb, scale_shift)
    temb_ch = 128
    resblock_t = ResnetBlock(
        in_channels=C * 2, out_channels=C * 2,
        temb_channels=temb_ch, use_scale_shift_norm=True,
    )
    temb_full = torch.randn(B, temb_ch)
    out_t = resblock_t(out, temb_full)
    assert out_t.shape == out.shape
    logger.info(f"  ResnetBlock (with temb): {out.shape} -> {out_t.shape}")

    # AttnBlock
    attn = AttnBlock(C * 2)
    out_a = attn(out)
    assert out_a.shape == out.shape
    logger.info(f"  AttnBlock: {out.shape} -> {out_a.shape}")

    # MultiHeadAttnBlock
    mha = MultiHeadAttnBlock(C * 2, num_heads=4)
    out_mha = mha(out)
    assert out_mha.shape == out.shape
    logger.info(f"  MultiHeadAttnBlock: {out.shape} -> {out_mha.shape}")

    # Downsample
    ds = Downsample(C * 2)
    out_ds = ds(out)
    assert out_ds.shape == (B, C * 2, H // 2, W // 2)
    logger.info(f"  Downsample: {out.shape} -> {out_ds.shape}")

    # Upsample
    us = Upsample(C * 2)
    out_us = us(out_ds)
    assert out_us.shape == out.shape
    logger.info(f"  Upsample: {out_ds.shape} -> {out_us.shape}")

    logger.info("Building blocks test PASSED\n")


def test_autoencoder():
    """Test AutoencoderKL."""
    logger.info("=== Testing AutoencoderKL ===")
    from ldm.config import VAEConfig
    from ldm.models.autoencoder import AutoencoderKL

    cfg = VAEConfig(
        in_channels=TEST_SLICES,
        out_channels=TEST_SLICES,
        z_channels=TEST_Z_CH,
        embed_dim=TEST_EMBED_DIM,
        ch=32,                          # smaller for test
        ch_mult=(1, 2, 4),             # 3 levels, 2 downsamples -> f=4
        num_res_blocks=1,
        attn_resolutions=(16,),         # attention at 16×16
        resolution=TEST_SIZE,
    )

    vae = AutoencoderKL(cfg)
    n_params = sum(p.numel() for p in vae.parameters())
    logger.info(f"  VAE params: {n_params / 1e6:.2f}M")

    x = torch.randn(TEST_BATCH, TEST_SLICES, TEST_SIZE, TEST_SIZE)

    # Encode
    posterior = vae.encode(x)
    z = posterior.sample()
    expected_lat = TEST_SIZE // 4  # f=4 for ch_mult=(1,2,4)
    assert z.shape == (TEST_BATCH, TEST_EMBED_DIM, expected_lat, expected_lat), \
        f"Expected latent ({TEST_BATCH}, {TEST_EMBED_DIM}, {expected_lat}, {expected_lat}), got {z.shape}"
    logger.info(f"  Encode: {x.shape} -> z: {z.shape}")

    # Decode
    x_recon = vae.decode(z)
    assert x_recon.shape == x.shape, f"Expected {x.shape}, got {x_recon.shape}"
    logger.info(f"  Decode: {z.shape} -> {x_recon.shape}")

    # Full forward
    x_recon2, posterior2 = vae(x)
    assert x_recon2.shape == x.shape
    kl = posterior2.kl().mean()
    logger.info(f"  Full forward: recon={x_recon2.shape}, KL={kl.item():.4f}")

    # Backward
    loss = (x_recon2 - x).pow(2).mean() + 1e-6 * kl
    loss.backward()
    logger.info(f"  Backward: loss={loss.item():.4f}")

    logger.info("AutoencoderKL test PASSED\n")


def test_diffusion_unet():
    """Test DiffusionUNet."""
    logger.info("=== Testing DiffusionUNet ===")
    from ldm.models.unet import DiffusionUNet

    lat_size = TEST_SIZE // 4  # latent spatial size (after VAE f=4)

    unet = DiffusionUNet(
        in_channels=2 * TEST_Z_CH,       # noisy_z + condition_z
        out_channels=TEST_Z_CH,
        model_channels=32,                # smaller for test
        channel_mult=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(2, 4),     # attention at 2x and 4x downsample
        num_heads=2,
        use_scale_shift_norm=True,
    )
    n_params = sum(p.numel() for p in unet.parameters())
    logger.info(f"  UNet params: {n_params / 1e6:.2f}M")

    z_noisy = torch.randn(TEST_BATCH, TEST_Z_CH, lat_size, lat_size)
    z_cond = torch.randn(TEST_BATCH, TEST_Z_CH, lat_size, lat_size)
    t = torch.randint(0, 1000, (TEST_BATCH,))

    x_in = torch.cat([z_noisy, z_cond], dim=1)
    noise_pred = unet(x_in, t)

    assert noise_pred.shape == z_noisy.shape, \
        f"Expected {z_noisy.shape}, got {noise_pred.shape}"
    logger.info(f"  Forward: input={x_in.shape}, t={t.shape} -> {noise_pred.shape}")

    # Backward
    loss = noise_pred.pow(2).mean()
    loss.backward()
    logger.info(f"  Backward: loss={loss.item():.4f}")

    logger.info("DiffusionUNet test PASSED\n")


def test_scheduler():
    """Test DDPM and DDIM schedulers."""
    logger.info("=== Testing Schedulers ===")
    from ldm.diffusion.scheduler import DDPMScheduler, DDIMScheduler

    ddpm = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    assert ddpm.betas.shape == (1000,)
    assert ddpm.alphas_cumprod.shape == (1000,)
    assert ddpm.alphas_cumprod[0] > ddpm.alphas_cumprod[-1], "alphas_cumprod should decrease"

    # Test add_noise
    x0 = torch.randn(TEST_BATCH, TEST_Z_CH, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 500])

    x_t = ddpm.add_noise(x0, noise, t)
    assert x_t.shape == x0.shape
    logger.info(f"  DDPM add_noise: {x0.shape} -> {x_t.shape}")

    # At t=0, x_t should be very close to x0 (sqrt(1-ᾱ₀) ≈ 0.029)
    t_zero = torch.tensor([0, 0])
    x_t0 = ddpm.add_noise(x0, noise, t_zero)
    diff_t0 = (x_t0 - x0).abs().mean().item()
    diff_t500 = (x_t - x0).abs().mean().item()
    assert diff_t0 < diff_t500, \
        f"At t=0 diff ({diff_t0:.4f}) should be < t=500 diff ({diff_t500:.4f})"
    logger.info(f"  Noise level: t=0 diff={diff_t0:.4f}, t=500 diff={diff_t500:.4f}")

    # Test DDPM step
    pred_noise = torch.randn_like(x0)
    x_prev = ddpm.ddpm_step(pred_noise, 500, x_t)
    assert x_prev.shape == x0.shape
    logger.info(f"  DDPM step: {x_t.shape} -> {x_prev.shape}")

    # Test DDIM
    ddim = DDIMScheduler(ddpm, num_inference_steps=50, eta=0.0)
    assert len(ddim.timesteps) == 50
    logger.info(f"  DDIM timesteps: {ddim.timesteps[:5]}... (total {len(ddim.timesteps)})")

    x_prev_ddim, pred_x0 = ddim.step(pred_noise, 0, x_t)
    assert x_prev_ddim.shape == x0.shape
    assert pred_x0.shape == x0.shape
    logger.info(f"  DDIM step: {x_t.shape} -> {x_prev_ddim.shape}")

    logger.info("Schedulers test PASSED\n")


def test_pipeline():
    """Test full ConditionalLDMPipeline (shapes only, untrained)."""
    logger.info("=== Testing ConditionalLDMPipeline ===")
    from ldm.config import VAEConfig, DiffusionUNetConfig, SchedulerConfig
    from ldm.models.autoencoder import AutoencoderKL
    from ldm.models.unet import DiffusionUNet
    from ldm.diffusion.scheduler import DDPMScheduler
    from ldm.diffusion.pipeline import ConditionalLDMPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    # Build small models for testing
    vae_cfg = VAEConfig(
        in_channels=TEST_SLICES, out_channels=TEST_SLICES,
        z_channels=TEST_Z_CH, embed_dim=TEST_EMBED_DIM,
        ch=32, ch_mult=(1, 2, 4), num_res_blocks=1,
        attn_resolutions=(16,), resolution=TEST_SIZE,
    )
    vae = AutoencoderKL(vae_cfg).to(device).eval()

    unet = DiffusionUNet(
        in_channels=2 * TEST_Z_CH, out_channels=TEST_Z_CH,
        model_channels=32, channel_mult=(1, 2, 4),
        num_res_blocks=1, attention_resolutions=(2, 4),
        num_heads=2, use_scale_shift_norm=True,
    ).to(device).eval()

    scheduler = DDPMScheduler(num_train_timesteps=100)  # fewer steps for speed
    scheduler.to(device)

    pipeline = ConditionalLDMPipeline(vae, unet, scheduler)

    ncct = torch.randn(1, TEST_SLICES, TEST_SIZE, TEST_SIZE, device=device)

    # Test DDIM sampling (few steps)
    cta_pred = pipeline.sample(ncct, num_inference_steps=5, verbose=False)
    assert cta_pred.shape == ncct.shape, \
        f"Expected {ncct.shape}, got {cta_pred.shape}"
    logger.info(f"  DDIM sample: {ncct.shape} -> {cta_pred.shape}")

    # Test with intermediates
    cta_pred2, intermediates = pipeline.sample(
        ncct, num_inference_steps=5, return_intermediates=True, verbose=False,
    )
    assert len(intermediates) == 5
    logger.info(f"  Intermediates: {len(intermediates)} steps")

    logger.info("ConditionalLDMPipeline test PASSED\n")


def test_training_step():
    """Test a single training step (forward + backward) for diffusion."""
    logger.info("=== Testing Training Step (Diffusion) ===")
    from ldm.config import VAEConfig
    from ldm.models.autoencoder import AutoencoderKL
    from ldm.models.unet import DiffusionUNet
    from ldm.diffusion.scheduler import DDPMScheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build small models
    vae_cfg = VAEConfig(
        in_channels=TEST_SLICES, out_channels=TEST_SLICES,
        z_channels=TEST_Z_CH, embed_dim=TEST_EMBED_DIM,
        ch=32, ch_mult=(1, 2, 4), num_res_blocks=1,
        attn_resolutions=(16,), resolution=TEST_SIZE,
    )
    vae = AutoencoderKL(vae_cfg).to(device).eval()

    unet = DiffusionUNet(
        in_channels=2 * TEST_Z_CH, out_channels=TEST_Z_CH,
        model_channels=32, channel_mult=(1, 2, 4),
        num_res_blocks=1, attention_resolutions=(2, 4),
        num_heads=2, use_scale_shift_norm=True,
    ).to(device).train()

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.to(device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    # Simulate training data
    ncct = torch.randn(TEST_BATCH, TEST_SLICES, TEST_SIZE, TEST_SIZE, device=device)
    cta = torch.randn(TEST_BATCH, TEST_SLICES, TEST_SIZE, TEST_SIZE, device=device)

    # Encode to latent space (VAE frozen)
    with torch.no_grad():
        z_ncct = vae.encode(ncct).mode()
        z_cta = vae.encode(cta).mode()

    # Add noise
    noise = torch.randn_like(z_cta)
    t = torch.randint(0, 1000, (TEST_BATCH,), device=device)
    z_noisy = scheduler.add_noise(z_cta, noise, t)

    # Forward
    model_input = torch.cat([z_noisy, z_ncct], dim=1)
    noise_pred = unet(model_input, t)

    # Loss (simple MSE on noise prediction)
    loss = torch.nn.functional.mse_loss(noise_pred, noise)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
    optimizer.step()

    logger.info(f"  Loss: {loss.item():.4f}, Grad norm: {grad_norm.item():.4f}")
    logger.info("Training step test PASSED\n")


def main():
    test_config()
    test_distributions()
    test_blocks()
    test_autoencoder()
    test_diffusion_unet()
    test_scheduler()
    test_pipeline()
    test_training_step()

    logger.info("=" * 50)
    logger.info("ALL LDM TESTS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
