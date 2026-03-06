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


def test_vae_gan_config():
    """Test VAEGANConfig serialization/deserialization."""
    logger.info("=== Testing VAEGANConfig ====")
    from ldm.config import LDMConfig, VAEGANConfig
    import tempfile, os

    cfg = LDMConfig()
    cfg.vae_gan.enabled = True
    cfg.vae_gan.gan_weight = 0.5
    cfg.vae_gan.perceptual_weight = 1.0
    cfg.vae_gan.disc_start_epoch = 3

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        tmp_path = f.name
    try:
        cfg.save(tmp_path)
        cfg2 = LDMConfig.load(tmp_path)
        assert cfg2.vae_gan.enabled == True
        assert cfg2.vae_gan.gan_weight == 0.5
        assert cfg2.vae_gan.perceptual_weight == 1.0
        assert cfg2.vae_gan.disc_start_epoch == 3
        assert cfg2.vae_gan.gan_loss_type == "hinge"
        assert cfg2.vae_gan.adaptive_weight == True
    finally:
        os.unlink(tmp_path)

    # Test default (disabled)
    cfg3 = LDMConfig()
    assert cfg3.vae_gan.enabled == False

    logger.info("VAEGANConfig test PASSED\n")


def test_vae_gan_loss():
    """Test VAEGANLoss: creation, G loss, D loss, gradient isolation."""
    logger.info("=== Testing VAEGANLoss ====")
    from ldm.config import VAEConfig, VAEGANConfig
    from ldm.models.autoencoder import AutoencoderKL
    from ldm.vae_gan_loss import VAEGANLoss

    device = torch.device("cpu")  # CPU for testing

    # Build small VAE
    vae_cfg = VAEConfig(
        in_channels=TEST_SLICES, out_channels=TEST_SLICES,
        z_channels=TEST_Z_CH, embed_dim=TEST_EMBED_DIM,
        ch=32, ch_mult=(1, 2, 4), num_res_blocks=1,
        attn_resolutions=(16,), resolution=TEST_SIZE,
    )
    vae = AutoencoderKL(vae_cfg).to(device)

    # Create VAEGANLoss with GAN enabled (no perceptual for CPU speed)
    gan_cfg = VAEGANConfig(
        enabled=True,
        ndf=16,  # small for test
        n_layers=2,
        use_spectral_norm=False,  # faster on CPU
        gan_loss_type="hinge",
        gan_weight=0.5,
        adaptive_weight=True,
        adaptive_weight_max=10.0,
        perceptual_weight=0.0,  # skip VGG for CPU test
        lr=1e-4,
        disc_start_epoch=0,  # active from start
    )

    vae_gan = VAEGANLoss(gan_cfg, vae, device, total_steps=100)
    assert vae_gan.discriminator is not None
    assert vae_gan.is_active(0) == True
    logger.info(f"  D params: {sum(p.numel() for p in vae_gan.discriminator.parameters()) / 1e6:.2f}M")

    # Test with disc_start_epoch > 0
    gan_cfg2 = VAEGANConfig(enabled=True, disc_start_epoch=5, ndf=16, n_layers=2,
                            use_spectral_norm=False, perceptual_weight=0.0)
    vae_gan2 = VAEGANLoss(gan_cfg2, vae, device, total_steps=100)
    assert vae_gan2.is_active(4) == False
    assert vae_gan2.is_active(5) == True

    # Test disabled
    gan_cfg_off = VAEGANConfig(enabled=False)
    vae_gan_off = VAEGANLoss(gan_cfg_off, vae, device)
    assert vae_gan_off.discriminator is None
    assert vae_gan_off.is_active(0) == False

    # ---- Test G loss computation ----
    x = torch.randn(TEST_BATCH, TEST_SLICES, TEST_SIZE, TEST_SIZE)
    recon, posterior = vae(x)
    nll_loss = torch.nn.functional.l1_loss(recon, x)

    g_result = vae_gan.compute_g_loss(
        recon=recon, target=x, posterior=posterior,
        epoch=0, nll_loss=nll_loss,
    )
    assert "g_loss" in g_result
    assert "gan_g" in g_result
    assert "adaptive_w" in g_result
    assert g_result["disc_factor"].item() == 1.0
    logger.info(f"  G loss: {g_result['g_loss'].item():.4f}, "
                f"GAN_g: {g_result['gan_g'].item():.4f}, "
                f"adaptive_w: {g_result['adaptive_w'].item():.4f}")

    # ---- Test D loss computation ----
    d_result = vae_gan.compute_d_loss(
        recon=recon.detach(), target=x.detach(), epoch=0,
    )
    assert "d_loss" in d_result
    assert "d_real" in d_result
    assert "d_fake" in d_result
    assert d_result["d_loss"].requires_grad  # should have grad for backprop
    logger.info(f"  D loss: {d_result['d_loss'].item():.4f}, "
                f"D_real: {d_result['d_real'].item():.4f}, "
                f"D_fake: {d_result['d_fake'].item():.4f}")

    # ---- Test gradient isolation ----
    # After G update, D params should have no grad
    vae.zero_grad()
    recon2, posterior2 = vae(x)
    nll2 = torch.nn.functional.l1_loss(recon2, x)
    g_result2 = vae_gan.compute_g_loss(
        recon=recon2, target=x, posterior=posterior2,
        epoch=0, nll_loss=nll2,
    )
    total_g = nll2 + g_result2["g_loss"]
    total_g.backward()

    # D should NOT have accumulated gradients from G backward
    d_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in vae_gan.discriminator.parameters()
    )
    assert not d_has_grad, "Gradient leaked from G to D!"

    # VAE SHOULD have gradients
    vae_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in vae.parameters()
    )
    assert vae_has_grad, "VAE should have gradients after G backward"
    logger.info("  Gradient isolation: PASSED (no D grad leak)")

    # ---- Test D step ----
    d_result3 = vae_gan.compute_d_loss(
        recon=recon2.detach(), target=x.detach(), epoch=0,
    )
    vae_gan.step_d(d_result3["d_loss"])
    logger.info("  D optimizer step: PASSED")

    # ---- Test state dict ----
    state = vae_gan.state_dict()
    assert "discriminator_state_dict" in state
    assert "optimizer_d_state_dict" in state
    assert "d_step_count" in state
    logger.info("  State dict: PASSED")

    logger.info("VAEGANLoss test PASSED\n")


def test_vae_gan_training_step():
    """Test a complete VAE+GAN training step (forward + backward for both)."""
    logger.info("=== Testing VAE+GAN Training Step ====")
    from ldm.config import VAEConfig, VAEGANConfig
    from ldm.models.autoencoder import AutoencoderKL
    from ldm.vae_gan_loss import VAEGANLoss

    device = torch.device("cpu")

    vae_cfg = VAEConfig(
        in_channels=TEST_SLICES, out_channels=TEST_SLICES,
        z_channels=TEST_Z_CH, embed_dim=TEST_EMBED_DIM,
        ch=32, ch_mult=(1, 2, 4), num_res_blocks=1,
        attn_resolutions=(16,), resolution=TEST_SIZE,
    )
    vae = AutoencoderKL(vae_cfg).to(device)

    gan_cfg = VAEGANConfig(
        enabled=True, ndf=16, n_layers=2,
        use_spectral_norm=False, gan_loss_type="hinge",
        gan_weight=0.5, adaptive_weight=True,
        perceptual_weight=0.0, disc_start_epoch=0,
    )
    vae_gan = VAEGANLoss(gan_cfg, vae, device, total_steps=100)

    optimizer_vae = torch.optim.AdamW(vae.parameters(), lr=1e-4)
    x = torch.randn(TEST_BATCH, TEST_SLICES, TEST_SIZE, TEST_SIZE)

    # Record initial weights
    vae_w0 = vae.decoder.conv_out.weight.data.clone()
    d_w0 = list(vae_gan.discriminator.parameters())[0].data.clone()

    # --- Phase 1: VAE update ---
    vae.train()
    recon, posterior = vae(x)
    l1_loss = torch.nn.functional.l1_loss(recon, x)
    kl_loss = posterior.kl().mean()
    recon_loss = l1_loss + 1e-6 * kl_loss

    g_loss_dict = vae_gan.compute_g_loss(
        recon=recon, target=x, posterior=posterior,
        epoch=0, nll_loss=recon_loss,
    )
    total_loss = recon_loss + g_loss_dict["g_loss"]

    optimizer_vae.zero_grad()
    total_loss.backward()
    optimizer_vae.step()

    # --- Phase 2: D update ---
    for p in vae.parameters():
        p.requires_grad = False
    d_loss_dict = vae_gan.compute_d_loss(
        recon=recon.detach(), target=x.detach(), epoch=0,
    )
    # Override D LR to non-zero (warmup scheduler starts at 0 for step 0)
    vae_gan.optimizer_d.param_groups[0]["lr"] = 0.1
    vae_gan.step_d(d_loss_dict["d_loss"])
    for p in vae.parameters():
        p.requires_grad = True

    # Verify weights changed
    vae_w1 = vae.decoder.conv_out.weight.data
    d_w1 = list(vae_gan.discriminator.parameters())[0].data
    assert not torch.allclose(vae_w0, vae_w1), "VAE weights should have changed"
    assert not torch.allclose(d_w0, d_w1), "D weights should have changed"

    logger.info(f"  VAE loss: {total_loss.item():.4f}")
    logger.info(f"  D loss: {d_loss_dict['d_loss'].item():.4f}")
    logger.info(f"  Both VAE and D weights updated correctly")
    logger.info("VAE+GAN training step test PASSED\n")


def test_vae_gan_lsgan():
    """Test VAEGANLoss with LSGAN loss type."""
    logger.info("=== Testing VAEGANLoss (LSGAN) ====")
    from ldm.config import VAEConfig, VAEGANConfig
    from ldm.models.autoencoder import AutoencoderKL
    from ldm.vae_gan_loss import VAEGANLoss

    device = torch.device("cpu")
    vae_cfg = VAEConfig(
        in_channels=TEST_SLICES, out_channels=TEST_SLICES,
        z_channels=TEST_Z_CH, embed_dim=TEST_EMBED_DIM,
        ch=32, ch_mult=(1, 2, 4), num_res_blocks=1,
        attn_resolutions=(16,), resolution=TEST_SIZE,
    )
    vae = AutoencoderKL(vae_cfg).to(device)

    gan_cfg = VAEGANConfig(
        enabled=True, ndf=16, n_layers=2,
        use_spectral_norm=False, gan_loss_type="lsgan",
        perceptual_weight=0.0, disc_start_epoch=0,
    )
    vae_gan = VAEGANLoss(gan_cfg, vae, device, total_steps=100)

    x = torch.randn(TEST_BATCH, TEST_SLICES, TEST_SIZE, TEST_SIZE)
    recon, posterior = vae(x)
    nll = torch.nn.functional.l1_loss(recon, x)

    g_result = vae_gan.compute_g_loss(recon=recon, target=x, posterior=posterior,
                                       epoch=0, nll_loss=nll)
    d_result = vae_gan.compute_d_loss(recon=recon.detach(), target=x.detach(), epoch=0)

    assert g_result["g_loss"].item() >= 0
    assert d_result["d_loss"].item() >= 0
    logger.info(f"  LSGAN G loss: {g_result['gan_g'].item():.4f}, D loss: {d_result['d_loss'].item():.4f}")
    logger.info("VAEGANLoss (LSGAN) test PASSED\n")


def test_latent_scale_factor():
    """Test latent_scale_factor in config, pipeline, and scheduler."""
    logger.info("=== Testing Latent Scale Factor ===")
    from ldm.config import LDMConfig, VAEConfig, SchedulerConfig
    from ldm.models.autoencoder import AutoencoderKL
    from ldm.diffusion.scheduler import DDPMScheduler
    from ldm.diffusion.pipeline import ConditionalLDMPipeline
    from ldm.models.unet import DiffusionUNet
    import tempfile, os

    # 1. Config round-trip
    cfg = LDMConfig()
    cfg.scheduler.latent_scale_factor = 0.35
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        tmp_path = f.name
    try:
        cfg.save(tmp_path)
        cfg2 = LDMConfig.load(tmp_path)
        assert abs(cfg2.scheduler.latent_scale_factor - 0.35) < 1e-6, \
            f"Scale factor mismatch: {cfg2.scheduler.latent_scale_factor}"
    finally:
        os.unlink(tmp_path)
    logger.info("  Config round-trip: PASSED")

    # 2. Pipeline applies scaling correctly
    device = torch.device("cpu")
    vae_cfg = VAEConfig(
        in_channels=TEST_SLICES, out_channels=TEST_SLICES,
        z_channels=TEST_Z_CH, embed_dim=TEST_EMBED_DIM,
        ch=32, ch_mult=(1, 2, 4), num_res_blocks=1,
        attn_resolutions=(16,), resolution=TEST_SIZE,
    )
    vae = AutoencoderKL(vae_cfg).to(device).eval()

    scale = 0.4
    x = torch.randn(1, TEST_SLICES, TEST_SIZE, TEST_SIZE)

    # Encode without scaling
    with torch.no_grad():
        z_raw = vae.encode(x).mode()

    # Pipeline with scaling
    scheduler = DDPMScheduler(num_train_timesteps=10, beta_schedule="linear")
    unet = DiffusionUNet.from_config(
        cfg.unet, z_channels=TEST_EMBED_DIM,
    ).to(device)
    pipeline = ConditionalLDMPipeline(
        vae, unet, scheduler, latent_scale_factor=scale,
    )
    z_scaled = pipeline.encode_condition(x)
    assert torch.allclose(z_scaled, z_raw * scale, atol=1e-5), \
        "Pipeline encode_condition should apply scale factor"
    logger.info(f"  Pipeline scaling: PASSED (raw_std={z_raw.std():.3f}, scaled_std={z_scaled.std():.3f})")

    # 3. Default scale_factor=0 in config
    cfg3 = LDMConfig()
    assert cfg3.scheduler.latent_scale_factor == 0.0
    logger.info("  Default value (0.0 = auto): PASSED")

    logger.info("Latent scale factor test PASSED\n")


def test_d_negative_samples():
    """Test D negative sample augmentation."""
    logger.info("=== Testing D Negative Samples ===")
    from ldm.config import VAEConfig, VAEGANConfig
    from ldm.models.autoencoder import AutoencoderKL
    from ldm.vae_gan_loss import VAEGANLoss

    device = torch.device("cpu")
    vae_cfg = VAEConfig(
        in_channels=TEST_SLICES, out_channels=TEST_SLICES,
        z_channels=TEST_Z_CH, embed_dim=TEST_EMBED_DIM,
        ch=32, ch_mult=(1, 2, 4), num_res_blocks=1,
        attn_resolutions=(16,), resolution=TEST_SIZE,
    )
    vae = AutoencoderKL(vae_cfg).to(device)

    # Test with neg augment disabled
    gan_cfg_off = VAEGANConfig(
        enabled=True, ndf=16, n_layers=2,
        use_spectral_norm=False, perceptual_weight=0.0,
        disc_start_epoch=0, d_neg_augment=False,
    )
    vae_gan_off = VAEGANLoss(gan_cfg_off, vae, device, total_steps=100)
    ncct = torch.randn(TEST_BATCH, TEST_SLICES, TEST_SIZE, TEST_SIZE)
    cta = torch.randn(TEST_BATCH, TEST_SLICES, TEST_SIZE, TEST_SIZE)
    assert vae_gan_off.generate_negative_samples(ncct, cta) is None
    logger.info("  Disabled: returns None — PASSED")

    # Test with neg augment enabled
    gan_cfg_on = VAEGANConfig(
        enabled=True, ndf=16, n_layers=2,
        use_spectral_norm=False, perceptual_weight=0.0,
        disc_start_epoch=0, d_neg_augment=True,
    )
    vae_gan_on = VAEGANLoss(gan_cfg_on, vae, device, total_steps=100)
    neg = vae_gan_on.generate_negative_samples(ncct, cta)
    assert neg is not None
    assert neg.shape == cta.shape
    logger.info(f"  Enabled: shape={neg.shape} — PASSED")

    # Test compute_d_loss with negative samples
    recon, _ = vae(cta)
    d_result = vae_gan_on.compute_d_loss(
        recon=recon.detach(), target=cta.detach(),
        epoch=0, neg_samples=neg,
    )
    assert "d_neg" in d_result
    assert d_result["d_loss"].requires_grad
    logger.info(f"  D loss with neg samples: {d_result['d_loss'].item():.4f}, "
                f"d_neg: {d_result['d_neg'].item():.4f} — PASSED")

    logger.info("D negative samples test PASSED\n")


def test_shared_utils():
    """Test shared MetricTracker from ldm.utils."""
    logger.info("=== Testing Shared Utils ===")
    from ldm.utils import MetricTracker, warmup_cosine_schedule

    # MetricTracker
    tracker = MetricTracker()
    tracker.update({"loss": 1.0, "lr": 0.01})
    tracker.update({"loss": 2.0, "lr": 0.01})
    result = tracker.result()
    assert abs(result["loss"] - 1.5) < 1e-6
    assert abs(result["lr"] - 0.01) < 1e-6
    logger.info(f"  MetricTracker: {tracker}")

    tracker.reset()
    assert len(tracker.result()) == 0
    logger.info("  MetricTracker reset: PASSED")

    # warmup_cosine_schedule
    lr_fn = warmup_cosine_schedule(warmup_steps=10, total_steps=100)
    assert abs(lr_fn(0) - 0.0) < 1e-6  # step 0 = 0
    assert abs(lr_fn(5) - 0.5) < 1e-6  # step 5 = 0.5 (warmup)
    assert abs(lr_fn(10) - 1.0) < 1e-6  # step 10 = 1.0 (end warmup)
    assert lr_fn(100) < 0.01  # near end = near 0
    logger.info("  warmup_cosine_schedule: PASSED")

    logger.info("Shared utils test PASSED\n")


def test_ddpm_no_clamp():
    """Test that DDPM step does NOT clamp pred_x0 (BUG-2 fix)."""
    logger.info("=== Testing DDPM No Clamp (BUG-2 Fix) ===")
    from ldm.diffusion.scheduler import DDPMScheduler

    scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="linear")

    # Create a sample with values outside [-1, 1] (like latent space)
    sample = torch.randn(1, 4, 8, 8) * 3.0  # std ~3, values in [-9, 9]
    noise_pred = torch.randn(1, 4, 8, 8)

    # DDPM step at t=50 (mid-schedule)
    result = scheduler.ddpm_step(noise_pred, 50, sample)

    # The result should NOT be clamped to [-1, 1]
    # With input std~3, output should also have values outside [-1, 1]
    assert result.abs().max() > 1.0, \
        "DDPM step should not clamp latent values to [-1, 1]"
    logger.info(f"  Output range: [{result.min():.2f}, {result.max():.2f}] (not clamped)")
    logger.info("DDPM no clamp test PASSED\n")


def main():
    test_config()
    test_distributions()
    test_blocks()
    test_autoencoder()
    test_diffusion_unet()
    test_scheduler()
    test_pipeline()
    test_training_step()
    test_vae_gan_config()
    test_vae_gan_loss()
    test_vae_gan_training_step()
    test_vae_gan_lsgan()
    test_latent_scale_factor()
    test_d_negative_samples()
    test_shared_utils()
    test_ddpm_no_clamp()

    logger.info("=" * 50)
    logger.info("ALL LDM TESTS PASSED! (16 tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
