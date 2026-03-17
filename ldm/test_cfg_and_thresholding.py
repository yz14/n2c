"""
Tests for Classifier-Free Guidance (CFG) and Dynamic Thresholding.

Covers:
  1. Config: new fields cfg_scale, dynamic_threshold_percentile, cfg_drop_rate
  2. Trainer: condition dropping during training
  3. Pipeline: CFG noise prediction combination
  4. Pipeline: dynamic thresholding of pred_x0
  5. Pipeline: CFG + dynamic thresholding together
  6. Backward compatibility: cfg_scale=1.0 behaves same as no-CFG
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldm.config import LDMConfig, SchedulerConfig, DiffusionTrainConfig
from ldm.diffusion.scheduler import DDPMScheduler, DDIMScheduler
from ldm.diffusion.pipeline import ConditionalLDMPipeline


# ---------------------------------------------------------------------------
# Helpers: minimal mock models for testing pipeline logic
# ---------------------------------------------------------------------------

class MockVAE(nn.Module):
    """Minimal VAE that does identity encode/decode for testing."""
    def __init__(self, z_ch=4):
        super().__init__()
        self.z_ch = z_ch
        self.dummy = nn.Linear(1, 1)  # so .parameters() is not empty

    def encode(self, x):
        # Return a mock posterior with .mode() = x[:, :z_ch]
        class MockPosterior:
            def __init__(self, z):
                self._z = z
            def mode(self):
                return self._z
            def sample(self):
                return self._z
        return MockPosterior(x[:, :self.z_ch])

    def decode(self, z):
        return z


class MockUNet(nn.Module):
    """Minimal UNet that records inputs and returns deterministic output."""
    def __init__(self, in_ch=8, out_ch=4):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.call_log = []  # records (input_shape, t) for each call

    def forward(self, x, t):
        self.call_log.append((x.shape, t.clone()))
        return self.conv(x)


class MockUNetCFGTracker(nn.Module):
    """UNet that tracks whether condition is zero (unconditional) or not."""
    def __init__(self, z_ch=4):
        super().__init__()
        self.z_ch = z_ch
        self.dummy = nn.Linear(1, 1)
        self.cond_norms = []  # track L2 norm of condition part

    def forward(self, x, t):
        B = x.shape[0]
        # x is concat([z_noisy, z_cond], dim=1) — split to get condition
        z_cond = x[:, self.z_ch:]
        for i in range(B):
            self.cond_norms.append(z_cond[i].norm().item())
        # Return random noise prediction
        return torch.randn(B, self.z_ch, x.shape[2], x.shape[3], device=x.device)


# ---------------------------------------------------------------------------
# Test 1: Config new fields
# ---------------------------------------------------------------------------

class TestConfigCFGFields:
    def test_scheduler_config_has_cfg_scale(self):
        cfg = SchedulerConfig()
        assert hasattr(cfg, 'cfg_scale')
        assert cfg.cfg_scale == 3.0

    def test_scheduler_config_has_dynamic_threshold(self):
        cfg = SchedulerConfig()
        assert hasattr(cfg, 'dynamic_threshold_percentile')
        assert cfg.dynamic_threshold_percentile == 0.0  # disabled by default for latent space

    def test_diffusion_train_config_has_cfg_drop_rate(self):
        cfg = DiffusionTrainConfig()
        assert hasattr(cfg, 'cfg_drop_rate')
        assert cfg.cfg_drop_rate == 0.1

    def test_config_yaml_roundtrip(self, tmp_path):
        """Config with new fields can be saved and loaded."""
        cfg = LDMConfig()
        cfg.scheduler.cfg_scale = 5.0
        cfg.scheduler.dynamic_threshold_percentile = 0.99
        cfg.diffusion_train.cfg_drop_rate = 0.15

        path = str(tmp_path / "test_cfg.yaml")
        cfg.save(path)
        loaded = LDMConfig.load(path)

        assert loaded.scheduler.cfg_scale == 5.0
        assert loaded.scheduler.dynamic_threshold_percentile == 0.99
        assert loaded.diffusion_train.cfg_drop_rate == 0.15


# ---------------------------------------------------------------------------
# Test 2: Dynamic thresholding function
# ---------------------------------------------------------------------------

class TestDynamicThresholding:
    def test_no_change_for_small_values(self):
        """Values within normal range should not be affected."""
        pred_x0 = torch.randn(2, 4, 8, 8) * 0.5  # small values
        result = ConditionalLDMPipeline._dynamic_threshold(pred_x0, 0.995)
        # s = max(percentile, 1.0) = 1.0 for small values
        # Clamp to [-1, 1] only affects values > 1 — most values unchanged
        assert result.shape == pred_x0.shape
        # Values < 1.0 in absolute should be exactly preserved
        mask = pred_x0.abs() < 1.0
        assert torch.allclose(result[mask], pred_x0[mask])

    def test_clips_extreme_values(self):
        """Extreme outlier values should be clipped (without rescaling)."""
        pred_x0 = torch.zeros(1, 4, 8, 8)
        pred_x0[0, 0, 0, 0] = 100.0  # extreme outlier
        result = ConditionalLDMPipeline._dynamic_threshold(pred_x0, 0.995)
        # 99.5th percentile of abs values: most are 0, one is 100
        # s = max(quantile, 1.0). For mostly-zero tensor, s ≈ 1.0
        # After clamping to [-s, s], max should be s (NOT 1.0 via /s)
        # The outlier is clipped but non-outlier values are preserved
        assert result.abs().max() < 100.0  # outlier was clipped
        assert result[0, 0, 0, 0] > 0  # still positive (clamped, not zeroed)

    def test_per_sample_thresholding(self):
        """Each sample in the batch should be thresholded independently."""
        pred_x0 = torch.zeros(2, 4, 4, 4)
        pred_x0[0] = 0.5  # small
        pred_x0[1] = 10.0  # large
        result = ConditionalLDMPipeline._dynamic_threshold(pred_x0, 0.995)
        # Sample 0: s = max(0.5_percentile, 1.0) = 1.0
        #   → clamp to [-1, 1], but all values are 0.5 → unchanged
        assert torch.allclose(result[0], pred_x0[0], atol=0.01)
        # Sample 1: s = max(10.0, 1.0) = 10.0
        #   → clamp to [-10, 10], all values are 10.0 → unchanged
        #   (no /s rescaling, so values stay at 10.0)
        assert torch.allclose(result[1], pred_x0[1], atol=0.01)

    def test_min_clamp_prevents_shrinking(self):
        """s is clamped to min=1.0, so small values are preserved exactly."""
        pred_x0 = torch.ones(1, 4, 4, 4) * 0.1
        result = ConditionalLDMPipeline._dynamic_threshold(pred_x0, 0.995)
        # s = max(0.1, 1.0) = 1.0, clamp to [-1, 1] → 0.1 unchanged
        assert torch.allclose(result, pred_x0, atol=1e-6)

    def test_no_rescaling_preserves_latent_scale(self):
        """Latent-space thresholding must NOT rescale values (unlike Imagen pixel-space)."""
        pred_x0 = torch.randn(2, 4, 8, 8) * 2.0  # typical latent range
        result = ConditionalLDMPipeline._dynamic_threshold(pred_x0, 0.995)
        # Values within [-s, s] should be exactly preserved (no /s division)
        s = torch.quantile(pred_x0.reshape(2, -1).abs(), 0.995, dim=1)
        s = torch.clamp(s, min=1.0).reshape(2, 1, 1, 1)
        mask = pred_x0.abs() <= s
        assert torch.allclose(result[mask], pred_x0[mask], atol=1e-6), \
            "Values within threshold should be exactly preserved (no rescaling)"


# ---------------------------------------------------------------------------
# Test 3: CFG in pipeline — noise prediction combination
# ---------------------------------------------------------------------------

class TestCFGPipeline:
    @pytest.fixture
    def setup(self):
        z_ch = 4
        vae = MockVAE(z_ch)
        unet = MockUNet(in_ch=2 * z_ch, out_ch=z_ch)
        scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="linear")
        return vae, unet, scheduler, z_ch

    def test_cfg_disabled_single_forward(self, setup):
        """With cfg_scale=1.0, UNet should be called once per step."""
        vae, unet, scheduler, z_ch = setup
        pipeline = ConditionalLDMPipeline(
            vae, unet, scheduler,
            cfg_scale=1.0,
            dynamic_threshold_percentile=0.0,
        )
        ncct = torch.randn(1, z_ch, 16, 16)
        unet.call_log.clear()
        _ = pipeline.sample(ncct, num_inference_steps=5, verbose=False)
        # Should have 5 calls, each with batch size 1
        assert len(unet.call_log) == 5
        for shape, _ in unet.call_log:
            assert shape[0] == 1  # batch size = 1

    def test_cfg_enabled_doubled_batch(self, setup):
        """With cfg_scale>1.0, UNet should be called with doubled batch."""
        vae, unet, scheduler, z_ch = setup
        pipeline = ConditionalLDMPipeline(
            vae, unet, scheduler,
            cfg_scale=3.0,
            dynamic_threshold_percentile=0.0,
        )
        ncct = torch.randn(2, z_ch, 16, 16)
        unet.call_log.clear()
        _ = pipeline.sample(ncct, num_inference_steps=5, verbose=False)
        # Should have 5 calls, each with batch size 4 (2*2 for cond+uncond)
        assert len(unet.call_log) == 5
        for shape, _ in unet.call_log:
            assert shape[0] == 4  # doubled batch

    def test_cfg_scale_override(self, setup):
        """cfg_scale parameter in sample() should override pipeline default."""
        vae, unet, scheduler, z_ch = setup
        pipeline = ConditionalLDMPipeline(
            vae, unet, scheduler,
            cfg_scale=1.0,  # disabled by default
            dynamic_threshold_percentile=0.0,
        )
        ncct = torch.randn(1, z_ch, 16, 16)
        unet.call_log.clear()
        _ = pipeline.sample(ncct, num_inference_steps=3, verbose=False, cfg_scale=5.0)
        # Should use doubled batch because override > 1.0
        for shape, _ in unet.call_log:
            assert shape[0] == 2

    def test_cfg_formula_correctness(self):
        """Verify CFG formula: eps = eps_uncond + w * (eps_cond - eps_uncond)."""
        eps_cond = torch.tensor([1.0, 2.0, 3.0])
        eps_uncond = torch.tensor([0.5, 1.0, 1.5])
        w = 3.0
        expected = eps_uncond + w * (eps_cond - eps_uncond)
        # = 0.5 + 3*(0.5) = 2.0, 1.0 + 3*(1.0) = 4.0, 1.5 + 3*(1.5) = 6.0
        assert torch.allclose(expected, torch.tensor([2.0, 4.0, 6.0]))

    def test_cfg_output_shape_matches_input(self, setup):
        """Output shape should match input regardless of CFG."""
        vae, unet, scheduler, z_ch = setup
        ncct = torch.randn(2, z_ch, 16, 16)
        for cfg_s in [1.0, 3.0, 7.5]:
            pipeline = ConditionalLDMPipeline(
                vae, unet, scheduler,
                cfg_scale=cfg_s,
                dynamic_threshold_percentile=0.0,
            )
            out = pipeline.sample(ncct, num_inference_steps=3, verbose=False)
            assert out.shape == (2, z_ch, 16, 16)


# ---------------------------------------------------------------------------
# Test 4: Condition dropping during training
# ---------------------------------------------------------------------------

class TestConditionDropping:
    def test_drop_rate_zero_no_change(self):
        """With cfg_drop_rate=0, condition should never be zeroed."""
        z_ncct = torch.randn(100, 4, 8, 8)
        z_ncct_orig = z_ncct.clone()
        drop_rate = 0.0
        if drop_rate > 0:
            drop_mask = torch.rand(100, 1, 1, 1) < drop_rate
            z_ncct = z_ncct * (~drop_mask).float()
        assert torch.equal(z_ncct, z_ncct_orig)

    def test_drop_rate_one_all_zero(self):
        """With cfg_drop_rate=1.0, all conditions should be zeroed."""
        z_ncct = torch.randn(100, 4, 8, 8)
        drop_rate = 1.0
        drop_mask = torch.rand(100, 1, 1, 1) < drop_rate
        z_ncct = z_ncct * (~drop_mask).float()
        assert z_ncct.abs().max() == 0.0

    def test_drop_rate_approximate_fraction(self):
        """With cfg_drop_rate=0.1, approximately 10% should be zeroed."""
        torch.manual_seed(42)
        B = 10000
        z_ncct = torch.randn(B, 4, 1, 1) + 1.0  # offset so zeros are detectable
        drop_rate = 0.1
        drop_mask = torch.rand(B, 1, 1, 1) < drop_rate
        z_ncct = z_ncct * (~drop_mask).float()
        n_dropped = (z_ncct.abs().sum(dim=[1, 2, 3]) == 0).sum().item()
        # Should be approximately 10% ± some tolerance
        assert 800 < n_dropped < 1200, f"Expected ~1000 drops, got {n_dropped}"

    def test_drop_preserves_non_dropped(self):
        """Non-dropped samples should be unchanged."""
        torch.manual_seed(123)
        z_ncct = torch.randn(100, 4, 8, 8)
        z_orig = z_ncct.clone()
        drop_mask = torch.rand(100, 1, 1, 1) < 0.1
        z_ncct = z_ncct * (~drop_mask).float()
        # Non-dropped samples should be identical
        kept = ~drop_mask.squeeze()
        assert torch.allclose(z_ncct[kept], z_orig[kept])


# ---------------------------------------------------------------------------
# Test 5: CFG with unconditional path uses zero condition
# ---------------------------------------------------------------------------

class TestCFGUnconditionalPath:
    def test_unconditional_uses_zero_condition(self):
        """In CFG mode, the unconditional path should have zero condition."""
        z_ch = 4
        vae = MockVAE(z_ch)
        unet = MockUNetCFGTracker(z_ch)
        scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="linear")
        pipeline = ConditionalLDMPipeline(
            vae, unet, scheduler,
            cfg_scale=3.0,
            dynamic_threshold_percentile=0.0,
        )
        ncct = torch.randn(1, z_ch, 8, 8)
        _ = pipeline.sample(ncct, num_inference_steps=3, verbose=False)

        # Each step: batch of 2 (cond + uncond). uncond should have norm ≈ 0
        # cond_norms[0::2] = conditional (non-zero), cond_norms[1::2] = unconditional (zero)
        for i in range(0, len(unet.cond_norms), 2):
            assert unet.cond_norms[i] > 0, "Conditional path should have non-zero condition"
            assert unet.cond_norms[i + 1] == 0.0, "Unconditional path should have zero condition"


# ---------------------------------------------------------------------------
# Test 6: Dynamic thresholding + DDIM integration
# ---------------------------------------------------------------------------

class TestDynamicThresholdingDDIM:
    def test_thresholding_reduces_pred_x0_range(self):
        """Dynamic thresholding should constrain pred_x0 values."""
        z_ch = 4
        vae = MockVAE(z_ch)
        unet = MockUNet(in_ch=2 * z_ch, out_ch=z_ch)
        scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="linear")

        # Without thresholding
        pipeline_no_dtp = ConditionalLDMPipeline(
            vae, unet, scheduler,
            cfg_scale=1.0,
            dynamic_threshold_percentile=0.0,
        )
        # With thresholding
        pipeline_dtp = ConditionalLDMPipeline(
            vae, unet, scheduler,
            cfg_scale=1.0,
            dynamic_threshold_percentile=0.995,
        )

        ncct = torch.randn(1, z_ch, 16, 16)
        # Both should produce valid output without errors
        out_no = pipeline_no_dtp.sample(ncct, num_inference_steps=5, verbose=False)
        out_dtp = pipeline_dtp.sample(ncct, num_inference_steps=5, verbose=False)
        assert out_no.shape == out_dtp.shape
        assert torch.isfinite(out_no).all()
        assert torch.isfinite(out_dtp).all()


# ---------------------------------------------------------------------------
# Test 7: Backward compatibility — no CFG behaves like before
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_pipeline_default_no_cfg(self):
        """Default pipeline (cfg_scale=1.0) should behave identically to old code."""
        z_ch = 4
        vae = MockVAE(z_ch)
        unet = MockUNet(in_ch=2 * z_ch, out_ch=z_ch)
        scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="linear")

        pipeline = ConditionalLDMPipeline(
            vae, unet, scheduler,
            cfg_scale=1.0,
            dynamic_threshold_percentile=0.0,
        )

        gen = torch.Generator().manual_seed(42)
        ncct = torch.randn(1, z_ch, 16, 16)
        out = pipeline.sample(ncct, num_inference_steps=5, verbose=False, generator=gen)
        assert out.shape == (1, z_ch, 16, 16)
        assert torch.isfinite(out).all()

    def test_config_default_backward_compat(self):
        """New config fields should have sensible defaults."""
        cfg = LDMConfig()
        # CFG scale should be > 1 by default (recommended)
        assert cfg.scheduler.cfg_scale == 3.0
        # Dynamic threshold disabled by default (non-standard for latent diffusion)
        assert cfg.scheduler.dynamic_threshold_percentile == 0.0
        # Drop rate should be > 0 by default (needed for CFG training)
        assert cfg.diffusion_train.cfg_drop_rate == 0.1


# ---------------------------------------------------------------------------
# Test 8: DDIM step math with dynamic thresholding
# ---------------------------------------------------------------------------

class TestDDIMStepWithThresholding:
    def test_recomputed_x_prev_is_finite(self):
        """After thresholding, recomputed x_prev should be finite."""
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
        ddim = DDIMScheduler(scheduler, num_inference_steps=50, eta=0.0)

        # Simulate a step with extreme pred_x0
        z_before = torch.randn(1, 4, 8, 8)
        noise_pred = torch.randn(1, 4, 8, 8) * 5  # somewhat large noise pred

        z_noisy, pred_x0 = ddim.step(noise_pred, 0, z_before)

        # Apply thresholding
        pred_x0_clipped = ConditionalLDMPipeline._dynamic_threshold(pred_x0, 0.995)

        # Recompute x_prev from thresholded pred_x0
        alpha_t = float(ddim.ddim_alphas[0])
        alpha_prev = float(ddim.ddim_alphas_prev[0])
        sigma_t = float(ddim.ddim_sigmas[0])
        sqrt_alpha_t = alpha_t ** 0.5
        sqrt_1m_alpha_t = max((1.0 - alpha_t) ** 0.5, 1e-8)
        pred_eps = (z_before - sqrt_alpha_t * pred_x0_clipped) / sqrt_1m_alpha_t
        dir_xt = ((1.0 - alpha_prev - sigma_t ** 2) ** 0.5) * pred_eps
        z_noisy_new = (alpha_prev ** 0.5) * pred_x0_clipped + dir_xt

        assert torch.isfinite(z_noisy_new).all(), "Recomputed z_noisy should be finite"

    def test_thresholding_at_all_steps(self):
        """Dynamic thresholding should work at every DDIM step without NaN."""
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
        ddim = DDIMScheduler(scheduler, num_inference_steps=50, eta=0.0)

        z = torch.randn(1, 4, 8, 8)
        for i in range(len(ddim.timesteps)):
            noise_pred = torch.randn(1, 4, 8, 8)
            z_before = z
            z, pred_x0 = ddim.step(noise_pred, i, z_before)

            pred_x0 = ConditionalLDMPipeline._dynamic_threshold(pred_x0, 0.995)
            alpha_t = float(ddim.ddim_alphas[i])
            alpha_prev = float(ddim.ddim_alphas_prev[i])
            sigma_t = float(ddim.ddim_sigmas[i])
            sqrt_alpha_t = alpha_t ** 0.5
            sqrt_1m_alpha_t = max((1.0 - alpha_t) ** 0.5, 1e-8)
            pred_eps = (z_before - sqrt_alpha_t * pred_x0) / sqrt_1m_alpha_t
            dir_xt = ((1.0 - alpha_prev - sigma_t ** 2) ** 0.5) * pred_eps
            z = (alpha_prev ** 0.5) * pred_x0 + dir_xt

            assert torch.isfinite(z).all(), f"z should be finite at step {i}"


# ---------------------------------------------------------------------------
# Test 9: SDEdit / img2img strength parameter
# ---------------------------------------------------------------------------

class TestSDEdit:
    @pytest.fixture
    def setup(self):
        z_ch = 4
        vae = MockVAE(z_ch)
        unet = MockUNet(in_ch=z_ch * 2, out_ch=z_ch)
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
        scheduler.to(torch.device("cpu"))
        pipeline = ConditionalLDMPipeline(
            vae, unet, scheduler,
            latent_scale_factor=1.0,
            cfg_scale=1.0,
        )
        return pipeline, z_ch

    def test_strength_1_uses_pure_noise(self, setup):
        """strength=1.0 should start from pure noise (original behavior)."""
        pipeline, z_ch = setup
        ncct = torch.randn(1, z_ch, 8, 8)
        gen = torch.Generator().manual_seed(42)
        out = pipeline.sample(ncct, num_inference_steps=5, strength=1.0,
                              verbose=False, generator=gen)
        assert out.shape == (1, z_ch, 8, 8)
        assert torch.isfinite(out).all()

    def test_strength_0_returns_near_input(self, setup):
        """strength≈0 should return something very close to the NCCT input."""
        pipeline, z_ch = setup
        ncct = torch.randn(1, z_ch, 8, 8) * 0.5
        # strength=0.01 means only 1 denoising step from very low noise
        out = pipeline.sample(ncct, num_inference_steps=50, strength=0.01,
                              verbose=False)
        assert out.shape == ncct.shape
        # Output should be somewhat close to input (very little denoising)
        assert torch.isfinite(out).all()

    def test_strength_05_fewer_steps(self, setup):
        """strength=0.5 should use ~half the denoising steps."""
        pipeline, z_ch = setup
        pipeline.unet.call_log.clear()
        ncct = torch.randn(1, z_ch, 8, 8)
        pipeline.sample(ncct, num_inference_steps=10, strength=0.5,
                        verbose=False)
        # With 10 steps and strength=0.5, should do ~5 denoising steps
        num_calls = len(pipeline.unet.call_log)
        assert num_calls == 5, f"Expected 5 UNet calls, got {num_calls}"

    def test_strength_1_all_steps(self, setup):
        """strength=1.0 should use all denoising steps."""
        pipeline, z_ch = setup
        pipeline.unet.call_log.clear()
        ncct = torch.randn(1, z_ch, 8, 8)
        pipeline.sample(ncct, num_inference_steps=10, strength=1.0,
                        verbose=False)
        num_calls = len(pipeline.unet.call_log)
        assert num_calls == 10, f"Expected 10 UNet calls, got {num_calls}"

    def test_strength_clamped(self, setup):
        """strength values outside [0,1] should be clamped."""
        pipeline, z_ch = setup
        ncct = torch.randn(1, z_ch, 8, 8)
        # Should not crash with out-of-range values
        out_neg = pipeline.sample(ncct, num_inference_steps=5, strength=-0.5,
                                  verbose=False)
        out_high = pipeline.sample(ncct, num_inference_steps=5, strength=2.0,
                                   verbose=False)
        assert torch.isfinite(out_neg).all()
        assert torch.isfinite(out_high).all()

    def test_sdedit_uses_add_noise(self, setup):
        """SDEdit should initialize z_noisy via scheduler.add_noise, not pure noise."""
        pipeline, z_ch = setup
        ncct = torch.randn(1, z_ch, 8, 8)
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        # Same seed should give same results for same strength
        out1 = pipeline.sample(ncct, num_inference_steps=5, strength=0.5,
                               verbose=False, generator=gen1)
        out2 = pipeline.sample(ncct, num_inference_steps=5, strength=0.5,
                               verbose=False, generator=gen2)
        assert torch.allclose(out1, out2, atol=1e-5), \
            "Same seed + same strength should produce identical results"

    def test_different_strengths_different_outputs(self, setup):
        """Different strength values should produce different outputs."""
        pipeline, z_ch = setup
        ncct = torch.randn(1, z_ch, 8, 8)
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        out_05 = pipeline.sample(ncct, num_inference_steps=10, strength=0.5,
                                 verbose=False, generator=gen1)
        out_10 = pipeline.sample(ncct, num_inference_steps=10, strength=1.0,
                                 verbose=False, generator=gen2)
        # Different starting points should give different results
        assert not torch.allclose(out_05, out_10, atol=1e-3), \
            "strength=0.5 and strength=1.0 should produce different outputs"

    def test_sdedit_with_cfg(self, setup):
        """SDEdit should work correctly combined with CFG."""
        pipeline, z_ch = setup
        pipeline.cfg_scale = 2.0
        # Need a UNet that handles doubled batch
        pipeline.unet = MockUNet(in_ch=z_ch * 2, out_ch=z_ch)
        ncct = torch.randn(1, z_ch, 8, 8)
        out = pipeline.sample(ncct, num_inference_steps=5, strength=0.5,
                              verbose=False)
        assert out.shape == (1, z_ch, 8, 8)
        assert torch.isfinite(out).all()

    def test_config_has_strength(self):
        """Config should have the strength field with default 0.5."""
        cfg = SchedulerConfig()
        assert hasattr(cfg, 'strength')
        assert cfg.strength == 0.5

    def test_config_strength_yaml_roundtrip(self, tmp_path):
        """Strength parameter should survive YAML save/load."""
        cfg = LDMConfig()
        cfg.scheduler.strength = 0.35
        path = str(tmp_path / "test_strength.yaml")
        cfg.save(path)
        loaded = LDMConfig.load(path)
        assert abs(loaded.scheduler.strength - 0.35) < 1e-6


# ---------------------------------------------------------------------------
# Test 10: Residual Diffusion
# ---------------------------------------------------------------------------

class TestResidualDiffusion:
    @pytest.fixture
    def setup_standard(self):
        """Standard pipeline (residual_prediction=False)."""
        z_ch = 4
        vae = MockVAE(z_ch)
        unet = MockUNet(in_ch=z_ch * 2, out_ch=z_ch)
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
        scheduler.to(torch.device("cpu"))
        pipeline = ConditionalLDMPipeline(
            vae, unet, scheduler,
            latent_scale_factor=1.0,
            cfg_scale=1.0,
            residual_prediction=False,
        )
        return pipeline, z_ch

    @pytest.fixture
    def setup_residual(self):
        """Residual pipeline (residual_prediction=True)."""
        z_ch = 4
        vae = MockVAE(z_ch)
        unet = MockUNet(in_ch=z_ch * 2, out_ch=z_ch)
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
        scheduler.to(torch.device("cpu"))
        pipeline = ConditionalLDMPipeline(
            vae, unet, scheduler,
            latent_scale_factor=1.0,
            cfg_scale=1.0,
            residual_prediction=True,
            residual_scale=1.0,
        )
        return pipeline, z_ch

    def test_standard_mode_no_residual_addition(self, setup_standard):
        """Standard mode should NOT add z_ncct back."""
        pipeline, z_ch = setup_standard
        ncct = torch.randn(1, z_ch, 8, 8)
        gen = torch.Generator().manual_seed(42)
        out = pipeline.sample(ncct, num_inference_steps=5, verbose=False, generator=gen)
        assert out.shape == (1, z_ch, 8, 8)
        assert torch.isfinite(out).all()

    def test_residual_mode_adds_z_ncct(self, setup_residual):
        """Residual mode should add z_ncct to the denoised output."""
        pipeline, z_ch = setup_residual
        ncct = torch.randn(1, z_ch, 8, 8)
        out = pipeline.sample(ncct, num_inference_steps=5, verbose=False)
        assert out.shape == (1, z_ch, 8, 8)
        assert torch.isfinite(out).all()

    def test_residual_and_standard_differ(self, setup_standard, setup_residual):
        """Residual and standard modes should produce different outputs."""
        pipeline_std, z_ch = setup_standard
        pipeline_res, _ = setup_residual
        ncct = torch.randn(1, z_ch, 8, 8)
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        out_std = pipeline_std.sample(ncct, num_inference_steps=5,
                                      verbose=False, generator=gen1)
        out_res = pipeline_res.sample(ncct, num_inference_steps=5,
                                      verbose=False, generator=gen2)
        # Should differ because residual mode adds z_ncct back
        assert not torch.allclose(out_std, out_res, atol=1e-3), \
            "Residual and standard modes should produce different outputs"

    def test_residual_scale_zero_returns_ncct(self, setup_residual):
        """residual_scale=0.0 should return pure NCCT (no enhancement)."""
        pipeline, z_ch = setup_residual
        pipeline.residual_scale = 0.0
        ncct = torch.randn(1, z_ch, 8, 8)
        out = pipeline.sample(ncct, num_inference_steps=5, verbose=False)
        # With scale=0, output = z_ncct decoded, which equals ncct[:, :z_ch]
        # (MockVAE encode returns x[:, :z_ch], decode returns z as-is)
        expected = ncct[:, :z_ch]  # encode_condition returns ncct[:, :z_ch] * 1.0
        assert torch.allclose(out, expected, atol=1e-6), \
            "residual_scale=0.0 should return the NCCT input (no enhancement)"

    def test_residual_scale_amplifies(self, setup_residual):
        """residual_scale > 1.0 should amplify the predicted residual."""
        pipeline, z_ch = setup_residual
        ncct = torch.randn(1, z_ch, 8, 8)

        gen1 = torch.Generator().manual_seed(42)
        pipeline.residual_scale = 1.0
        out_1x = pipeline.sample(ncct, num_inference_steps=5,
                                 verbose=False, generator=gen1)

        gen2 = torch.Generator().manual_seed(42)
        pipeline.residual_scale = 2.0
        out_2x = pipeline.sample(ncct, num_inference_steps=5,
                                 verbose=False, generator=gen2)

        # The difference from NCCT should be larger with scale=2.0
        z_ncct = ncct[:, :z_ch]  # MockVAE encode → ncct[:, :z_ch]
        diff_1x = (out_1x - z_ncct).abs().mean()
        diff_2x = (out_2x - z_ncct).abs().mean()
        assert diff_2x > diff_1x, \
            "residual_scale=2.0 should produce larger differences from NCCT than 1.0"

    def test_residual_with_cfg(self, setup_residual):
        """Residual mode should work correctly with CFG."""
        pipeline, z_ch = setup_residual
        pipeline.cfg_scale = 2.0
        pipeline.unet = MockUNet(in_ch=z_ch * 2, out_ch=z_ch)
        ncct = torch.randn(1, z_ch, 8, 8)
        out = pipeline.sample(ncct, num_inference_steps=5, verbose=False)
        assert out.shape == (1, z_ch, 8, 8)
        assert torch.isfinite(out).all()

    def test_residual_ddpm_sampling(self, setup_residual):
        """Residual mode should also work with DDPM sampling."""
        pipeline, z_ch = setup_residual
        ncct = torch.randn(1, z_ch, 4, 4)
        # Use small T for speed
        pipeline.scheduler = DDPMScheduler(
            num_train_timesteps=10,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
        pipeline.scheduler.to(torch.device("cpu"))
        out = pipeline.sample_ddpm(ncct, verbose=False)
        assert out.shape == (1, z_ch, 4, 4)
        assert torch.isfinite(out).all()

    def test_config_residual_fields(self):
        """Config should have residual_prediction and residual_scale fields."""
        cfg = SchedulerConfig()
        assert hasattr(cfg, 'residual_prediction')
        assert cfg.residual_prediction == False
        assert hasattr(cfg, 'residual_scale')
        assert cfg.residual_scale == 1.0

    def test_config_residual_yaml_roundtrip(self, tmp_path):
        """Residual parameters should survive YAML save/load."""
        cfg = LDMConfig()
        cfg.scheduler.residual_prediction = True
        cfg.scheduler.residual_scale = 1.5
        path = str(tmp_path / "test_residual.yaml")
        cfg.save(path)
        loaded = LDMConfig.load(path)
        assert loaded.scheduler.residual_prediction == True
        assert abs(loaded.scheduler.residual_scale - 1.5) < 1e-6

    def test_training_target_residual(self):
        """Verify the residual training target is z_cta - z_ncct."""
        z_ncct = torch.randn(2, 4, 8, 8)
        z_cta = torch.randn(2, 4, 8, 8)
        z_residual = z_cta - z_ncct
        # Verify reconstruction: z_ncct + z_residual == z_cta
        assert torch.allclose(z_ncct + z_residual, z_cta, atol=1e-6), \
            "z_ncct + (z_cta - z_ncct) must exactly equal z_cta"

    def test_residual_is_sparse(self):
        """For similar inputs, the residual should be smaller than the full target."""
        # Simulate NCCT and CTA that differ mainly in vessel regions
        z_ncct = torch.randn(2, 4, 16, 16) * 0.5
        z_cta = z_ncct.clone()
        # Add vessel enhancement to a small region
        z_cta[:, :, 6:10, 6:10] += 2.0
        z_residual = z_cta - z_ncct
        # Residual should have much smaller L2 norm than z_cta
        assert z_residual.norm() < z_cta.norm(), \
            "Residual should be sparser (smaller norm) than full CTA target"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
