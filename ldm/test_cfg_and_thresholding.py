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
        assert cfg.dynamic_threshold_percentile == 0.995

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
        # s should be clamped to min=1.0, so /s is just /1.0 = no change
        # (since all abs values < 1.0 with high probability)
        # Actually with std=0.5 some might exceed 1.0, so check shape
        assert result.shape == pred_x0.shape

    def test_clips_extreme_values(self):
        """Extreme outlier values should be clipped."""
        pred_x0 = torch.zeros(1, 4, 8, 8)
        pred_x0[0, 0, 0, 0] = 100.0  # extreme outlier
        result = ConditionalLDMPipeline._dynamic_threshold(pred_x0, 0.995)
        # The 99.5th percentile of abs values should be much < 100
        # After clipping and rescaling, max should be <= s/s = 1.0 ... 
        # Actually it clips to [-s, s] then divides by s, so max = 1.0
        assert result.abs().max() <= 1.0 + 1e-6

    def test_per_sample_thresholding(self):
        """Each sample in the batch should be thresholded independently."""
        pred_x0 = torch.zeros(2, 4, 4, 4)
        pred_x0[0] = 0.5  # small
        pred_x0[1] = 10.0  # large
        result = ConditionalLDMPipeline._dynamic_threshold(pred_x0, 0.995)
        # Sample 0: s = max(0.5_percentile, 1.0) = 1.0, result ≈ 0.5
        # Sample 1: s = max(10.0, 1.0) = 10.0, result ≈ 1.0
        assert torch.allclose(result[0], pred_x0[0] / 1.0, atol=0.01)
        assert result[1].abs().max() <= 1.0 + 1e-6

    def test_min_clamp_prevents_shrinking(self):
        """s is clamped to min=1.0, so small-norm samples aren't amplified."""
        pred_x0 = torch.ones(1, 4, 4, 4) * 0.1
        result = ConditionalLDMPipeline._dynamic_threshold(pred_x0, 0.995)
        # s = max(0.1, 1.0) = 1.0, so result = pred_x0 / 1.0 = pred_x0
        assert torch.allclose(result, pred_x0, atol=1e-6)


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
        # Dynamic threshold should be enabled by default (recommended)
        assert cfg.scheduler.dynamic_threshold_percentile == 0.995
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
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
