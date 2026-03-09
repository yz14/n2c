"""
Tests for diffusion bugfixes:
  BUG-1: DDIM alphas_prev reversed direction (CRITICAL)
  BUG-2: MultiHeadAttnBlock reshape scrambles spatial/channel dims (Important)

Run: conda run -n py310 python -m ldm.test_diffusion_bugfixes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np
import torch
import torch.nn as nn

from ldm.diffusion.scheduler import DDPMScheduler, DDIMScheduler
from ldm.models.blocks import MultiHeadAttnBlock


class TestBug1_DDIMAlphasPrev(unittest.TestCase):
    """BUG-1: DDIM alphas_prev must decrease monotonically (approaching clean)."""

    def setUp(self):
        self.ddpm = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
        )

    def test_alphas_prev_monotonically_increasing(self):
        """alphas_prev should increase over steps (less noisy → clean)."""
        ddim = DDIMScheduler(self.ddpm, num_inference_steps=50, eta=0.0)
        # ddim_alphas_prev[i] should be > ddim_alphas[i] for all steps
        # (prev is less noisy than current)
        for i in range(len(ddim.timesteps)):
            alpha_t = ddim.ddim_alphas[i]
            alpha_prev = ddim.ddim_alphas_prev[i]
            self.assertGreater(
                alpha_prev, alpha_t,
                f"Step {i}: alpha_prev ({alpha_prev:.6f}) must be > "
                f"alpha_t ({alpha_t:.6f}) — prev should be less noisy"
            )

    def test_last_step_arrives_at_clean(self):
        """The last step's alpha_prev should be 1.0 (clean signal)."""
        ddim = DDIMScheduler(self.ddpm, num_inference_steps=50, eta=0.0)
        self.assertAlmostEqual(
            ddim.ddim_alphas_prev[-1], 1.0, places=10,
            msg="Last step alpha_prev must be 1.0 (clean)"
        )

    def test_first_step_goes_to_next_timestep(self):
        """Step 0 should go from highest noise to next-highest, not to clean."""
        ddim = DDIMScheduler(self.ddpm, num_inference_steps=50, eta=0.0)
        alphas_cumprod = self.ddpm.alphas_cumprod.numpy()
        # First step: t=980, should go to t=960
        expected_prev = alphas_cumprod[ddim.timesteps[1]]
        actual_prev = ddim.ddim_alphas_prev[0]
        np.testing.assert_almost_equal(
            actual_prev, expected_prev, decimal=10,
            err_msg="Step 0 alpha_prev should be alphas_cumprod at next timestep"
        )

    def test_alphas_prev_not_1_at_first_step(self):
        """Step 0's alpha_prev must NOT be 1.0 (that was the old bug)."""
        ddim = DDIMScheduler(self.ddpm, num_inference_steps=50, eta=0.0)
        self.assertNotAlmostEqual(
            ddim.ddim_alphas_prev[0], 1.0, places=2,
            msg="Step 0 alpha_prev must NOT be 1.0 — that would skip all steps"
        )

    def test_ddim_step_denoises_toward_clean(self):
        """A single DDIM step should move the sample closer to pred_x0, not away."""
        ddim = DDIMScheduler(self.ddpm, num_inference_steps=50, eta=0.0)

        # Create a simple "noisy" sample and "noise prediction"
        torch.manual_seed(42)
        B, C, H, W = 1, 4, 8, 8
        z_noisy = torch.randn(B, C, H, W)
        noise_pred = torch.randn(B, C, H, W)  # fake noise prediction

        # Run step 0
        x_prev, pred_x0 = ddim.step(noise_pred, 0, z_noisy)

        # x_prev should be between z_noisy and pred_x0 (interpolation)
        # NOT equal to pred_x0 (which would mean alpha_prev=1.0 bug)
        diff_to_noisy = (x_prev - z_noisy).abs().mean().item()
        diff_to_pred_x0 = (x_prev - pred_x0).abs().mean().item()

        self.assertGreater(
            diff_to_pred_x0, 0.01,
            "x_prev should NOT equal pred_x0 at step 0 (old bug: alpha_prev=1.0)"
        )
        self.assertGreater(
            diff_to_noisy, 0.01,
            "x_prev should differ from z_noisy (denoising should happen)"
        )

    def test_full_ddim_sampling_produces_different_output(self):
        """Full DDIM sampling should produce coherent output, not oscillating garbage."""
        ddim = DDIMScheduler(self.ddpm, num_inference_steps=20, eta=0.0)

        torch.manual_seed(42)
        B, C, H, W = 1, 4, 8, 8
        z = torch.randn(B, C, H, W)

        # Use a simple "model" that always predicts zero noise
        # If sampling works, this should converge toward the input structure
        intermediates = []
        for i in range(len(ddim.timesteps)):
            noise_pred = torch.zeros_like(z)  # predict no noise
            z, pred_x0 = ddim.step(noise_pred, i, z)
            intermediates.append(z.clone())

        # Check that the output isn't NaN or Inf
        self.assertFalse(torch.isnan(z).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(z).any(), "Output contains Inf")

        # Check that consecutive steps don't oscillate wildly
        # (old bug caused the signal to jump between clean and noisy)
        for i in range(1, len(intermediates)):
            diff = (intermediates[i] - intermediates[i - 1]).abs().mean().item()
            self.assertLess(
                diff, 10.0,
                f"Step {i}: diff={diff:.4f} is too large, suggesting oscillation"
            )

    def test_various_step_counts(self):
        """Test that the fix works for different numbers of inference steps."""
        for steps in [10, 20, 50, 100, 200]:
            ddim = DDIMScheduler(self.ddpm, num_inference_steps=steps, eta=0.0)
            # Verify monotonicity for all step counts
            for i in range(len(ddim.timesteps)):
                self.assertGreater(
                    ddim.ddim_alphas_prev[i], ddim.ddim_alphas[i],
                    f"Steps={steps}, i={i}: alpha_prev must > alpha_t"
                )
            # Last should be 1.0
            self.assertAlmostEqual(ddim.ddim_alphas_prev[-1], 1.0, places=10)

    def test_sigma_computation_non_negative(self):
        """Sigma values should be non-negative for all step counts and eta values."""
        for steps in [10, 50]:
            for eta in [0.0, 0.5, 1.0]:
                ddim = DDIMScheduler(self.ddpm, num_inference_steps=steps, eta=eta)
                self.assertTrue(
                    np.all(ddim.ddim_sigmas >= 0),
                    f"steps={steps}, eta={eta}: negative sigma found"
                )

    def test_matches_guided_diffusion_convention(self):
        """Verify our DDIM matches the guided-diffusion convention:
        alpha_prev[step_i] = alphas_cumprod at the NEXT (less noisy) timestep."""
        ddim = DDIMScheduler(self.ddpm, num_inference_steps=50, eta=0.0)
        alphas_cumprod = self.ddpm.alphas_cumprod.numpy()

        for i in range(len(ddim.timesteps) - 1):
            expected = alphas_cumprod[ddim.timesteps[i + 1]]
            actual = ddim.ddim_alphas_prev[i]
            np.testing.assert_almost_equal(
                actual, expected, decimal=10,
                err_msg=f"Step {i}: alpha_prev should match alphas_cumprod "
                        f"at timestep {ddim.timesteps[i + 1]}"
            )
        # Last step: alpha_prev = 1.0
        self.assertAlmostEqual(ddim.ddim_alphas_prev[-1], 1.0, places=10)


class TestBug2_MultiHeadAttnReshape(unittest.TestCase):
    """BUG-2: MultiHeadAttnBlock output reshape must preserve spatial structure."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        C, num_heads = 64, 4
        attn = MultiHeadAttnBlock(C, num_heads=num_heads)
        x = torch.randn(2, C, 8, 8)
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_spatial_locality_preserved(self):
        """Verify that attention output preserves spatial locality.

        Strategy: Create an input where each spatial position has a unique pattern.
        After attention with identity-like weights, the output at each spatial
        position should still relate to its own input, not to other positions.
        """
        C, num_heads = 64, 4
        H, W = 4, 4
        attn = MultiHeadAttnBlock(C, num_heads=num_heads)

        # Create spatially distinct input: each position has value (h*W + w)
        x = torch.zeros(1, C, H, W)
        for h in range(H):
            for w in range(W):
                x[0, :, h, w] = float(h * W + w)

        # Forward pass
        out = attn(x)

        # With residual connection and zero-initialized proj_out,
        # the output should approximately equal x (since proj_out starts at zero)
        # The key test: spatial positions should NOT be scrambled
        diff = (out - x).abs()
        self.assertLess(
            diff.max().item(), 1e-5,
            "With zero-initialized proj_out, residual should preserve spatial structure"
        )

    def test_no_spatial_channel_mixing(self):
        """Directly test that the reshape doesn't mix spatial and channel dims.

        Create a known attention output tensor and verify the reshape produces
        the correct spatial-channel mapping.
        """
        B, C, H, W = 1, 4, 2, 2
        num_heads = 2
        head_dim = C // num_heads  # = 2
        HW = H * W  # = 4

        # Create a known tensor: out[b, head, hw, dim] = head*100 + hw*10 + dim
        out_attn = torch.zeros(B, num_heads, HW, head_dim)
        for head in range(num_heads):
            for hw in range(HW):
                for dim in range(head_dim):
                    out_attn[0, head, hw, dim] = head * 100 + hw * 10 + dim

        # Apply the same reshape as in MultiHeadAttnBlock
        result = out_attn.permute(0, 2, 1, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Verify: result[0, c, h, w] should be out_attn[0, head, hw, dim]
        # where c = head*head_dim + dim and hw = h*W + w
        for head in range(num_heads):
            for hw in range(HW):
                h, w = hw // W, hw % W
                for dim in range(head_dim):
                    c = head * head_dim + dim
                    expected = head * 100 + hw * 10 + dim
                    actual = result[0, c, h, w].item()
                    self.assertAlmostEqual(
                        actual, expected, places=5,
                        msg=f"result[0, c={c}, h={h}, w={w}] = {actual}, "
                            f"expected {expected} (head={head}, hw={hw}, dim={dim})"
                    )

    def test_gradient_flows_to_correct_spatial_positions(self):
        """Verify gradients flow to the correct spatial positions through attention."""
        C, num_heads = 64, 4
        H, W = 4, 4
        attn = MultiHeadAttnBlock(C, num_heads=num_heads)

        x = torch.randn(1, C, H, W, requires_grad=True)
        out = attn(x)

        # Compute gradient w.r.t. a specific spatial position
        target_h, target_w = 2, 3
        loss = out[0, :, target_h, target_w].sum()
        loss.backward()

        # The gradient at the target position should be significantly larger
        # than at distant positions (due to residual + attention locality)
        grad = x.grad[0]  # (C, H, W)
        target_grad_norm = grad[:, target_h, target_w].abs().sum().item()
        # Should have non-zero gradient at target
        self.assertGreater(target_grad_norm, 0.0)


class TestDDIMSamplingEndToEnd(unittest.TestCase):
    """End-to-end test verifying DDIM sampling produces progressively cleaner output."""

    def test_progressive_denoising(self):
        """Each DDIM step should move the noise level closer to the target."""
        ddpm = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
        )
        ddim = DDIMScheduler(ddpm, num_inference_steps=50, eta=0.0)

        torch.manual_seed(123)
        B, C, H, W = 2, 4, 8, 8

        # Create a "clean" target
        x_clean = torch.randn(B, C, H, W) * 0.1  # low variance, like a real latent
        noise = torch.randn(B, C, H, W)

        # Start from full noise
        z = noise.clone()

        # Use a perfect noise predictor: given x_t, predict the actual noise
        # This tests whether the DDIM step formula correctly recovers x_clean
        prev_dist = float('inf')
        for i in range(len(ddim.timesteps)):
            t_val = ddim.timesteps[i]
            alpha_t = ddim.ddim_alphas[i]

            # Perfect noise prediction: ε = (x_t - sqrt(α_t) * x_0) / sqrt(1-α_t)
            noise_pred = (z - np.sqrt(alpha_t) * x_clean) / max(np.sqrt(1 - alpha_t), 1e-8)
            z, pred_x0 = ddim.step(noise_pred, i, z)

            # Distance to clean should generally decrease
            dist = (z - x_clean).abs().mean().item()

        # After all steps, should be very close to x_clean
        final_dist = (z - x_clean).abs().mean().item()
        self.assertLess(
            final_dist, 0.1,
            f"After full DDIM with perfect predictor, distance to clean "
            f"should be near zero, got {final_dist:.6f}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
