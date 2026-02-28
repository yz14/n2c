"""
LDM Inference Pipeline for NCCT→CTA translation.

Performs full 3D volume inference using 2.5D sliding window:
  1. Load NCCT volume from NPZ
  2. Normalize HU values to [-1, 1]
  3. Resize spatial dimensions to model input size
  4. Slide through depth with stride C (num_slices):
     - Extract 3C-slice context window (C before + C center + C after)
     - Extract middle C slices (matching training pipeline)
     - Run LDM pipeline: encode NCCT → DDIM denoise → decode → predicted CTA
  5. Assemble predicted C-slice chunks into full 3D volume
  6. Resize back to original spatial dimensions
  7. Denormalize to HU values and save

Usage:
    python -m ldm.inference \\
        --config configs/ldm_default.yaml \\
        --vae_ckpt outputs_vae/checkpoint_best.pt \\
        --diff_ckpt outputs_diffusion/checkpoint_best.pt \\
        --input path/to/volume.npz \\
        --output path/to/output.npz
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldm.config import LDMConfig
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.unet import DiffusionUNet
from ldm.diffusion.scheduler import DDPMScheduler
from ldm.diffusion.pipeline import ConditionalLDMPipeline
from data.dataset import normalize_hu, denormalize_hu

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Volume inference engine
# ---------------------------------------------------------------------------

class VolumeInference:
    """
    2.5D sliding-window inference for full 3D volume NCCT→CTA translation.

    The sliding window mirrors the training data pipeline:
      - Context window: 3C slices (C before + C center + C after)
      - Output per window: C center slices
      - Stride: C slices along depth

    Boundary handling: edge slices are repeated to pad incomplete windows.
    """

    def __init__(
        self,
        pipeline: ConditionalLDMPipeline,
        num_slices: int = 3,
        image_size: int = 256,
        hu_min: float = -1024.0,
        hu_max: float = 3071.0,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        batch_size: int = 4,
    ):
        self.pipeline = pipeline
        self.num_slices = num_slices  # C
        self.image_size = image_size
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.batch_size = batch_size

    @property
    def device(self) -> torch.device:
        return self.pipeline.device

    def predict_volume(
        self,
        ncct_volume: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Run inference on a full 3D NCCT volume.

        Args:
            ncct_volume: (D, H, W) raw HU values.
            verbose: show progress.

        Returns:
            (D, H, W) predicted CTA volume in HU values.
        """
        D_orig, H_orig, W_orig = ncct_volume.shape
        C = self.num_slices

        # --- Step 1: Normalize ---
        ncct_norm = normalize_hu(ncct_volume.astype(np.float32), self.hu_min, self.hu_max)
        ncct_tensor = torch.from_numpy(ncct_norm)  # (D, H, W)

        # --- Step 2: Resize spatial dims ---
        if H_orig != self.image_size or W_orig != self.image_size:
            ncct_tensor = self._resize_volume(ncct_tensor, self.image_size)

        D = ncct_tensor.shape[0]

        # --- Step 3: Pad depth for complete windows ---
        # We need at least C slices before and after each center window
        # Pad C slices on top and bottom
        ncct_padded = self._pad_depth(ncct_tensor, C)
        D_padded = ncct_padded.shape[0]

        # --- Step 4: Sliding window inference ---
        # Center positions: C, C+C, C+2C, ... covering original depth
        output_slices = []
        windows = []
        center_starts = list(range(C, C + D, C))

        # Collect all windows
        for center_start in center_starts:
            context_start = center_start - C
            context_end = min(center_start + 2 * C, D_padded)
            context_slices = 3 * C

            window = ncct_padded[context_start:context_end]  # (up to 3C, H, W)

            # Pad if at the bottom boundary
            if window.shape[0] < context_slices:
                window = self._pad_to_n(window, context_slices)

            # Extract middle C slices (same as GPUAugmentor does during training)
            middle = window[C:C + C]  # (C, H, W) — the center block
            windows.append(middle)

        # Batch inference
        n_windows = len(windows)
        pbar_desc = "Inference"
        iterator = range(0, n_windows, self.batch_size)
        if verbose:
            iterator = tqdm(list(iterator), desc=pbar_desc)

        for batch_start in iterator:
            batch_end = min(batch_start + self.batch_size, n_windows)
            batch_input = torch.stack(
                windows[batch_start:batch_end], dim=0
            ).to(self.device)  # (B, C, H, W)

            # Run LDM pipeline
            with torch.no_grad():
                cta_pred = self.pipeline.sample(
                    batch_input,
                    num_inference_steps=self.ddim_steps,
                    eta=self.ddim_eta,
                    verbose=False,
                )  # (B, C, H, W)

            output_slices.append(cta_pred.cpu())

        # --- Step 5: Assemble output volume ---
        all_output = torch.cat(output_slices, dim=0)  # (n_windows, C, H, W)
        # Flatten: (n_windows * C, H, W), then trim to original depth
        cta_volume = all_output.reshape(-1, all_output.shape[-2], all_output.shape[-1])
        cta_volume = cta_volume[:D]  # Trim excess from last window

        # --- Step 6: Resize back to original spatial dims ---
        if H_orig != self.image_size or W_orig != self.image_size:
            cta_volume = self._resize_volume(cta_volume, H_orig, W_orig)

        # --- Step 7: Denormalize ---
        cta_hu = denormalize_hu(cta_volume.numpy(), self.hu_min, self.hu_max)

        return cta_hu

    @staticmethod
    def _resize_volume(
        tensor: torch.Tensor, target_h: int, target_w: Optional[int] = None,
    ) -> torch.Tensor:
        """Resize (D, H, W) volume spatially."""
        if target_w is None:
            target_w = target_h
        x = tensor.unsqueeze(0)  # (1, D, H, W)
        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x.squeeze(0)

    @staticmethod
    def _pad_depth(tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """Pad depth by repeating edge slices."""
        top = tensor[:1].repeat(pad_size, 1, 1)
        bot = tensor[-1:].repeat(pad_size, 1, 1)
        return torch.cat([top, tensor, bot], dim=0)

    @staticmethod
    def _pad_to_n(tensor: torch.Tensor, n: int) -> torch.Tensor:
        """Pad tensor from (actual, H, W) to (n, H, W) by repeating last slice."""
        actual = tensor.shape[0]
        if actual >= n:
            return tensor[:n]
        pad = tensor[-1:].repeat(n - actual, 1, 1)
        return torch.cat([tensor, pad], dim=0)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_vae_from_checkpoint(
    cfg: LDMConfig, ckpt_path: str, device: torch.device,
) -> AutoencoderKL:
    """Load VAE from a Stage 1 checkpoint (prefers EMA weights)."""
    vae = AutoencoderKL(cfg.vae)
    state = torch.load(ckpt_path, map_location=device)

    if "vae_state_dict" in state:
        vae.load_state_dict(state["vae_state_dict"])
        # If EMA params available, overwrite with them (better quality)
        if "ema_params" in state and isinstance(state["ema_params"], list):
            logger.info("  Loading EMA weights for VAE")
            for p, ema_p in zip(vae.parameters(), state["ema_params"]):
                p.data.copy_(ema_p.data)
    else:
        vae.load_state_dict(state, strict=False)

    logger.info(f"Loaded VAE from: {ckpt_path}")
    return vae.to(device).eval()


def load_unet_from_checkpoint(
    cfg: LDMConfig, ckpt_path: str, device: torch.device,
) -> DiffusionUNet:
    """Load diffusion UNet from a Stage 2 checkpoint (prefers EMA weights)."""
    unet = DiffusionUNet.from_config(cfg.unet, z_channels=cfg.vae.embed_dim)
    state = torch.load(ckpt_path, map_location=device)

    if "unet_state_dict" in state:
        unet.load_state_dict(state["unet_state_dict"])
        if "ema_params" in state and isinstance(state["ema_params"], list):
            logger.info("  Loading EMA weights for UNet")
            for p, ema_p in zip(unet.parameters(), state["ema_params"]):
                p.data.copy_(ema_p.data)
    else:
        unet.load_state_dict(state, strict=False)

    logger.info(f"Loaded UNet from: {ckpt_path}")
    return unet.to(device).eval()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LDM Inference: NCCT → CTA")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to LDM YAML config")
    parser.add_argument("--vae_ckpt", type=str, required=True,
                        help="Path to VAE checkpoint (Stage 1)")
    parser.add_argument("--diff_ckpt", type=str, required=True,
                        help="Path to diffusion UNet checkpoint (Stage 2)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input NPZ file (must contain 'ncct' key)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output NPZ file")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM sampling steps (default: 50)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for sliding window inference")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM eta (0=deterministic, >0=stochastic)")
    args = parser.parse_args()

    # Setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Config
    cfg = LDMConfig.load(args.config)
    cfg.sync_channels()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load models
    vae = load_vae_from_checkpoint(cfg, args.vae_ckpt, device)
    unet = load_unet_from_checkpoint(cfg, args.diff_ckpt, device)

    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.scheduler.num_train_timesteps,
        beta_schedule=cfg.scheduler.beta_schedule,
        beta_start=cfg.scheduler.beta_start,
        beta_end=cfg.scheduler.beta_end,
        prediction_type=cfg.scheduler.prediction_type,
    ).to(device)

    pipeline = ConditionalLDMPipeline(vae, unet, scheduler)

    # Load input volume
    logger.info(f"Loading input: {args.input}")
    npz = np.load(args.input)
    ncct_volume = npz["ncct"]  # (D, H, W)
    logger.info(f"  Volume shape: {ncct_volume.shape}")

    # Run inference
    engine = VolumeInference(
        pipeline=pipeline,
        num_slices=cfg.data.num_slices,
        image_size=cfg.data.image_size,
        hu_min=cfg.data.hu_min,
        hu_max=cfg.data.hu_max,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.eta,
        batch_size=args.batch_size,
    )

    t0 = time.time()
    cta_pred = engine.predict_volume(ncct_volume, verbose=True)
    elapsed = time.time() - t0
    logger.info(f"Inference complete in {elapsed:.1f}s")
    logger.info(f"  Output shape: {cta_pred.shape}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"cta_pred": cta_pred}

    # Include original CTA if present (for evaluation)
    if "cta" in npz:
        save_dict["cta_gt"] = npz["cta"]
    save_dict["ncct"] = ncct_volume

    np.savez_compressed(str(output_path), **save_dict)
    logger.info(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
