"""
2.5D sliding window inference for NCCT→CTA translation.

Strategy:
  - Input 3D volume: (D, H, W)
  - Slide a window of size 3C along D, feeding (3C, H, W) to the model
  - Model predicts (C, H, W), but we only keep the middle C slices
    (from index C to 2C) of the prediction
  - Stride = C (non-overlapping middle slices)
  - Boundary handling: pad D at start/end so every output slice is covered
  - Final output: (D, H, W) stitched from middle predictions

Usage:
    python inference.py --config outputs/config.yaml --checkpoint outputs/checkpoint_best.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from models.unet import UNet
from data.dataset import normalize_hu, denormalize_hu

logger = logging.getLogger(__name__)


def load_model(cfg: Config, checkpoint_path: str, device: torch.device, use_ema: bool = True) -> torch.nn.Module:
    """Load trained UNet model from checkpoint."""
    mcfg = cfg.model
    model = UNet(
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

    state = torch.load(checkpoint_path, map_location=device)

    if use_ema and "ema_params" in state:
        # Load EMA parameters into model
        ema_params = state["ema_params"]
        for p, ema_p in zip(model.parameters(), ema_params):
            p.data.copy_(ema_p.data)
        logger.info("Loaded EMA parameters")
    else:
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded model parameters")

    model = model.to(device)
    model.eval()
    return model


def resize_slice(tensor: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    """Resize (D, H, W) tensor to (D, size, size)."""
    x = tensor.unsqueeze(0)  # (1, D, H, W)
    if mode == "nearest":
        x = F.interpolate(x, size=(size, size), mode="nearest")
    else:
        x = F.interpolate(x, size=(size, size), mode=mode, align_corners=False)
    return x.squeeze(0)


def resize_back(tensor: torch.Tensor, H: int, W: int, mode: str = "bilinear") -> torch.Tensor:
    """Resize (D, H_model, W_model) tensor back to (D, H, W)."""
    x = tensor.unsqueeze(0)  # (1, D, H_model, W_model)
    if mode == "nearest":
        x = F.interpolate(x, size=(H, W), mode="nearest")
    else:
        x = F.interpolate(x, size=(H, W), mode=mode, align_corners=False)
    return x.squeeze(0)


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    ncct_volume: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    """
    Predict full 3D CTA volume from NCCT using 2.5D sliding window.

    Sliding window strategy:
      1. Pad D so that D_padded is divisible by C.
      2. Add extra C slices at front and back (replicate boundary).
         This creates an "extended" volume of size D_ext = D_padded + 2C.
      3. Slide windows of size 3C with stride C along the extended volume.
         For window i starting at i*C:
           - Context:    extended[i*C : i*C + 3C]
           - Model input: extended[i*C + C : i*C + 2C]  (middle C slices)
           - Prediction covers positions [i*C + C, i*C + 2C) in extended space,
             which maps to [i*C, (i+1)*C) in padded space (offset by extra_front=C).
      4. Stitch all C-slice predictions to cover [0, D_padded).
      5. Crop back to D_orig and resize to original H, W.

    Args:
        model:       trained UNet model (eval mode)
        ncct_volume: (D, H, W) float32 array in HU values
        cfg:         Config object
        device:      torch device

    Returns:
        pred_volume: (D, H, W) float32 array in HU values
    """
    C = cfg.data.num_slices
    context = 3 * C
    image_size = cfg.data.image_size
    hu_min, hu_max = cfg.data.hu_min, cfg.data.hu_max

    D_orig, H_orig, W_orig = ncct_volume.shape

    # Normalize HU → [-1, 1]
    ncct_norm = normalize_hu(ncct_volume.copy(), hu_min, hu_max)

    # Step 1: Pad D so D_padded is a multiple of C
    D_padded = int(np.ceil(D_orig / C)) * C
    pad_total = D_padded - D_orig
    pad_front = pad_total // 2
    pad_back = pad_total - pad_front

    if pad_total > 0:
        ncct_padded = np.pad(
            ncct_norm,
            ((pad_front, pad_back), (0, 0), (0, 0)),
            mode="edge",
        )
    else:
        ncct_padded = ncct_norm

    # Convert to tensor and resize H, W
    ncct_tensor = torch.from_numpy(ncct_padded).float()  # (D_padded, H, W)
    if H_orig != image_size or W_orig != image_size:
        ncct_tensor = resize_slice(ncct_tensor, image_size)

    # Step 2: Add extra C padding at front and back for boundary windows
    D_pad = ncct_tensor.shape[0]
    ncct_extended = F.pad(
        ncct_tensor.unsqueeze(0).unsqueeze(0),  # (1, 1, D_pad, H, W)
        (0, 0, 0, 0, C, C),
        mode="replicate",
    ).squeeze(0).squeeze(0)  # (D_pad + 2C, H, W)

    # Step 3: Sliding window inference
    D_ext = ncct_extended.shape[0]
    n_windows = (D_ext - context) // C + 1
    pred_slices = []

    for i in tqdm(range(n_windows), desc="Inference", leave=False):
        start = i * C
        window = ncct_extended[start:start + context]           # (3C, H_m, W_m)
        model_input = window[C:2*C].unsqueeze(0).to(device)    # (1, C, H_m, W_m)
        pred = model(model_input)                               # (1, C, H_m, W_m)
        pred_slices.append(pred.squeeze(0).cpu())

    # Step 4: Stitch — each prediction covers C slices in padded space
    pred_full = torch.cat(pred_slices, dim=0)  # (n_windows * C, H_m, W_m)
    pred_padded = pred_full[:D_pad]

    # Step 5: Remove D padding and resize back to original spatial dims
    if pad_total > 0:
        pred_orig = pred_padded[pad_front:pad_front + D_orig]
    else:
        pred_orig = pred_padded[:D_orig]

    if H_orig != image_size or W_orig != image_size:
        pred_orig = resize_back(pred_orig, H_orig, W_orig)

    # Denormalize back to HU
    pred_hu = denormalize_hu(pred_orig.numpy(), hu_min, hu_max)
    return pred_hu


def run_inference(
    cfg: Config,
    checkpoint_path: str,
    output_dir: str,
    use_ema: bool = True,
):
    """Run inference on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model = load_model(cfg, checkpoint_path, device, use_ema=use_ema)

    # Load test file list
    split_dir = Path(cfg.data.split_dir)
    test_file = split_dir / "test.txt"
    if not test_file.exists():
        logger.error(f"Test split file not found: {test_file}")
        return

    with open(test_file, "r") as f:
        test_files = [line.strip() for line in f if line.strip()]
    logger.info(f"Found {len(test_files)} test volumes")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for filename in tqdm(test_files, desc="Test volumes"):
        filepath = Path(cfg.data.data_dir) / filename
        logger.info(f"Processing: {filename}")

        # Load volume
        npz = np.load(filepath, mmap_mode="r")
        ncct_volume = npz["ncct"][:].astype(np.float32)  # (D, H, W)

        # Predict
        pred_cta = predict_volume(model, ncct_volume, cfg, device)

        # Save result
        stem = Path(filename).stem
        save_path = output_path / f"{stem}_pred.npz"
        np.savez_compressed(save_path, pred_cta=pred_cta)
        logger.info(f"  Saved: {save_path} shape={pred_cta.shape}")

    logger.info("Inference complete.")


def main():
    parser = argparse.ArgumentParser(description="2.5D sliding window inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Output directory")
    parser.add_argument("--no_ema", action="store_true", help="Use model weights instead of EMA")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = Config.load(args.config)
    cfg.sync_channels()

    run_inference(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        use_ema=not args.no_ema,
    )


if __name__ == "__main__":
    main()
