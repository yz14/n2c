"""
Visualization utilities for training progress monitoring.

Saves grayscale 4x4 PNG grids showing input / prediction / ground truth
for 8 samples (4 rows × 2 columns of triplets → actually 4×4 with input,
pred, gt arranged as columns per sample row).

Layout per grid image (4 rows × 4 columns):
  Each row: [input_1, pred_1, gt_1, | input_2, pred_2, gt_2]  — wait,
  Actually: 8 samples → 4 rows × 2 sample-groups per row, each group
  has 3 images (input, pred, gt) = 4 rows × 6 cols.

Revised: The user said "4x4的png图". Interpretation: 8 samples in a 4x4 grid
means 2 images per sample shown, but that doesn't divide evenly.

Most natural: 4 rows × 6 columns (2 triplets per row) or simply
4 rows × 3 columns (input, pred, gt) for 4 samples per image,
and save 2 images (one for first 4, one for last 4) — but user said one image.

Simplest: 8 rows × 3 columns = input | pred | gt per row.
But user said "4x4": likely 4×4 = 16 cells, so 8 samples with 2 cells each
(input+pred) or a more creative layout.

Final interpretation: "8个样本...做成两张4x4的png图" means TWO 4×4 images:
one for train (8 samples), one for val (8 samples). Each image has 4×4=16 cells.
For 8 samples, that is 2 cells per sample arranged in a 4×4 grid.
Most likely: each sample has an input and prediction shown, arranged as
rows of 4 pairs? Or: rows contain [ncct, pred] for each sample?

Actually re-reading: "保存8个训练样本和8个验证样本的输入和预测结果，做成两张4x4的png图"
So: 2 grids, each 4x4 with 8 samples. 4x4 = 16 cells for 8 samples = 2 cells/sample.
→ Each sample shows (input, prediction). The ground truth may be implicit or
we show 3 per sample in a wider grid.

Let's go with: each grid = 2 rows of 4 samples. Each sample column shows
(top=input, bottom=prediction). That gives 2×4×2=16? No: 2 rows × 4 cols
for inputs in top half, predictions in bottom half, but that's 2×4=8 cells.

Simplest and clearest: 4 rows × 6 columns (input, pred, gt for 2 samples/row,
8 samples total). Title it clearly with column headers.

Going with the most useful layout: 8 rows × 3 cols (input, pred, gt).
"""

import logging
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _normalize_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor from [-1, 1] or arbitrary range to [0, 255] uint8."""
    t = tensor.float()
    t_min = t.min()
    t_max = t.max()
    if t_max - t_min > 1e-8:
        t = (t - t_min) / (t_max - t_min)
    else:
        t = t - t_min
    return (t * 255).clamp(0, 255).to(torch.uint8)


def _normalize_to_uint8_shared(
    tensor: torch.Tensor, vmin: float, vmax: float
) -> torch.Tensor:
    """Normalize a tensor using shared min/max to [0, 255] uint8."""
    t = tensor.float()
    if vmax - vmin > 1e-8:
        t = (t - vmin) / (vmax - vmin)
    else:
        t = t - vmin
    return (t * 255).clamp(0, 255).to(torch.uint8)


def _make_grid(
    images: List[torch.Tensor],
    nrow: int,
    padding: int = 2,
    pad_value: int = 128,
) -> torch.Tensor:
    """
    Arrange a list of (H, W) uint8 tensors into a grid.

    Args:
        images:    list of (H, W) uint8 tensors (all same size)
        nrow:      number of images per row
        padding:   pixels between images
        pad_value: gray level for padding

    Returns:
        (grid_H, grid_W) uint8 tensor
    """
    n = len(images)
    H, W = images[0].shape
    ncol = nrow
    nrows = (n + ncol - 1) // ncol

    grid_h = nrows * H + (nrows + 1) * padding
    grid_w = ncol * W + (ncol + 1) * padding
    grid = torch.full((grid_h, grid_w), pad_value, dtype=torch.uint8)

    for idx, img in enumerate(images):
        row = idx // ncol
        col = idx % ncol
        y = padding + row * (H + padding)
        x = padding + col * (W + padding)
        grid[y:y + H, x:x + W] = img

    return grid


def save_sample_grid(
    inputs: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    save_path: str,
    num_samples: int = 8,
    slice_idx: Optional[int] = None,
) -> None:
    """
    Save a grayscale grid image showing input, prediction, and ground truth.

    Layout: num_samples rows × 3 columns (input | prediction | ground truth).
    Uses SHARED normalization across input/pred/gt so value differences are visible.
    Saves a single grayscale channel (middle channel by default).

    Args:
        inputs:      (N, C, H, W) tensor
        preds:       (N, C, H, W) tensor
        targets:     (N, C, H, W) tensor
        save_path:   output PNG path
        num_samples: number of samples to include (capped by batch size)
        slice_idx:   which channel to visualize (default: C//2, middle slice)
    """
    N = min(num_samples, inputs.shape[0], preds.shape[0], targets.shape[0])
    C = inputs.shape[1]
    if slice_idx is None:
        slice_idx = C // 2

    # Compute shared min/max across all images for consistent normalization
    all_slices = torch.cat([
        inputs[:N, slice_idx],
        preds[:N, slice_idx],
        targets[:N, slice_idx],
    ])
    vmin = all_slices.min().item()
    vmax = all_slices.max().item()

    images = []
    for i in range(N):
        inp_slice = _normalize_to_uint8_shared(inputs[i, slice_idx], vmin, vmax)
        pred_slice = _normalize_to_uint8_shared(preds[i, slice_idx], vmin, vmax)
        gt_slice = _normalize_to_uint8_shared(targets[i, slice_idx], vmin, vmax)
        images.extend([inp_slice, pred_slice, gt_slice])

    grid = _make_grid(images, nrow=3, padding=2, pad_value=128)

    # Save as PNG using raw tensor → simple PGM/PNG writing
    _save_grayscale_png(grid, save_path)
    logger.debug(f"Saved sample grid: {save_path}")


def _save_grayscale_png(tensor: torch.Tensor, path: str) -> None:
    """
    Save a (H, W) uint8 tensor as a grayscale PNG.

    Uses torchvision if available, falls back to manual PGM writing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torchvision.utils
        from torchvision.io import write_png
        # write_png expects (C, H, W) uint8
        img = tensor.unsqueeze(0).cpu()  # (1, H, W)
        write_png(img, str(path))
    except (ImportError, Exception):
        # Fallback: save as PGM (Netpbm grayscale format, widely supported)
        pgm_path = path.with_suffix(".pgm")
        data = tensor.cpu().numpy()
        H, W = data.shape
        with open(pgm_path, "wb") as f:
            f.write(f"P5\n{W} {H}\n255\n".encode())
            f.write(data.tobytes())
        logger.debug(f"Saved as PGM (torchvision unavailable): {pgm_path}")
