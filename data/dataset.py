"""
2.5D Medical Image Dataset for NCCT→CTA translation.

Pipeline:
  1. Load NPZ with mmap_mode='r' (lazy per-key, mmap for uncompressed NPZ)
  2. Random slice position d in D dimension
  3. Extract [d, d+3C) slices → (3C, H, W)
  4. Apply spatial augmentation (same transform to ncct, cta, mask)
  5. Apply pixel augmentation (ncct only)
  6. Extract middle C slices → (C, H, W)
  7. Resize to (C, H_out, W_out)

Note: np.load with mmap_mode='r' enables memory-mapped reading for
uncompressed .npz files. Only the sliced region is loaded into memory,
which greatly reduces memory usage for multi-worker data loading.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict

from .transforms import (
    sample_spatial_params,
    apply_spatial_augmentation,
    sample_pixel_params,
    apply_pixel_augmentation,
)

logger = logging.getLogger(__name__)


def normalize_hu(data: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Normalize HU values to [-1, 1]."""
    data = np.clip(data, hu_min, hu_max)
    data = (data - hu_min) / (hu_max - hu_min)  # [0, 1]
    data = data * 2.0 - 1.0                      # [-1, 1]
    return data


def denormalize_hu(data: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Denormalize from [-1, 1] back to HU values."""
    data = (data + 1.0) / 2.0                    # [0, 1]
    data = data * (hu_max - hu_min) + hu_min      # [hu_min, hu_max]
    return data


class NCCTDataset(Dataset):
    """
    2.5D dataset for NCCT-to-CTA image translation.

    Each sample returns:
        ncct:      (C, H, W) normalized input
        cta:       (C, H, W) normalized target
        ncct_lung: (C, H, W) binary lung mask
        filename:  source NPZ filename
    """

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        num_slices: int = 3,
        image_size: int = 256,
        hu_min: float = -1024.0,
        hu_max: float = 3071.0,
        augment: bool = False,
        aug_prob: float = 0.5,
        max_angle: float = 15.0,
        scale_range: float = 0.1,
        translate_frac: float = 0.05,
        noise_std: float = 0.02,
        brightness_range: float = 0.1,
        contrast_range: float = 0.1,
    ):
        self.data_dir = Path(data_dir)
        self.num_slices = num_slices
        self.image_size = image_size
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.augment = augment
        self.aug_prob = aug_prob
        # Spatial augmentation params
        self.max_angle = max_angle
        self.scale_range = scale_range
        self.translate_frac = translate_frac
        # Pixel augmentation params
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

        # Load file list
        with open(split_file, "r") as f:
            self.filenames = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(self.filenames)} files from {split_file}")

        # Pre-scan to get depth (D) for each file to validate slice extraction
        self._depth_cache: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.filenames)

    def _get_depth(self, filepath: Path) -> int:
        """Get or cache the D dimension of a volume."""
        key = str(filepath)
        if key not in self._depth_cache:
            # Use mmap to just read shape without loading data
            with np.load(filepath, mmap_mode="r") as npz:
                self._depth_cache[key] = npz["ncct"].shape[0]
        return self._depth_cache[key]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename = self.filenames[idx]
        filepath = self.data_dir / filename
        C = self.num_slices
        context_slices = 3 * C  # total slices to extract for augmentation context

        # --- Step 1: Load with mmap and extract slices ---
        npz = np.load(filepath, mmap_mode="r")

        depth = npz["ncct"].shape[0]
        max_start = depth - context_slices
        if max_start <= 0:
            # Volume too thin, use all slices and pad if needed
            d = 0
            context_slices = min(context_slices, depth)
        else:
            d = np.random.randint(0, max_start)

        # Extract 3C slices — this is where mmap reads from disk
        ncct_chunk = npz["ncct"][d:d + context_slices].astype(np.float32)
        cta_chunk = npz["cta"][d:d + context_slices].astype(np.float32)
        mask_chunk = npz["ncct_lung"][d:d + context_slices].astype(np.float32)

        # Normalize HU to [-1, 1]
        ncct_chunk = normalize_hu(ncct_chunk, self.hu_min, self.hu_max)
        cta_chunk = normalize_hu(cta_chunk, self.hu_min, self.hu_max)
        # mask stays as [0, 1]

        # Convert to tensors: (3C, H, W)
        ncct_t = torch.from_numpy(ncct_chunk)
        cta_t = torch.from_numpy(cta_chunk)
        mask_t = torch.from_numpy(mask_chunk)

        # --- Step 2: Spatial augmentation (on 3C slices) ---
        if self.augment and np.random.random() < self.aug_prob:
            spatial_params = sample_spatial_params(
                max_angle=self.max_angle,
                scale_range=self.scale_range,
                translate_frac=self.translate_frac,
            )
            ncct_t, cta_t, mask_t = apply_spatial_augmentation(
                ncct_t, cta_t, mask_t, spatial_params
            )

        # --- Step 3: Pixel augmentation (ncct only) ---
        if self.augment and np.random.random() < self.aug_prob:
            pixel_params = sample_pixel_params(
                brightness_range=self.brightness_range,
                contrast_range=self.contrast_range,
                noise_std=self.noise_std,
            )
            ncct_t = apply_pixel_augmentation(ncct_t, pixel_params)

        # --- Step 4: Extract middle C slices ---
        actual_slices = ncct_t.shape[0]
        if actual_slices >= context_slices:
            start = C  # middle C starts at index C (after first C context slices)
            ncct_t = ncct_t[start:start + C]
            cta_t = cta_t[start:start + C]
            mask_t = mask_t[start:start + C]
        else:
            # Handle thin volumes: take what we can and pad
            ncct_t, cta_t, mask_t = self._pad_to_c(ncct_t, cta_t, mask_t, C)

        # --- Step 5: Resize to (C, H_out, W_out) ---
        target_size = self.image_size
        if ncct_t.shape[-2] != target_size or ncct_t.shape[-1] != target_size:
            ncct_t = self._resize(ncct_t, target_size, mode="bilinear")
            cta_t = self._resize(cta_t, target_size, mode="bilinear")
            mask_t = self._resize(mask_t, target_size, mode="nearest")

        return {
            "ncct": ncct_t,        # (C, H, W)
            "cta": cta_t,          # (C, H, W)
            "ncct_lung": mask_t,   # (C, H, W)
            "filename": filename,
        }

    @staticmethod
    def _resize(tensor: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
        """Resize (C, H, W) tensor to (C, size, size)."""
        x = tensor.unsqueeze(0)  # (1, C, H, W)
        if mode == "nearest":
            x = F.interpolate(x, size=(size, size), mode="nearest")
        else:
            x = F.interpolate(x, size=(size, size), mode=mode, align_corners=False)
        return x.squeeze(0)

    @staticmethod
    def _pad_to_c(
        ncct: torch.Tensor,
        cta: torch.Tensor,
        mask: torch.Tensor,
        C: int,
    ):
        """Pad thin volumes to exactly C slices by repeating edge slices."""
        actual = ncct.shape[0]
        if actual >= C:
            # Center crop
            start = (actual - C) // 2
            return ncct[start:start+C], cta[start:start+C], mask[start:start+C]
        # Pad by repeating edge
        pad_total = C - actual
        pad_top = pad_total // 2
        pad_bot = pad_total - pad_top
        ncct = torch.cat(
            [ncct[:1].repeat(pad_top, 1, 1), ncct, ncct[-1:].repeat(pad_bot, 1, 1)],
            dim=0
        )
        cta = torch.cat(
            [cta[:1].repeat(pad_top, 1, 1), cta, cta[-1:].repeat(pad_bot, 1, 1)],
            dim=0
        )
        mask = torch.cat(
            [mask[:1].repeat(pad_top, 1, 1), mask, mask[-1:].repeat(pad_bot, 1, 1)],
            dim=0
        )
        return ncct, cta, mask
