"""
2.5D Medical Image Dataset for NCCT→CTA translation.

Pipeline:
  1. Load NPZ with mmap_mode='r' (lazy per-key, mmap for uncompressed NPZ)
  2. Random slice position d in D dimension
  3. Extract [d, d+3C) slices → (3C, H, W)
  4. Resize to (3C, H_out, W_out)
  5. Return (3C, H, W) tensors — augmentation is handled later on GPU

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
from typing import Dict

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

    Returns (3C, H, W) tensors for augmentation on GPU.
    Middle C slices are extracted after GPU augmentation in the trainer.

    Each sample returns:
        ncct:      (3C, H, W) normalized input
        cta:       (3C, H, W) normalized target
        ncct_lung: (3C, H, W) binary lung mask
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
    ):
        self.data_dir = Path(data_dir)
        self.num_slices = num_slices
        self.image_size = image_size
        self.hu_min = hu_min
        self.hu_max = hu_max

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
            with np.load(filepath, mmap_mode="r") as npz:
                self._depth_cache[key] = npz["ncct"].shape[0]
        return self._depth_cache[key]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename = self.filenames[idx]
        filepath = self.data_dir / filename
        C = self.num_slices
        context_slices = 3 * C  # total slices for 3-block context

        # --- Step 1: Load with mmap and extract slices ---
        npz = np.load(filepath, mmap_mode="r")

        depth = npz["ncct"].shape[0]
        max_start = depth - context_slices
        if max_start <= 0:
            d = 0
            context_slices = min(context_slices, depth)
        else:
            d = np.random.randint(0, max_start)

        # Extract 3C slices — this is where mmap reads from disk
        ncct_chunk = npz["ncct"][d:d + context_slices].astype(np.float32)
        cta_chunk = npz["cta"][d:d + context_slices].astype(np.float32)
        mask_chunk = npz["ncct_lung_cvx"][d:d + context_slices].astype(np.float32)
        mask_chunk *= ((ncct_chunk > (self.hu_min + 4.0)).astype(np.float32) +
                       (cta_chunk  > (self.hu_min + 4.0)).astype(np.float32))

        # Normalize HU to [-1, 1]
        ncct_chunk = normalize_hu(ncct_chunk, self.hu_min, self.hu_max)
        cta_chunk = normalize_hu(cta_chunk, self.hu_min, self.hu_max)

        # Convert to tensors: (3C, H, W) or (actual_slices, H, W)
        ncct_t = torch.from_numpy(ncct_chunk)
        cta_t = torch.from_numpy(cta_chunk)
        mask_t = torch.from_numpy(mask_chunk)

        # --- Step 2: Pad thin volumes to exactly 3C if needed ---
        if ncct_t.shape[0] < 3 * C:
            ncct_t = self._pad_to_n(ncct_t, 3 * C)
            cta_t = self._pad_to_n(cta_t, 3 * C)
            mask_t = self._pad_to_n(mask_t, 3 * C)

        # --- Step 3: Resize to (3C, H_out, W_out) ---
        target_size = self.image_size
        if ncct_t.shape[-2] != target_size or ncct_t.shape[-1] != target_size:
            ncct_t = self._resize(ncct_t, target_size, mode="bilinear")
            cta_t = self._resize(cta_t, target_size, mode="bilinear")
            mask_t = self._resize(mask_t, target_size, mode="nearest")

        return {
            "ncct": ncct_t,        # (3C, H, W)
            "cta": cta_t,          # (3C, H, W)
            "ncct_lung": mask_t,   # (3C, H, W)
            "filename": filename,
        }

    @staticmethod
    def _resize(tensor: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
        """Resize (D, H, W) tensor to (D, size, size)."""
        x = tensor.unsqueeze(0)  # (1, D, H, W)
        if mode == "nearest":
            x = F.interpolate(x, size=(size, size), mode="nearest")
        else:
            x = F.interpolate(x, size=(size, size), mode=mode, align_corners=False)
        return x.squeeze(0)

    @staticmethod
    def _pad_to_n(tensor: torch.Tensor, n: int) -> torch.Tensor:
        """Pad tensor from (actual, H, W) to (n, H, W) by repeating edge slices."""
        actual = tensor.shape[0]
        if actual >= n:
            return tensor[:n]
        pad_total = n - actual
        pad_top = pad_total // 2
        pad_bot = pad_total - pad_top
        return torch.cat(
            [tensor[:1].repeat(pad_top, 1, 1), tensor, tensor[-1:].repeat(pad_bot, 1, 1)],
            dim=0,
        )
