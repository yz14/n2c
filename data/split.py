"""
Data splitting utility.

Splits NPZ files into train/valid/test sets (8:1:1) with:
- Patient-level grouping (same patient never in different splits)
- Quality-aware stratified splitting (sorted by registration quality)

File naming convention: {patient_id}-{quality_score}.npz
  e.g., 694268-9539.npz → patient_id="694268", quality=95.39%
"""

import os
import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def parse_filename(filename: str) -> Tuple[str, float]:
    """
    Parse NPZ filename to extract patient_id and quality score.

    Args:
        filename: e.g., "694268-9539.npz"

    Returns:
        (patient_id, quality_score): e.g., ("694268", 95.39)
    """
    stem = Path(filename).stem  # "694268-9539"
    # Split on the LAST hyphen to handle IDs containing hyphens
    last_dash = stem.rfind("-")
    if last_dash == -1:
        raise ValueError(f"Invalid filename format: {filename}")
    patient_id = stem[:last_dash]
    quality_str = stem[last_dash + 1:]
    quality_score = int(quality_str) / 100.0  # 9539 → 95.39
    return patient_id, quality_score


def group_by_patient(filenames: List[str]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Group filenames by patient_id.

    Returns:
        {patient_id: [(filename, quality_score), ...]}
    """
    groups = defaultdict(list)
    for fn in filenames:
        pid, quality = parse_filename(fn)
        groups[pid].append((fn, quality))
    return dict(groups)


def split_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split dataset into train/valid/test by patient groups, stratified by quality.

    Strategy:
      1. Group all files by patient_id
      2. Compute mean quality per patient (handles multiple scans)
      3. Sort patients by mean quality (descending)
      4. Distribute patients into 3 bins by quality, then split within each bin

    Args:
        data_dir:    directory containing .npz files
        output_dir:  directory to save train.txt, valid.txt, test.txt
        train_ratio: fraction for training
        valid_ratio: fraction for validation
        test_ratio:  fraction for testing
        seed:        random seed

    Returns:
        (train_files, valid_files, test_files) lists of filenames
    """
    import random
    random.seed(seed)

    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all npz files
    all_files = sorted([f.name for f in data_dir.glob("*.npz")])
    if not all_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    logger.info(f"Found {len(all_files)} NPZ files in {data_dir}")

    # Group by patient
    patient_groups = group_by_patient(all_files)
    num_patients = len(patient_groups)
    logger.info(f"Found {num_patients} unique patients")

    # Compute mean quality per patient and sort descending
    patient_quality = []
    for pid, file_list in patient_groups.items():
        mean_quality = sum(q for _, q in file_list) / len(file_list)
        patient_quality.append((pid, mean_quality))
    patient_quality.sort(key=lambda x: x[1], reverse=True)

    # Stratified split: sort by quality, interleave into groups,
    # then assign to train/valid/test ensuring minimum 1 per split.
    n = num_patients
    n_test = max(1, round(n * test_ratio))
    n_valid = max(1, round(n * valid_ratio))
    n_train = n - n_valid - n_test
    assert n_train >= 1, f"Not enough patients ({n}) for 8:1:1 split"

    # Interleave by quality: take every k-th patient for val/test
    # so that each split gets a spread of quality levels.
    all_pids = [pid for pid, _ in patient_quality]  # sorted by quality desc
    random.shuffle(all_pids)

    # Re-sort after shuffle to keep quality-stratified ordering
    # Use a deterministic interleave: assign indices round-robin
    # After shuffling, pick first n_test for test, next n_valid for valid, rest train
    # But we want quality spread → interleave first, then assign
    sorted_pids = [pid for pid, _ in patient_quality]  # quality desc
    # Assign every (n // n_test)-th patient to test, similar for valid
    test_indices = set()
    step_test = max(1, n // n_test)
    for i in range(n_test):
        idx = min(i * step_test + step_test // 2, n - 1)
        test_indices.add(idx)

    remaining = [i for i in range(n) if i not in test_indices]
    random.shuffle(remaining)
    valid_indices = set(remaining[:n_valid])
    train_indices = set(remaining[n_valid:])

    train_patients = [sorted_pids[i] for i in sorted(train_indices)]
    valid_patients = [sorted_pids[i] for i in sorted(valid_indices)]
    test_patients = [sorted_pids[i] for i in sorted(test_indices)]

    # Collect files for each split
    def collect_files(patient_list):
        files = []
        for pid in patient_list:
            for fn, _ in patient_groups[pid]:
                files.append(fn)
        return sorted(files)

    train_files = collect_files(train_patients)
    valid_files = collect_files(valid_patients)
    test_files = collect_files(test_patients)

    # Write to txt files
    for name, file_list in [
        ("train.txt", train_files),
        ("valid.txt", valid_files),
        ("test.txt", test_files),
    ]:
        path = output_dir / name
        with open(path, "w") as f:
            f.write("\n".join(file_list))
        logger.info(f"  {name}: {len(file_list)} files → {path}")

    # Log summary
    logger.info(f"Split summary: train={len(train_files)}, "
                f"valid={len(valid_files)}, test={len(test_files)}")

    # Verify no patient leakage
    train_pids = set(train_patients)
    valid_pids = set(valid_patients)
    test_pids = set(test_patients)
    assert train_pids.isdisjoint(valid_pids), "Patient leakage: train ∩ valid"
    assert train_pids.isdisjoint(test_pids), "Patient leakage: train ∩ test"
    assert valid_pids.isdisjoint(test_pids), "Patient leakage: valid ∩ test"
    logger.info("No patient leakage detected.")

    return train_files, valid_files, test_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="Split NCCT dataset")
    parser.add_argument("--data_dir", type=str, default="D:/codes/data/ncct_tiny")
    parser.add_argument("--output_dir", type=str, default="./splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.data_dir, args.output_dir, seed=args.seed)
