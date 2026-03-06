"""
Shared utilities for LDM trainers.

Contains common classes and functions used by both VAE (Stage 1)
and Diffusion (Stage 2) trainers to avoid code duplication.
"""

import math
from typing import Dict

import torch


class MetricTracker:
    """Track running averages of metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = {}
        self._count = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self._sum[k] = self._sum.get(k, 0.0) + v * n
            self._count[k] = self._count.get(k, 0) + n

    def result(self) -> Dict[str, float]:
        return {k: self._sum[k] / self._count[k] for k in self._sum}

    def __str__(self):
        return ", ".join(f"{k}: {v:.6f}" for k, v in self.result().items())


def warmup_cosine_schedule(warmup_steps: int, total_steps: int):
    """Create a warmup + cosine decay LR lambda."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda
