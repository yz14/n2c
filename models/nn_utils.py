"""
Neural network utilities adapted from guided-diffusion.

Provides building blocks for the UNet model:
  - GroupNorm32: GroupNorm with float32 computation
  - conv_nd / avg_pool_nd: dimension-agnostic conv and pooling
  - zero_module: zero-initialize module parameters
  - checkpoint: gradient checkpointing wrapper
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_pretrained_weights(
    model: nn.Module,
    pretrained_path: str,
    component_key: Optional[str] = None,
    device: Optional[torch.device] = None,
    strict: bool = False,
) -> None:
    """
    Load pretrained weights into a model from a checkpoint or raw state dict file.

    Supports two file formats:
      1. Full training checkpoint (dict with 'model_state_dict', 'reg_net_state_dict', etc.)
         → extracts the relevant component via `component_key`.
      2. Raw state dict (dict of parameter tensors)
         → loads directly.

    Args:
        model:          target nn.Module to load weights into
        pretrained_path: path to .pt/.pth file
        component_key:  key to extract from checkpoint dict (e.g., 'model_state_dict',
                        'reg_net_state_dict', 'discriminator_state_dict').
                        If None, tries common keys then falls back to raw state dict.
        device:         map_location for torch.load
        strict:         if False (default), ignore missing/unexpected keys
    """
    path = Path(pretrained_path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

    logger.info(f"Loading pretrained weights from: {pretrained_path}")
    state = torch.load(pretrained_path, map_location=device or "cpu")

    # Extract state dict from checkpoint if needed
    state_dict = _extract_state_dict(state, component_key)

    # Load with detailed logging
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    n_loaded = len(state_dict) - len(unexpected)
    n_model = len(list(model.state_dict().keys()))
    logger.info(f"  Loaded {n_loaded}/{n_model} parameter tensors")
    if missing:
        logger.warning(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def _extract_state_dict(state: dict, component_key: Optional[str] = None) -> dict:
    """
    Extract a model state dict from a checkpoint or raw state dict.

    If component_key is provided, look for that key first.
    Otherwise, try common checkpoint keys, then treat as raw state dict.
    """
    if not isinstance(state, dict):
        raise ValueError(f"Expected dict, got {type(state)}")

    # If a specific key is requested, use it
    if component_key and component_key in state:
        logger.info(f"  Extracted component: '{component_key}'")
        return state[component_key]

    # Auto-detect: try common checkpoint keys
    common_keys = [
        "model_state_dict", "state_dict", "model",
        "reg_net_state_dict", "discriminator_state_dict",
    ]
    for key in common_keys:
        if key in state:
            # Check if this looks like a state dict (values are tensors)
            candidate = state[key]
            if isinstance(candidate, dict) and any(
                isinstance(v, torch.Tensor) for v in candidate.values()
            ):
                logger.info(f"  Auto-detected component key: '{key}'")
                return candidate

    # Treat the whole dict as a raw state dict if values are tensors
    if any(isinstance(v, torch.Tensor) for v in state.values()):
        logger.info("  Treating file as raw state dict")
        return state

    raise ValueError(
        f"Cannot extract state dict. Available keys: {list(state.keys())}"
    )


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that always computes in float32 for numerical stability."""

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels):
    """Standard normalization layer: GroupNorm with 32 groups."""
    return GroupNorm32(32, channels)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters towards source using exponential moving average.

    Args:
        target_params: target parameter iterator
        source_params: source parameter iterator
        rate: EMA decay rate (closer to 1 = slower update)
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def checkpoint(func, inputs, params, flag):
    """
    Gradient checkpointing wrapper.

    When flag=True, intermediate activations are not cached during forward pass,
    reducing memory at the cost of recomputing them during backward pass.

    Args:
        func:   function to evaluate
        inputs: argument sequence for func
        params: parameters func depends on (not explicitly passed)
        flag:   if False, disable checkpointing
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [
            x.detach().requires_grad_(True) for x in ctx.input_tensors
        ]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
