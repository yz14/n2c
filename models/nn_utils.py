"""
Neural network utilities adapted from guided-diffusion.

Provides building blocks for the UNet model:
  - GroupNorm32: GroupNorm with float32 computation
  - conv_nd / avg_pool_nd: dimension-agnostic conv and pooling
  - zero_module: zero-initialize module parameters
  - checkpoint: gradient checkpointing wrapper
"""

import torch
import torch.nn as nn


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
