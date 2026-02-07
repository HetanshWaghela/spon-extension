"""
SPON Extensions - Core Module

Provides utilities for device-agnostic execution across CUDA, MPS (Apple Silicon), and CPU.
"""

import torch


def get_device(requested: str = "auto") -> torch.device:
    """
    Get the best available device.

    Priority: CUDA > MPS > CPU.
    If `requested` is not "auto", it is respected as-is (e.g. "cuda:0", "mps", "cpu").
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_default_dtype(device: torch.device) -> torch.dtype:
    """
    Get the recommended dtype for a device.

    - CUDA: bfloat16 (best precision/performance trade-off)
    - MPS:  float16 (bfloat16 support is incomplete on MPS)
    - CPU:  float32
    """
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def clear_memory(device: torch.device | None = None):
    """Clear GPU/accelerator memory cache."""
    if device is not None:
        dev_type = device.type
    else:
        dev_type = None

    if dev_type == "cuda" or (dev_type is None and torch.cuda.is_available()):
        torch.cuda.empty_cache()
    if dev_type == "mps" or (dev_type is None and torch.backends.mps.is_available()):
        torch.mps.empty_cache()
