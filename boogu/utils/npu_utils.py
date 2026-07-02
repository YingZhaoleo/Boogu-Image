"""Small helpers for Ascend NPU inference paths."""

from __future__ import annotations

import importlib
import os
from typing import Optional

import torch


def import_torch_npu():
    """Import torch_npu when available so PyTorch registers the NPU backend."""
    try:
        return importlib.import_module("torch_npu")
    except ImportError:
        return None


def is_torch_npu_available() -> bool:
    import_torch_npu()
    return hasattr(torch, "npu") and torch.npu.is_available()


def _device_type(device) -> Optional[str]:
    if device is None:
        return None
    try:
        return torch.device(device).type
    except (RuntimeError, TypeError, ValueError):
        return str(device).strip().lower().split(":", 1)[0] or None


def is_npu_device(device: Optional[torch.device]) -> bool:
    return _device_type(device) == "npu"


def is_npu_requested(device: Optional[torch.device] = None) -> bool:
    """Return true when the runtime target is explicitly or implicitly Ascend NPU."""
    explicit_device_type = _device_type(device)
    if explicit_device_type == "npu":
        return True
    if explicit_device_type in {"cpu", "cuda"}:
        return False

    env_device_type = _device_type(os.environ.get("device"))
    if env_device_type == "npu":
        return True
    if env_device_type in {"cpu", "cuda"}:
        return False

    # If both CUDA and NPU are present, require an explicit NPU hint so GPU runs
    # keep CUDA-only kernels available during module import.
    if is_torch_npu_available() and not torch.cuda.is_available():
        return True

    return bool(os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))


def set_device_if_npu(device: torch.device) -> None:
    if not is_npu_device(device):
        return
    import_torch_npu()
    if hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
        torch.npu.set_device(device)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create a seeded generator, falling back to CPU if the backend lacks support."""
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except (RuntimeError, TypeError):
        return torch.Generator(device="cpu").manual_seed(seed)
