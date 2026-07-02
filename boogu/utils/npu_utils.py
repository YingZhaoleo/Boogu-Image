"""Small helpers for Ascend NPU inference paths."""

from __future__ import annotations

import importlib
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


def is_npu_device(device: Optional[torch.device]) -> bool:
    return device is not None and torch.device(device).type == "npu"
