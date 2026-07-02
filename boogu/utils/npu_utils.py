"""Small helpers for Ascend NPU inference paths."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = torch_npu.npu.is_available()
except ImportError:
    torch_npu = None
    NPU_AVAILABLE = False


def import_torch_npu():
    """Import torch_npu when available so PyTorch registers the NPU backend."""
    return torch_npu


def is_torch_npu_available() -> bool:
    return NPU_AVAILABLE


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

    return is_torch_npu_available()


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
