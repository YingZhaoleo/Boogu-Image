import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.npu_utils import import_torch_npu, is_npu_device


def swiglu(x, y):
    torch_npu = import_torch_npu()
    if (
        torch_npu is not None
        and hasattr(torch_npu, "npu_swiglu")
        and is_npu_device(x.device)
        and x.device == y.device
        and x.shape == y.shape
        and x.dtype == y.dtype
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        try:
            return torch_npu.npu_swiglu(torch.cat((x, y), dim=-1).contiguous(), dim=-1)
        except (RuntimeError, TypeError, ValueError):
            pass

    return F.silu(x.float(), inplace=False).to(x.dtype) * y


class NpuRMSNorm(nn.Module):
    def __init__(
        self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        factory_kwargs = {"device": device, "dtype": dtype}
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        torch_npu = import_torch_npu()
        if (
            torch_npu is not None
            and hasattr(torch_npu, "npu_rms_norm")
            and self.weight is not None
            and is_npu_device(x.device)
            and 2 <= x.dim() <= 8
        ):
            eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
            try:
                return torch_npu.npu_rms_norm(x, self.weight, eps)[0]
            except (RuntimeError, TypeError, ValueError):
                pass

        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
