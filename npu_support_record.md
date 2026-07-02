# Ascend NPU Support Record

This branch adapts Boogu-Image inference so the model can run on Ascend NPU while preserving the existing GPU path.

## Migration Notes

- Import `torch_npu` from the shared helper when available so PyTorch registers the `npu` backend before model placement.
- Treat FlashAttention and Triton kernels as CUDA-only unless the runtime explicitly targets CUDA.
- Prefer PyTorch SDPA on NPU, with guarded Ascend fused kernels only when tensor layout, dtype, and mask constraints are satisfied.
- Keep RoPE on the active NPU path in real-valued `(cos, sin)` form and avoid complex rotary tensors.

## Kernel And Import Changes

- Added `torch_npu` helpers in `boogu/utils/npu_utils.py` for NPU registration, availability checks, device checks, device setup, and generator fallback.
- Updated `boogu/utils/import_utils.py` so FlashAttention and Triton are disabled when NPU is requested, avoiding CUDA-only imports on Ascend.
- Added a direct guard in `boogu/ops/triton/layer_norm.py` so Triton layer norm cannot evaluate CUDA autotune decorators during NPU-targeted runs.
- Added guarded NPU kernel paths for RMSNorm, SwiGLU, RoPE, and optional fused attention, with PyTorch fallbacks for unsupported shapes, dtypes, or masks.
- Switched the active RoPE path to real-valued `(cos, sin)` tensors so NPU inference avoids complex RoPE tensors.
- Kept FlashAttention and Triton paths available for GPU runs when CUDA is explicitly targeted.

## Runtime Failure Guards

- Accepted `npu` and `npu:x` device strings in shared device validation.
- Importing `boogu.utils.npu_utils` attempts `torch_npu` registration when the package is available.
- Added NPU-safe generator creation with CPU fallback when backend generator support is unavailable.
- Added NPU device setup through `torch.npu.set_device(...)` when the selected device is NPU.
- Disabled diffusers group-offload streams on NPU because CUDA stream assumptions are unsafe there.
- Added NPU cache clearing through `torch_npu.npu.empty_cache()` or `torch.npu.empty_cache()` in pipeline cleanup paths.
- Made Cache-DiT an optional CUDA-only dependency path; NPU inference does not import it, and Cache-DiT caching is rejected on NPU.

## Documentation

- Updated `README.md` and `README_CN.md` to state that this branch supports Ascend NPU inference and to summarize the major NPU support changes.

