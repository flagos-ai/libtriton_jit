# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ==============================================================================
# fused_add_rms_norm.py - Fused Add + RMS Normalization Triton Kernel
# Based on FlagGems Ascend implementation (NPU-compatible)
# ==============================================================================

import torch
import triton
from triton import language as tl
import math


@triton.jit
def fused_add_rms_norm_kernel(
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weight
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    """Fused add + RMS normalization kernel (NPU-compatible).

    Performs in-place:
    - residual = x + residual
    - x = rms_norm(residual) * weight

    Uses iterative approach compatible with NPU.
    """
    pid = tl.program_id(0)
    # Compute row base offset (avoid in-place pointer arithmetic for NPU compatibility)
    x_row_offset = pid * x_stride_r
    r_row_offset = pid * r_stride_r

    # First pass: compute variance
    _var_base = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + x_row_offset + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + r_row_offset + cols, mask, other=0.0).to(tl.float32)
        x += r
        _var_base += x * x / N

    var = tl.sum(_var_base)
    rrms = 1 / tl.sqrt(var + eps)

    # Second pass: apply normalization and write outputs
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + x_row_offset + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + r_row_offset + cols, mask, other=0.0).to(tl.float32)
        x += r
        w = tl.load(W + cols, mask, other=0.0)
        y = (x * rrms).to(X.dtype.element_ty) * w

        # Write back to residual and input
        tl.store(R + r_row_offset + cols * r_stride_c, x.to(R.dtype.element_ty), mask=mask)
        tl.store(X + x_row_offset + cols * x_stride_c, y, mask=mask)


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, eps: float = 1e-5):
    """Fused residual addition and RMS normalization (NPU-compatible).

    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified.

    Args:
        x: Input tensor (will be overwritten with normalized output)
        residual: Residual tensor (will be overwritten with x + residual)
        weight: Weight tensor for scaling
        eps: Small constant for numerical stability

    Returns:
        Tuple of (normalized_output, updated_residual) - same tensors as input but modified
    """
    assert x.shape == residual.shape, "Input and residual must have same shape"
    assert x.size(-1) == weight.size(0), "Hidden dim must match weight size"

    # Get normalized shape dimensions
    normalized_shape = [x.size(-1)]
    dim = x.ndim - len(normalized_shape)
    M = min(math.prod(x.shape[:dim]) if dim > 0 else 1, 65535)
    N = math.prod(normalized_shape)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 8192)

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    try:
        from torch.cuda import device as cuda_device
        with cuda_device(x.device):
            fused_add_rms_norm_kernel[M,](
                x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
            )
    except:
        fused_add_rms_norm_kernel[M,](
            x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
        )

    return x, residual


if __name__ == "__main__":
    x = torch.randn(16, 1024, device="cuda")
    residual = torch.randn(16, 1024, device="cuda")
    weight = torch.ones(1024, device="cuda")

    # Make copies for reference computation
    x_orig = x.clone()
    res_orig = residual.clone()

    output, res_out = fused_add_rms_norm(x, residual, weight)

    # Reference implementation
    x_add = x_orig + res_orig
    rms = torch.sqrt(x_add.pow(2).mean(-1, keepdim=True) + 1e-5)
    expected = (x_add / rms) * weight

    torch.cuda.synchronize()
    print(f"Output max diff: {(output - expected).abs().max().item()}")
    print(f"Residual max diff: {(res_out - x_add).abs().max().item()}")
    print(f"Match: {torch.allclose(output, expected, rtol=1e-4, atol=1e-4)}")
