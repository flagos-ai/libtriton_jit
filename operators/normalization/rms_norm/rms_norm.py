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
# rms_norm.py - RMS Normalization Triton Kernel (NPU-compatible)
# Based on FlagGems Ascend implementation with iterative variance computation
# ==============================================================================

import torch
import triton
from triton import language as tl
import math


@triton.jit
def rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMS normalization kernel (NPU-compatible).

    Uses iterative approach to compute variance, avoiding explicit tl.sum
    for the full reduction.
    """
    row_idx = tl.program_id(0)
    input_ptr = input_ptr + row_idx * input_row_stride
    output_ptr = output_ptr + row_idx * output_row_stride

    # Compute variance iteratively
    _var_base = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _var_base += x * x / n_cols  # Accumulate squared values

    # Sum across block for final variance
    var = tl.sum(_var_base, axis=0)
    rrms = 1.0 / tl.sqrt(var + eps)

    # Apply normalization and weight
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
        y = (x * rrms).to(output_ptr.dtype.element_ty) * w
        tl.store(output_ptr + cols, y, mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS Normalization (NPU-compatible).

    Args:
        x: Input tensor of any shape, normalized along last dimension
        weight: Weight tensor of shape [hidden_dim]
        eps: Small constant for numerical stability

    Returns:
        Normalized output tensor of same shape as input
    """
    assert x.size(-1) == weight.size(0), "Hidden dim must match weight size"

    orig_shape = x.shape
    x_flat = x.view(-1, x.size(-1)).contiguous()
    n_rows, n_cols = x_flat.shape

    output = torch.empty_like(x_flat)

    # Use block size that works well for NPU (max 8192)
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 8192)

    grid = (n_rows,)

    try:
        from torch.cuda import device as cuda_device
        with cuda_device(x.device):
            rms_norm_kernel[grid](
                x_flat, weight, output,
                x_flat.stride(0), output.stride(0),
                n_cols, eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
            )
    except:
        rms_norm_kernel[grid](
            x_flat, weight, output,
            x_flat.stride(0), output.stride(0),
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    return output.view(orig_shape)


if __name__ == "__main__":
    x = torch.randn(16, 1024, device="cuda")
    weight = torch.ones(1024, device="cuda")

    result = rms_norm(x, weight)

    # Reference implementation
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    expected = (x / rms) * weight

    torch.cuda.synchronize()
    print(f"Max diff: {(result - expected).abs().max().item()}")
    print(f"Match: {torch.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
