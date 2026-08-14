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
# softmax.py - Softmax Triton Kernel (NPU-compatible)
# Based on FlagGems Ascend implementation with iterative reduction
# ==============================================================================

import torch
import triton
from triton import language as tl


@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    input_row_stride,
    output_row_stride,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """Softmax kernel for innermost dimension (K=1 case)."""
    pid_m = tl.program_id(0)

    input_ptr += pid_m * input_row_stride
    output_ptr += pid_m * output_row_stride

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(input_ptr + n_offsets, mask=mask, other=float("-inf"))
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = e / z
        tl.store(output_ptr + n_offsets, out, mask=mask)
    else:
        # Multi-pass for large N
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)

        # First pass: compute max and sum(exp) using online algorithm
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=float("-inf"))
            m_new = tl.maximum(m, inp)
            # Handle -inf case
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        # Second pass: compute softmax output
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=float("-inf"))
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)


@triton.jit
def softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """Softmax kernel for non-innermost dimension (K>1 case)."""
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(0)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=float("-inf"))
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        # First pass: online softmax algorithm
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=float("-inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)  # (TILE_K,)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
        m = m_reduced

        # Second pass: compute output
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=float("-inf"))
            o = tl.exp(inp - m[None, :]) / z[None, :]
            tl.store(output_ptr + offsets, o, mask=mask)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax along a dimension (NPU-compatible).

    Uses iterative online softmax algorithm that avoids explicit reduction ops.
    """
    if dim < 0:
        dim = x.ndim + dim

    assert dim >= 0 and dim < x.ndim, f"Invalid dim {dim}"

    # Compute M (product of dims before softmax dim), N (softmax dim), K (product of dims after)
    M = 1
    N = x.shape[dim]
    for i in range(dim):
        M *= x.shape[i]
    K = x.numel() // M // N

    x = x.contiguous()
    out = torch.empty_like(x)

    # Determine tile sizes
    TILE_N = min(triton.next_power_of_2(N), 4096)
    TILE_K = min(triton.next_power_of_2(K), 64) if K > 1 else 1
    ONE_TILE_PER_CTA = TILE_N >= N

    try:
        from torch.cuda import device as cuda_device
        device_ctx = cuda_device(x.device)
    except:
        device_ctx = None

    if K > 1:
        # Non-inner dimension case
        grid = (M, triton.cdiv(K, TILE_K), 1)

        if device_ctx:
            with device_ctx:
                softmax_kernel_non_inner[grid](
                    out, x, M, N, K,
                    TILE_N=TILE_N, TILE_K=TILE_K, ONE_TILE_PER_CTA=ONE_TILE_PER_CTA
                )
        else:
            softmax_kernel_non_inner[grid](
                out, x, M, N, K,
                TILE_N=TILE_N, TILE_K=TILE_K, ONE_TILE_PER_CTA=ONE_TILE_PER_CTA
            )
    else:
        # Inner dimension case - reshape to 2D for simplicity
        x_flat = x.view(M, N)
        out_flat = out.view(M, N)
        grid = (M, 1, 1)

        if device_ctx:
            with device_ctx:
                softmax_kernel_inner[grid](
                    out_flat, x_flat, M, N,
                    x_flat.stride(0), out_flat.stride(0),
                    TILE_N=TILE_N, ONE_TILE_PER_CTA=ONE_TILE_PER_CTA,
                    num_warps=4
                )
        else:
            softmax_kernel_inner[grid](
                out_flat, x_flat, M, N,
                x_flat.stride(0), out_flat.stride(0),
                TILE_N=TILE_N, ONE_TILE_PER_CTA=ONE_TILE_PER_CTA,
                num_warps=4
            )

    return out


if __name__ == "__main__":
    x = torch.randn(16, 1024, device="cuda")

    result = softmax(x, dim=-1)
    expected = torch.softmax(x, dim=-1)

    torch.cuda.synchronize()
    print(f"Max diff: {(result - expected).abs().max().item()}")
    print(f"Match: {torch.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
