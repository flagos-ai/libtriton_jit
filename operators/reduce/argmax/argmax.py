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
# argmax.py - Argmax Reduction Triton Kernel (NPU-compatible)
# Based on FlagGems Ascend implementation with two-pass strategy
# ==============================================================================

import torch
import triton
from triton import language as tl
import math


@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """First pass: compute block-wise argmax values and indices."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=float('-inf'))

    # Get max value and index within this block
    max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
    max_index = max_index + pid * BLOCK_SIZE

    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, max_val)
    tl.store(max_index_ptr, max_index)


@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    """Second pass: reduce block results to final argmax."""
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=float('-inf'))
    index_val = tl.argmax(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


@triton.jit
def argmax_dim_kernel(
    inp,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute argmax along a specific dimension."""
    # set offset
    pid_m = tl.program_id(0)

    for pid_k in range(K):
        m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

        max_values = tl.full([BLOCK_M], dtype=tl.float32, value=float('-inf'))
        argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)

        for start_n in range(0, N, BLOCK_N):
            n_offset = start_n + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
            mask = (m_offset[:, None] < M) and (n_offset[None, :] < N)
            inp_ptrs = inp + offset
            inp_vals = tl.load(inp_ptrs, mask=mask, other=float('-inf'))

            local_max = tl.max(inp_vals, 1)
            local_argmax = tl.argmax(inp_vals, 1)

            # if return indices is not supported, call a tl.argmax in addition
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(update, start_n + local_argmax, argmax_values)

        offset_index = m_offset * K + pid_k
        out_index_ptrs = out_index + offset_index
        mask1 = m_offset < M
        tl.store(out_index_ptrs, argmax_values, mask=mask1)


def argmax(inp: torch.Tensor, dim: int = None, keepdim: bool = False) -> torch.Tensor:
    """Compute argmax along a dimension.

    Args:
        inp: Input tensor
        dim: Dimension to reduce. If None, reduces over all elements.
        keepdim: Whether to keep the reduced dimension.

    Returns:
        Tensor of indices of maximum values.
    """
    if dim is None:
        # Full tensor argmax
        M = inp.numel()
        dtype = inp.dtype

        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
        if keepdim:
            shape = [1] * inp.dim()
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        try:
            from torch.cuda import device as cuda_device
            with cuda_device(inp.device):
                argmax_kernel_1[(mid_size, 1, 1)](
                    inp, mid_value, mid_index, M, block_size
                )
                argmax_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
        except:
            argmax_kernel_1[(mid_size, 1, 1)](
                inp, mid_value, mid_index, M, block_size
            )
            argmax_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)

        return out

    # Dimensional argmax
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim]) if dim > 0 else 1
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
    if not keepdim:
        out_index = torch.squeeze(out_index, dim)

    BLOCK_M = 8
    BLOCK_N = 256

    def grid(meta):
        axis0 = triton.cdiv(M, meta["BLOCK_M"])
        axis0 = min(axis0, 4096)
        return (axis0,)

    try:
        from torch.cuda import device as cuda_device
        with cuda_device(inp.device):
            argmax_dim_kernel[grid](
                inp,
                out_index if keepdim else out_index.unsqueeze(dim),
                M, N, K,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
            )
    except:
        argmax_dim_kernel[grid](
            inp,
            out_index if keepdim else out_index.unsqueeze(dim),
            M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )

    return out_index


if __name__ == "__main__":
    x = torch.randn(16, 4 * 1024, device="cuda")
    result = argmax(x, dim=1)
    expected = torch.argmax(x, dim=1)

    torch.cuda.synchronize()
    print(f"Indices match: {torch.equal(result, expected)}")
