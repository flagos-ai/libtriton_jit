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
# max.py - Max Reduction Triton Kernel (NPU-compatible)
# Based on FlagGems Ascend implementation with two-pass strategy
# ==============================================================================

import torch
import triton
from triton import language as tl
import math


def get_dtype_min(dtype):
    """Get minimum value for a dtype."""
    if dtype == tl.float16:
        return -65504.0
    elif dtype == tl.bfloat16:
        return -3.4028235e+38
    elif dtype == tl.float32:
        return -3.4028235e+38
    elif dtype == tl.float64:
        return -1.7976931348623157e+308
    elif dtype == tl.int32:
        return -2147483648
    elif dtype == tl.int64:
        return -9223372036854775808
    else:
        return float('-inf')


@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """First pass: compute block-wise max values."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    # Use -inf for masked values to not affect max
    inp_val = tl.load(inp_ptrs, mask=mask, other=float('-inf'))
    max_val = tl.max(inp_val, axis=0)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    """Second pass: reduce block maxes to final result."""
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=float('-inf'))
    max_val = tl.max(mid_val, axis=0)
    tl.store(out, max_val)


@triton.jit
def max_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute max reduction along the last dimension."""
    pid = tl.program_id(0)
    workers = tl.num_programs(0)

    total_workloads = tl.cdiv(M, BLOCK_M)
    workloads = tl.cdiv(total_workloads, workers)

    for w in range(workloads):
        work_id = pid + w * workers
        rows = work_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        ninp = inp + rows * N
        nout = out + rows
        row_mask = rows < M

        # Initialize with -inf
        acc = tl.full([BLOCK_M, BLOCK_N], value=float('-inf'), dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask
            a = tl.load(ninp + cols, mask, other=float('-inf'))
            acc = tl.maximum(acc, a)

        row_max = tl.max(acc, axis=1)[:, None]
        tl.store(nout, row_max, row_mask)


@triton.jit
def max_with_indices_kernel(
    inp,
    out_vals,
    out_idx,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute max with indices along the last dimension."""
    pid = tl.program_id(0)
    workers = tl.num_programs(0)

    total_workloads = tl.cdiv(M, BLOCK_M)
    workloads = tl.cdiv(total_workloads, workers)

    for w in range(workloads):
        work_id = pid + w * workers
        m_offset = work_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = m_offset < M

        max_values = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
        argmax_values = tl.full([BLOCK_M], value=0, dtype=tl.int64)

        for start_n in range(0, N, BLOCK_N):
            n_offset = start_n + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = row_mask[:, None] and (n_offset[None, :] < N)
            inp_ptrs = inp + offset
            inp_vals = tl.load(inp_ptrs, mask=mask, other=float('-inf'))

            # Get local max and argmax
            local_max = tl.max(inp_vals, 1)
            local_argmax = tl.argmax(inp_vals, 1)

            # Update global max
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(update, start_n + local_argmax, argmax_values)

        # Store results
        tl.store(out_vals + m_offset, max_values, mask=row_mask)
        tl.store(out_idx + m_offset, argmax_values, mask=row_mask)


def dim_compress(inp, dims):
    """Compress tensor along specified dimensions."""
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


def max_dim(inp: torch.Tensor, dim: int = None, keepdim: bool = False):
    """Compute max along a dimension.

    Returns:
        values: Max values
        indices: Indices of max values (only when dim is specified)
    """
    # Handle full tensor max (no dim specified)
    if dim is None:
        M = inp.numel()
        dtype = inp.dtype

        # Two-pass reduction for full tensor max
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        if keepdim:
            shape = [1] * inp.dim()
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        else:
            out = torch.empty([], dtype=dtype, device=inp.device)

        try:
            from torch.cuda import device as cuda_device
            with cuda_device(inp.device):
                max_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
                max_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
        except:
            max_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
            max_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)

        return out

    # Handle dimension-specific max
    if dim < 0:
        dim = inp.ndim + dim

    # Permute to put reduction dim last
    perm = list(range(inp.ndim))
    perm.remove(dim)
    perm.append(dim)
    permuted = inp.permute(perm).contiguous()

    M = permuted.numel() // permuted.size(-1)
    N = permuted.size(-1)

    out_shape = list(permuted.shape[:-1])
    out_vals = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    out_idx = torch.empty(out_shape, dtype=torch.int64, device=inp.device)

    BLOCK_M, BLOCK_N = 8, 256

    def grid(meta):
        axis0 = triton.cdiv(M, meta["BLOCK_M"])
        axis0 = min(axis0, 4096)
        return (axis0,)

    try:
        from torch.cuda import device as cuda_device
        with cuda_device(inp.device):
            max_with_indices_kernel[grid](
                permuted.view(M, N), out_vals.view(M), out_idx.view(M),
                M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
            )
    except:
        max_with_indices_kernel[grid](
            permuted.view(M, N), out_vals.view(M), out_idx.view(M),
            M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )

    # Unpermute result
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm[:-1]):
        inv_perm[p] = i
    out_vals = out_vals.permute(inv_perm)
    out_idx = out_idx.permute(inv_perm)

    if keepdim:
        out_vals = out_vals.unsqueeze(dim)
        out_idx = out_idx.unsqueeze(dim)

    return out_vals, out_idx


if __name__ == "__main__":
    x = torch.randn(16, 4 * 1024, device="cuda")
    result_vals, result_idx = max_dim(x, dim=1)
    expected_vals, expected_idx = torch.max(x, dim=1)

    torch.cuda.synchronize()
    print(f"Values match: {torch.allclose(result_vals, expected_vals)}")
    print(f"Indices match: {torch.equal(result_idx, expected_idx)}")
