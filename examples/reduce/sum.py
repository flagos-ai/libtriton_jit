# ==============================================================================
# sum.py - Sum Reduction Triton Kernel (NPU-compatible)
# Based on FlagGems Ascend implementation with two-pass strategy
# ==============================================================================

import math

import torch
import triton
from triton import language as tl


@triton.jit
def sum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """First pass: compute partial sums for each block."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)

    # Compute sum for this block
    block_sum = tl.sum(inp_val, axis=0)
    mid_ptr = mid + pid
    tl.store(mid_ptr, block_sum)


@triton.jit
def sum_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    """Second pass: reduce partial sums to final result."""
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0)
    final_sum = tl.sum(mid_val, axis=0)
    tl.store(out, final_sum)


@triton.jit
def sum_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute sum reduction along the last dimension (for dim reduction)."""
    # Map the program id to the row of inp it should compute
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

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask
            a = tl.load(ninp + cols, mask, other=0.0)
            acc += a

        row_sum = tl.sum(acc, axis=1)[:, None]
        tl.store(nout, row_sum, row_mask)


# ------ wrapper ------
_integer_dtypes = {
    torch.bool,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    """Sum reduction along specified dimensions."""
    if dtype is None:
        dtype = inp.dtype
        if dtype in _integer_dtypes:
            dtype = torch.int64

    # Handle full reduction case (no dim specified or all dims)
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype

        # Two-pass reduction for full tensor sum
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
        if keepdim:
            shape = [1] * inp.dim()
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        else:
            out = torch.empty([], dtype=dtype, device=inp.device)

        # Import backend-specific device context
        try:
            from torch.cuda import device as cuda_device

            device_ctx = cuda_device(inp.device)
        except ImportError:
            device_ctx = None

        if device_ctx:
            with device_ctx:
                sum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
                sum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
        else:
            sum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
            sum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)

        return out.to(dtype)

    if not dim:  # [] empty list
        raise ValueError("Cannot sum over an empty list of dimensions")

    if isinstance(dim, int):
        dim = [dim]

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)  # move reduction dim to the end and make it contiguous
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=dtype, device=inp.device)

    # Determine grid size (cap at 4096 for NPU)
    def grid(meta):
        axis0 = triton.cdiv(M, meta["BLOCK_M"])
        axis0 = min(axis0, 4096)
        return (axis0,)

    # Use device context if available
    try:
        from torch.cuda import device as cuda_device

        with cuda_device(inp.device):
            sum_dim_kernel[grid](inp, out, M, N, BLOCK_M=4, BLOCK_N=256)
    except ImportError:
        sum_dim_kernel[grid](inp, out, M, N, BLOCK_M=4, BLOCK_N=256)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


if __name__ == "__main__":
    torch_ops_my_ops_sum_dim = torch.library.custom_op(
        "my_ops::sum.dim_IntList",
        mutates_args=(),
        device_types="cuda",
        # the scheme should not include op name
        schema="(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
    )(sum_dim)
    x = torch.randn(16, 4 * 1024, device="cuda")
    result1 = sum_dim(x, [1])
    result2 = torch.sum(x, [1])
    result3 = torch.ops.my_ops.sum_dim_IntList(x, [1])

    torch.cuda.synchronize()
    for _ in range(10):
        torch.sum(x, [1])
    torch.cuda.synchronize()
    for _ in range(10):
        sum_dim(x, [1])
    torch.cuda.synchronize()
    for _ in range(10):
        torch.ops.my_ops.sum_dim_IntList(x, [1])
    torch.cuda.synchronize()
