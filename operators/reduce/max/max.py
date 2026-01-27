import torch
import triton
from triton import language as tl


@triton.jit
def max_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute max reduction along the last dimension."""
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_ids < M

    # Initialize with minimum value
    acc = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        col_ids = off + tl.arange(0, BLOCK_N)
        col_mask = col_ids < N
        mask = row_mask[:, None] & col_mask[None, :]

        a = tl.load(in_ptr + row_ids[:, None] * N + col_ids, mask, other=float('-inf'))
        block_max = tl.max(a, axis=1)
        acc = tl.maximum(acc, block_max)

    tl.store(out_ptr + row_ids, acc, row_mask)


@triton.jit
def max_with_indices_kernel(
    in_ptr,
    out_vals_ptr,
    out_idx_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute max reduction with indices along the last dimension."""
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_ids < M

    max_vals = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    max_indices = tl.zeros([BLOCK_M], dtype=tl.int64)

    for off in range(0, N, BLOCK_N):
        col_ids = off + tl.arange(0, BLOCK_N)
        col_mask = col_ids < N
        mask = row_mask[:, None] & col_mask[None, :]

        a = tl.load(in_ptr + row_ids[:, None] * N + col_ids, mask, other=float('-inf'))

        # Find max in this block
        block_max = tl.max(a, axis=1)
        block_argmax = tl.argmax(a, axis=1) + off

        # Update global max
        update_mask = block_max > max_vals
        max_vals = tl.where(update_mask, block_max, max_vals)
        max_indices = tl.where(update_mask, block_argmax, max_indices)

    tl.store(out_vals_ptr + row_ids, max_vals, row_mask)
    tl.store(out_idx_ptr + row_ids, max_indices, row_mask)


def max_dim(inp: torch.Tensor, dim: int, keepdim: bool = False):
    """Compute max along a dimension."""
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

    BLOCK_M, BLOCK_N = 4, 512
    grid = (triton.cdiv(M, BLOCK_M),)

    with torch.cuda.device(inp.device):
        max_with_indices_kernel[grid](
            permuted.view(M, N), out_vals.view(M), out_idx.view(M),
            M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=8
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
