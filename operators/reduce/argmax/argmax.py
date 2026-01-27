import torch
import triton
from triton import language as tl


@triton.jit
def argmax_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute argmax along the last dimension."""
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

    tl.store(out_ptr + row_ids, max_indices, row_mask)


def argmax(inp: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Compute argmax along a dimension."""
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
    out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)

    BLOCK_M, BLOCK_N = 4, 512
    grid = (triton.cdiv(M, BLOCK_M),)

    with torch.cuda.device(inp.device):
        argmax_kernel[grid](
            permuted.view(M, N), out.view(M),
            M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=8
        )

    # Inverse permute
    inv_perm = [0] * (len(perm) - 1)
    for i, p in enumerate(perm[:-1]):
        inv_perm[p] = i
    out = out.permute(inv_perm)

    if keepdim:
        out = out.unsqueeze(dim)

    return out


if __name__ == "__main__":
    x = torch.randn(16, 4 * 1024, device="cuda")
    result = argmax(x, dim=1)
    expected = torch.argmax(x, dim=1)

    torch.cuda.synchronize()
    print(f"Indices match: {torch.equal(result, expected)}")
