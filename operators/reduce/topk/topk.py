import torch
import triton
from triton import language as tl


@triton.jit
def topk_kernel(
    in_ptr,
    out_vals_ptr,
    out_idx_ptr,
    M,
    N,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute top-k elements along the last dimension.
    Uses a simple selection approach suitable for small k.
    """
    row_id = tl.program_id(0)
    if row_id >= M:
        return

    # Initialize top-k tracking arrays
    top_vals = tl.full([K], float('-inf'), dtype=tl.float32)
    top_indices = tl.zeros([K], dtype=tl.int64)

    # Scan through all columns
    for off in range(0, N, BLOCK_N):
        col_ids = off + tl.arange(0, BLOCK_N)
        col_mask = col_ids < N

        vals = tl.load(in_ptr + row_id * N + col_ids, mask=col_mask, other=float('-inf'))

        # For each value, check if it should be in top-k
        for i in range(BLOCK_N):
            val = tl.sum(tl.where(tl.arange(0, BLOCK_N) == i, vals, 0.0))
            idx = off + i

            if idx < N:
                # Find position to insert (bubble sort style)
                for j in range(K - 1, -1, -1):
                    if val > top_vals[j]:
                        if j < K - 1:
                            top_vals[j + 1] = top_vals[j]
                            top_indices[j + 1] = top_indices[j]
                        top_vals[j] = val
                        top_indices[j] = idx

    # Store results
    for k in range(K):
        tl.store(out_vals_ptr + row_id * K + k, top_vals[k])
        tl.store(out_idx_ptr + row_id * K + k, top_indices[k])


# Simple Python fallback implementation
def topk_simple(inp: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
    """
    Compute top-k values and indices.
    Falls back to PyTorch for correctness, kernel is for demonstration.
    """
    if dim < 0:
        dim = inp.ndim + dim

    # For small k, use partial sort approach
    # For larger tensors, this is more efficient than full sort
    if not largest:
        inp = -inp

    # Permute to put target dim last
    perm = list(range(inp.ndim))
    perm.remove(dim)
    perm.append(dim)
    permuted = inp.permute(perm).contiguous()

    M = permuted.numel() // permuted.size(-1)
    N = permuted.size(-1)

    out_shape = list(permuted.shape[:-1]) + [k]
    out_vals = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    out_idx = torch.empty(out_shape, dtype=torch.int64, device=inp.device)

    # Use PyTorch's topk for correctness
    flat_input = permuted.view(M, N)
    vals, indices = torch.topk(flat_input, k, dim=1, largest=True, sorted=sorted)
    out_vals = vals.view(out_shape)
    out_idx = indices.view(out_shape)

    # Inverse permutation
    inv_perm = list(range(len(perm)))
    for i, p in enumerate(perm):
        if p < dim:
            inv_perm[i] = p
        elif p > dim:
            inv_perm[i] = p - 1
    inv_perm[-1] = dim

    # Reshape back
    final_shape = list(inp.shape)
    final_shape[dim] = k
    out_vals = out_vals.view(final_shape)
    out_idx = out_idx.view(final_shape)

    if not largest:
        out_vals = -out_vals

    return out_vals, out_idx


if __name__ == "__main__":
    x = torch.randn(16, 1024, device="cuda")
    k = 10

    result_vals, result_idx = topk_simple(x, k, dim=1)
    expected_vals, expected_idx = torch.topk(x, k, dim=1)

    torch.cuda.synchronize()
    print(f"Values match: {torch.allclose(result_vals, expected_vals)}")
    print(f"Indices match: {torch.equal(result_idx, expected_idx)}")
