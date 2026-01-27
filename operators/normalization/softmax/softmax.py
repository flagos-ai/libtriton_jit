import torch
import triton
from triton import language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Online softmax kernel - numerically stable."""
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Load row with masking
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)

    # Subtract max and exponentiate
    numerator = tl.exp(row - row_max)

    # Sum for normalization
    denominator = tl.sum(numerator, axis=0)

    # Normalize
    softmax_output = numerator / denominator

    # Store
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax along a dimension."""
    if dim < 0:
        dim = x.ndim + dim

    # Permute to put softmax dim last
    perm = list(range(x.ndim))
    perm.remove(dim)
    perm.append(dim)
    x_permuted = x.permute(perm).contiguous()

    # Flatten batch dims
    orig_shape = x_permuted.shape
    n_rows = x_permuted.numel() // x_permuted.size(-1)
    n_cols = x_permuted.size(-1)
    x_flat = x_permuted.view(n_rows, n_cols)

    # Allocate output
    output = torch.empty_like(x_flat)

    # Compute block size (next power of 2 >= n_cols)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Launch kernel
    grid = (n_rows,)
    with torch.cuda.device(x.device):
        softmax_kernel[grid](
            x_flat, output,
            x_flat.stride(0), output.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    # Reshape and inverse permute
    output = output.view(orig_shape)
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    output = output.permute(inv_perm)

    return output


if __name__ == "__main__":
    x = torch.randn(16, 1024, device="cuda")

    result = softmax(x, dim=-1)
    expected = torch.softmax(x, dim=-1)

    torch.cuda.synchronize()
    print(f"Max diff: {(result - expected).abs().max().item()}")
    print(f"Match: {torch.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
