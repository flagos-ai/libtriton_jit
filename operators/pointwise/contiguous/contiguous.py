import torch
import triton
from triton import language as tl


@triton.jit
def copy_1d_kernel(X, Out, n, BLOCK_N: tl.constexpr):
    """Simple 1D copy kernel for contiguous tensors."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    x = tl.load(X + offsets, mask=mask)
    tl.store(Out + offsets, x, mask=mask)


@triton.jit
def copy_strided_2d_kernel(
    X, Out,
    n_rows, n_cols,
    stride_x_row, stride_x_col,
    stride_out_row, stride_out_col,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """2D strided copy kernel for making tensors contiguous."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    row_offsets = row_start + tl.arange(0, BLOCK_M)
    col_offsets = col_start + tl.arange(0, BLOCK_N)

    # Create 2D masks
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols

    # Compute input indices
    x_ptrs = X + row_offsets[:, None] * stride_x_row + col_offsets[None, :] * stride_x_col
    mask = row_mask[:, None] & col_mask[None, :]

    # Load from strided input
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Compute output indices (contiguous)
    out_ptrs = Out + row_offsets[:, None] * stride_out_row + col_offsets[None, :] * stride_out_col

    # Store to contiguous output
    tl.store(out_ptrs, x, mask=mask)


def contiguous(x: torch.Tensor) -> torch.Tensor:
    """Make tensor contiguous using Triton kernel."""
    if x.is_contiguous():
        return x

    out = torch.empty(x.size(), dtype=x.dtype, device=x.device, memory_format=torch.contiguous_format)

    if x.dim() == 1 or x.numel() == 0:
        # 1D case
        n = x.numel()
        BLOCK_N = 1024
        grid = (triton.cdiv(n, BLOCK_N), 1, 1)
        with torch.cuda.device(x.device):
            copy_1d_kernel[grid](
                x.flatten(), out.flatten(), n,
                BLOCK_N=BLOCK_N, num_warps=8, num_stages=1
            )
    else:
        # 2D case (flatten higher dims into first two)
        x_2d = x.view(-1, x.size(-1))
        out_2d = out.view(-1, out.size(-1))

        n_rows, n_cols = x_2d.shape
        BLOCK_M, BLOCK_N = 32, 32

        grid = (triton.cdiv(n_rows, BLOCK_M), triton.cdiv(n_cols, BLOCK_N), 1)
        with torch.cuda.device(x.device):
            copy_strided_2d_kernel[grid](
                x_2d, out_2d,
                n_rows, n_cols,
                x_2d.stride(0), x_2d.stride(1),
                out_2d.stride(0), out_2d.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=1
            )

    return out


if __name__ == "__main__":
    # Test with non-contiguous tensor
    x = torch.randn(64, 128, device="cuda")
    x_nc = x.t()  # Make non-contiguous

    print(f"Original contiguous: {x.is_contiguous()}")
    print(f"Transposed contiguous: {x_nc.is_contiguous()}")

    result = contiguous(x_nc)
    expected = x_nc.contiguous()

    print(f"Result contiguous: {result.is_contiguous()}")
    print(f"Match: {torch.allclose(result, expected)}")
