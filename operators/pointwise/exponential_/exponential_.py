import torch
import triton
from triton import language as tl


@triton.jit
def exponential_kernel(X, Out, lambd, n, BLOCK_N: tl.constexpr):
    """
    Generate exponentially distributed random numbers in-place.
    Uses inverse transform sampling: X ~ Exp(lambda) => X = -ln(U)/lambda where U ~ Uniform(0,1)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    # Load uniform random values
    u = tl.load(X + offsets, mask=mask)

    # Clamp to avoid log(0)
    u = tl.maximum(u, 1e-7)

    # Inverse transform: -ln(u) / lambda
    result = -tl.log(u) / lambd

    tl.store(Out + offsets, result, mask=mask)


def exponential_inplace(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Fill tensor with exponentially distributed random numbers (in-place)."""
    # First fill with uniform random numbers
    x.uniform_(0, 1)

    n = x.numel()
    BLOCK_N = 1024
    grid = (triton.cdiv(n, BLOCK_N), 1, 1)
    with torch.cuda.device(x.device):
        exponential_kernel[grid](x, x, lambd, n, BLOCK_N=BLOCK_N, num_warps=8, num_stages=1)
    return x


if __name__ == "__main__":
    N = 128 * 1024
    x = torch.randn(N, device="cuda")

    result = exponential_inplace(x.clone(), lambd=1.0)
    torch.cuda.synchronize()

    print(f"Result[0:5]: {result[0:5]}")
    print(f"Mean (expected ~1.0 for lambda=1): {result.mean().item():.4f}")
    print(f"Min: {result.min().item():.4f}")
    print(f"All positive: {(result > 0).all().item()}")
